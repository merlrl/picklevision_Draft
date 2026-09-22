import argparse
import os
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

# Without this, cv2.VideoCapture(...) on Windows can take 50-60s to open a USB
# camera because MSMF negotiates hardware color-conversion transforms for every
# advertised mode before returning. Must be set before any VideoCapture call.
os.environ.setdefault("OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS", "0")

import cv2
import numpy as np
import supervision as sv
import torch
from inference import get_model
from inference_sdk import InferenceHTTPClient

try:
    from ultralytics import YOLO
except ModuleNotFoundError:
    YOLO = None


def _open_camera(index_or_path):
    """Open a video source, using the MSMF backend for integer camera indices on Windows.

    Explicit CAP_MSMF avoids DSHOW's "can't capture by index" failure on some builds
    and, combined with OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS=0 above, opens in
    under a second instead of 50-60s.
    """
    if isinstance(index_or_path, int) and os.name == "nt":
        return cv2.VideoCapture(index_or_path, cv2.CAP_MSMF)
    return cv2.VideoCapture(index_or_path)


@dataclass
class BallEvent:
    frame_index: int
    track_id: int
    centroid: tuple[float, float]
    velocity: float = 0.0
    landing_point: tuple[float, float] | None = None
    line_call: str = "UNKNOWN"
    timestamp: float = field(default_factory=time.time)
    # Tags which camera produced this event. A future 2nd-camera setup runs one
    # PickleVisionTracker per camera (each with its own CourtMapper.src_points
    # but the SAME real-world dst_points), then fuses their events by matching
    # timestamp/frame_index across camera_id -- e.g. preferring whichever camera
    # has a non-occluded detection for a given moment.
    camera_id: str = "cam0"


class CourtMapper:
    """Simple homography-based court calibration for a single camera.

    The default mapping assumes the full image is treated as a court rectangle in normalized coordinates.
    If a real court is visible, the user can pass four image corners using --court-corners to calibrate it.
    """

    def __init__(self, src_points=None, dst_points=None, court_width=20.0, court_length=44.0):
        if src_points is None:
            src_points = np.array([[0, 0], [1920, 0], [1920, 1080], [0, 1080]], dtype=np.float32)

        if dst_points is None:
            # court_length=44 covers the full court; pass 22 to scope calibration to
            # just one half (baseline to net) -- useful when only one half is
            # reliably visible/accurate from a given camera angle.
            dst_points = np.array(
                [[0, 0], [court_width, 0], [court_width, court_length], [0, court_length]], dtype=np.float32
            )

        self.src_points = np.array(src_points, dtype=np.float32)
        self.dst_points = np.array(dst_points, dtype=np.float32)
        self.court_width = court_width
        self.court_length = court_length
        self.h_matrix = cv2.getPerspectiveTransform(self.src_points, self.dst_points)

    @staticmethod
    def parse_corners(raw_value):
        if raw_value is None:
            return None

        values = [float(v.strip()) for v in str(raw_value).replace(";", " ").split() if v.strip()]
        if len(values) != 8:
            raise ValueError(
                "--court-corners must contain exactly 8 values: "
                "x1,y1,x2,y2,x3,y3,x4,y4 or space-separated values."
            )

        points = np.array(
            [
                [values[0], values[1]],
                [values[2], values[3]],
                [values[4], values[5]],
                [values[6], values[7]],
            ],
            dtype=np.float32,
        )
        return points

    def map_point(self, point):
        if point is None:
            return None

        src_point = np.array([[point[0], point[1]]], dtype=np.float32)
        mapped = cv2.perspectiveTransform(src_point[None, :, :], self.h_matrix)[0][0]
        return float(mapped[0]), float(mapped[1])

    def classify(self, point):
        if point is None:
            return "UNKNOWN"

        mapped_point = self.map_point(point)
        if mapped_point is None:
            return "UNKNOWN"

        x, y = mapped_point
        x_min = float(np.min(self.dst_points[:, 0]))
        x_max = float(np.max(self.dst_points[:, 0]))
        y_min = float(np.min(self.dst_points[:, 1]))
        y_max = float(np.max(self.dst_points[:, 1]))

        if x_min <= x <= x_max and y_min <= y <= y_max:
            return "IN"
        return "OUT"


class PickleVisionTracker:
    """Single-camera draft for Project PickleVision using YOLOv8 detection + tracking."""

    def __init__(
        self,
        model_name: str = "yolov8n.pt",
        tracker_config: str = "bytetrack.yaml",
        conf: float = 0.25,
        iou: float = 0.5,
        imgsz: int = 640,
        device: str | None = None,
        target_class_id: int | None = 32,
        max_history: int = 30,
        court_mapper: CourtMapper | None = None,
        target_fps: int = 120,
        target_width: int = 1920,
        target_height: int = 1080,
        max_missed_frames: int = 15,
        max_match_distance: float = 250.0,
        ball_color_lower: tuple[int, int, int] = (25, 60, 60),
        ball_color_upper: tuple[int, int, int] = (45, 255, 255),
        ball_color_min_ratio: float = 0.20,
        min_aspect_ratio: float = 0.6,
        min_box_dimension: int = 20,
        require_ball_color: bool = True,
        camera_id: str = "cam0",
        zoom_to_court: bool = False,
        zoom_padding: float = 0.15,
        use_roboflow: bool = False,
        roboflow_api_url: str = "http://localhost:9001",
        roboflow_api_key: str | None = None,
        roboflow_workspace_name: str | None = None,
        roboflow_model_id: str | None = None,
        roboflow_workflow_id: str | None = None,
    ) -> None:
        # Two mutually exclusive detection backends: a local Ultralytics model
        # (default), or a Roboflow model/Workflow running on a self-hosted
        # inference server (for a custom-trained model when weights export isn't
        # available on the account's plan). Everything downstream of "get
        # boxes+confs for this frame" -- single-ball lock-on, prediction, color
        # filter, court mapping, drawing -- is identical either way.
        #
        # roboflow_model_id (e.g. "project-slug/3") calls the model directly and
        # is preferred: it's a plain, well-documented response schema, and it
        # references an exact version. roboflow_workflow_id is a fallback for
        # when a Workflow's own custom logic is actually needed -- note a
        # Workflow's "Project" block pins its own model version independently of
        # whatever is set as "Current Model" on the project's Deployments page,
        # so it can silently keep serving an old version after retraining.
        self.use_roboflow = use_roboflow
        if self.use_roboflow:
            self.model = None
            self.device = "roboflow-server"
            self.roboflow_client = InferenceHTTPClient.init(api_url=roboflow_api_url, api_key=roboflow_api_key)
            self.roboflow_workspace_name = roboflow_workspace_name
            self.roboflow_model_id = roboflow_model_id
            self.roboflow_workflow_id = roboflow_workflow_id
            self._roboflow_next_id = 0
        else:
            if YOLO is None:
                raise RuntimeError(
                    "Ultralytics is not installed in this environment. "
                    "Install it with: pip install ultralytics"
                )
            self.model = YOLO(model_name)
            self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
            self.model.to(self.device)

        self.tracker_config = tracker_config
        self.conf = conf
        self.iou = iou
        self.imgsz = imgsz
        self.target_class_id = target_class_id
        self.max_history = max_history
        self.track_history = defaultdict(list)
        self.events: list[BallEvent] = []
        self.frame_index = 0
        self.court_mapper = court_mapper or CourtMapper()
        self.target_fps = target_fps
        self.target_width = target_width
        self.target_height = target_height
        self.camera_id = camera_id

        # "Digital zoom": crop detection/display to the calibrated court region so
        # the same imgsz budget is spent entirely on the area that matters, instead
        # of also feeding the model background/ceiling/etc. Only meaningful with a
        # real calibrated court_mapper (--court-corners), not the default full-frame
        # fallback. zoom_roi is (x1, y1, x2, y2) in original frame pixels, resolved
        # lazily against the actual frame/camera size the first time it's needed.
        self.zoom_to_court = zoom_to_court and court_mapper is not None
        self.zoom_padding = zoom_padding
        self.zoom_roi: tuple[int, int, int, int] | None = None

        # Single-ball lock-on state (used only when target_class_id is set).
        # Keeps the tracker following one ball instead of every round object YOLO
        # finds, and bridges brief motion-blur misses instead of dropping the
        # track on the first missed frame.
        self.max_missed_frames = max_missed_frames
        self.max_match_distance = max_match_distance
        self.primary_track_id: int | None = None
        self.primary_trajectory: list[tuple[float, float]] = []
        self.missed_frames = 0

        # Color check (HSV) to distinguish the pickleball's optic yellow-green from
        # other round objects YOLO's generic "sports ball" class also matches.
        self.ball_color_lower = np.array(ball_color_lower, dtype=np.uint8)
        self.ball_color_upper = np.array(ball_color_upper, dtype=np.uint8)
        self.ball_color_min_ratio = ball_color_min_ratio
        self.require_ball_color = require_ball_color

        # A pickleball's dimples/holes mean even a correct full-ball box is never
        # near-100% solid ball-color, so color alone can't reliably reject a small
        # false positive that happens to sit on a solid-colored patch (a shirt
        # logo, or -- ironically -- a gap between the ball's own holes). Size and
        # shape are more robust: both observed failure modes (a shirt graphic, a
        # single hole) produced anomalously small/non-square boxes vs. a real ball.
        self.min_aspect_ratio = min_aspect_ratio
        self.min_box_dimension = min_box_dimension

    def _get_ball_centroid(self, box):
        x1, y1, x2, y2 = map(float, box)
        center_x = (x1 + x2) / 2.0
        center_y = (y1 + y2) / 2.0
        return center_x, center_y

    def _estimate_velocity(self, track_points):
        if len(track_points) < 2:
            return 0.0

        last_two = track_points[-2:]
        dx = last_two[1][0] - last_two[0][0]
        dy = last_two[1][1] - last_two[0][1]
        distance = np.hypot(dx, dy)
        return float(distance)

    def _detect_ball_contact(self, track_points):
        """Detects a bounce by looking for a V-shape change in the Y-axis.
        
        Physics: A true court bounce is defined by the ball moving downward (increasing Y)
        then suddenly moving upward (decreasing Y). X-axis changes are ignored because they
        represent spin, curvature, or lateral movement, not bounces.
        """
        if len(track_points) < 5:
            return None

        recent = track_points[-5:]
        y_vals = [p[1] for p in recent]

        # Calculate the change in Y (dy)
        dy = np.diff(y_vals)

        # If the ball was going down (+dy) and suddenly goes up (-dy), it bounced
        direction_change_y = np.sum(np.sign(dy[:-1]) != np.sign(dy[1:])) > 0

        # Speed drop is a fallback for when a ball rolls or loses momentum near the boundary
        speed_drop = self._estimate_velocity(recent) < 4.0

        if direction_change_y or speed_drop:
            return recent[-1]

        return None

    def _classify_in_out(self, court_point):
        """Use homography-based court mapping to assign a line-call result.

        `court_point` is in whatever frame `process_frame` is currently operating
        on -- when zoomed, that's the cropped region, so it's offset back to
        original-frame coordinates first to match the calibrated homography.
        """
        if court_point is None:
            return "UNKNOWN"

        if self.zoom_roi is not None:
            offset_x, offset_y, _, _ = self.zoom_roi
            court_point = (court_point[0] + offset_x, court_point[1] + offset_y)

        return self.court_mapper.classify(court_point)

    def _compute_zoom_roi(self, frame_width, frame_height):
        """Bounding box (with padding) around the calibrated court corners, in
        original frame pixels -- the region `process_frame` crops to when
        `zoom_to_court` is enabled.
        """
        xs = self.court_mapper.src_points[:, 0]
        ys = self.court_mapper.src_points[:, 1]
        x_min, x_max = float(np.min(xs)), float(np.max(xs))
        y_min, y_max = float(np.min(ys)), float(np.max(ys))
        pad_x = (x_max - x_min) * self.zoom_padding
        pad_y = (y_max - y_min) * self.zoom_padding

        x1 = max(0, int(x_min - pad_x))
        y1 = max(0, int(y_min - pad_y))
        x2 = min(frame_width, int(x_max + pad_x))
        y2 = min(frame_height, int(y_max + pad_y))
        if x2 <= x1 or y2 <= y1:
            return None
        return (x1, y1, x2, y2)

    def _ball_color_ratio(self, frame, box):
        """Fraction of pixels inside `box` matching the pickleball's optic yellow-green.

        Used to tell the real ball apart from other round objects of a different
        color that YOLO's generic "sports ball" class also matches.
        """
        x1, y1, x2, y2 = (int(v) for v in box)
        x1, y1 = max(x1, 0), max(y1, 0)
        x2, y2 = min(x2, frame.shape[1]), min(y2, frame.shape[0])
        if x2 <= x1 or y2 <= y1:
            return 0.0

        crop = frame[y1:y2, x1:x2]
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.ball_color_lower, self.ball_color_upper)
        return float(np.count_nonzero(mask)) / mask.size

    def _is_plausible_ball_shape(self, box):
        """Reject boxes too small or too non-square to plausibly be a round ball.

        Catches false positives color alone can't: a small logo patch or a single
        dimple on the ball's own surface can be just as solidly ball-colored as a
        genuine ball, but tends to be either much smaller than a real detection or
        an odd (non-square) shape, since a round object's box should be roughly 1:1.
        """
        x1, y1, x2, y2 = box
        w, h = x2 - x1, y2 - y1
        if min(w, h) < self.min_box_dimension:
            return False
        aspect_ratio = min(w, h) / max(w, h) if max(w, h) > 0 else 0
        return aspect_ratio >= self.min_aspect_ratio

    def _narrow_candidate_pool(self, frame, boxes):
        """Progressively filter candidates by shape, then color, falling back to a
        looser stage whenever a stricter one eliminates everything -- so a real
        ball lacking a strong color match (e.g. mostly in shadow) still gets
        considered via shape alone, rather than the filter silently doing nothing.
        """
        all_indices = np.arange(len(boxes))
        shape_matches = np.array([i for i in all_indices if self._is_plausible_ball_shape(boxes[i])])
        pool = shape_matches if len(shape_matches) else all_indices

        if self.require_ball_color:
            color_ratios = np.array([self._ball_color_ratio(frame, boxes[i]) for i in pool])
            color_matches = pool[color_ratios >= self.ball_color_min_ratio]
            if len(color_matches):
                pool = color_matches

        return pool

    def _select_primary_detection(self, frame, boxes, ids, confs):
        """Pick exactly one detection to follow so a stray round object never hijacks the ball ID.

        Once locked on, the same track ID is kept as long as it keeps appearing. If
        ByteTrack assigns a new ID after a brief loss (e.g. re-detecting post-blur),
        the closest candidate to the ball's last known position is adopted instead,
        provided it's within `max_match_distance` -- otherwise it's treated as a
        different object and ignored. Candidates matching the pickleball's shape
        and optic yellow-green color are preferred over same-class objects that don't.
        """
        if not ids:
            return None

        if self.primary_track_id in ids:
            idx = ids.index(self.primary_track_id)
            return boxes[idx], ids[idx]

        candidate_pool = self._narrow_candidate_pool(frame, boxes)

        if self.primary_trajectory:
            last_point = np.array(self.primary_trajectory[-1])
            centroids = np.array([self._get_ball_centroid(b) for b in boxes])
            distances = np.linalg.norm(centroids - last_point, axis=1)
            best_idx = int(candidate_pool[np.argmin(distances[candidate_pool])])
            if distances[best_idx] <= self.max_match_distance:
                print(f"[Ball Lock] Re-acquired via position match: box={boxes[best_idx]}, distance={distances[best_idx]:.0f}px")
                return boxes[best_idx], ids[best_idx]
            return None

        # No active lock yet: acquire whichever plausible-shaped, ball-colored
        # detection the model is most confident about.
        if confs is not None and len(confs):
            best_idx = int(candidate_pool[np.argmax(confs[candidate_pool])])
        else:
            best_idx = int(candidate_pool[0])
        box = boxes[best_idx]
        print(f"[Ball Lock] Acquired: box={box}, conf={confs[best_idx] if confs is not None and len(confs) else 'n/a'}, color_ratio={self._ball_color_ratio(frame, box):.2f}")
        return box, ids[best_idx]

    def _predict_primary_position(self):
        """Extrapolate the ball's position for a few frames using its last known velocity.

        Bridges brief detection gaps (typically motion blur at high ball speed) so the
        trajectory and bounce logic don't reset on every single missed frame.
        """
        if len(self.primary_trajectory) < 2 or self.missed_frames > self.max_missed_frames:
            return None

        (x1, y1), (x2, y2) = self.primary_trajectory[-2], self.primary_trajectory[-1]
        return (2 * x2 - x1, 2 * y2 - y1)

    def _handle_missed_detection(self, frame):
        self.missed_frames += 1
        predicted_point = self._predict_primary_position()
        if predicted_point is None:
            self.primary_track_id = None
            self.primary_trajectory = []
            return frame

        px, py = predicted_point
        predicted_box = (px - 10, py - 10, px + 10, py + 10)
        self._draw_primary_tracking(frame, predicted_box, self.primary_track_id, predicted=True)
        return frame

    def _draw_primary_tracking(self, frame, box, track_id, predicted=False):
        x1, y1, x2, y2 = box
        center_x, center_y = self._get_ball_centroid(box)

        self.primary_trajectory.append((center_x, center_y))
        if len(self.primary_trajectory) > self.max_history:
            self.primary_trajectory.pop(0)

        velocity = self._estimate_velocity(self.primary_trajectory)
        landing_point = self._detect_ball_contact(self.primary_trajectory)
        if landing_point is not None and not predicted:
            call = self._classify_in_out(landing_point)
            self.events.append(
                BallEvent(
                    frame_index=self.frame_index,
                    track_id=track_id,
                    centroid=(center_x, center_y),
                    velocity=velocity,
                    landing_point=landing_point,
                    line_call=call,
                    camera_id=self.camera_id,
                )
            )

        box_color = (0, 165, 255) if predicted else (0, 255, 0)
        label = f"Ball ID: {track_id}" + (" (predicted)" if predicted else "")
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), box_color, 2)
        cv2.putText(
            frame,
            label,
            (int(x1), max(0, int(y1) - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            box_color,
            2,
        )

        trajectory = self.primary_trajectory
        for i in range(1, len(trajectory)):
            cv2.line(
                frame,
                (int(trajectory[i - 1][0]), int(trajectory[i - 1][1])),
                (int(trajectory[i][0]), int(trajectory[i][1])),
                (0, 0, 255),
                2,
            )

        if landing_point is not None:
            px, py = map(int, landing_point)
            cv2.circle(frame, (px, py), 6, (255, 0, 0), -1)
            call = self._classify_in_out(landing_point)
            cv2.putText(frame, f"{call}", (px + 8, py - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    def _draw_tracking(self, frame, boxes, track_ids):
        detections = []

        for box, track_id in zip(boxes, track_ids):
            x1, y1, x2, y2 = box
            center_x, center_y = self._get_ball_centroid(box)
            track = self.track_history[track_id]
            track.append((center_x, center_y))

            if len(track) > self.max_history:
                track.pop(0)

            velocity = self._estimate_velocity(track)
            landing_point = self._detect_ball_contact(track)
            if landing_point is not None:
                call = self._classify_in_out(landing_point)
                self.events.append(
                    BallEvent(
                        frame_index=self.frame_index,
                        track_id=track_id,
                        centroid=(center_x, center_y),
                        velocity=velocity,
                        landing_point=landing_point,
                        line_call=call,
                        camera_id=self.camera_id,
                    )
                )

            detections.append((track_id, (x1, y1, x2, y2), center_x, center_y, velocity, landing_point))

            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(
                frame,
                f"ID: {track_id}",
                (int(x1), max(0, int(y1) - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2,
            )

            if len(track) > 1:
                for i in range(1, len(track)):
                    cv2.line(frame, (int(track[i - 1][0]), int(track[i - 1][1])), (int(track[i][0]), int(track[i][1])), (0, 0, 255), 2)

            if landing_point is not None:
                px, py = map(int, landing_point)
                cv2.circle(frame, (px, py), 6, (255, 0, 0), -1)
                call = self._classify_in_out(landing_point)
                cv2.putText(frame, f"{call}", (px + 8, py - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        return detections

    def process_frame(self, frame):
        self.frame_index += 1

        if self.zoom_to_court and self.zoom_roi is None:
            self.zoom_roi = self._compute_zoom_roi(frame.shape[1], frame.shape[0])

        detect_frame = frame
        if self.zoom_roi is not None:
            zx1, zy1, zx2, zy2 = self.zoom_roi
            detect_frame = frame[zy1:zy2, zx1:zx2]

        if self.use_roboflow:
            return self._process_frame_roboflow(detect_frame)
        return self._process_frame_ultralytics(detect_frame)

    def _fetch_roboflow_predictions(self, detect_frame):
        """Get the raw predictions list, preferring a direct model_id call.

        Direct model inference (self.roboflow_model_id, e.g. "project-slug/3")
        references an exact trained version and returns a plain, flat schema.
        The Workflow path (self.roboflow_workflow_id) is a fallback for when a
        Workflow's own custom logic is genuinely needed -- but a Workflow's
        "Project" block pins its own model version independently of whatever is
        set as "Current Model" on the Deployments page, so after retraining it
        can silently keep serving a stale version even though the UI looks updated.
        """
        if self.roboflow_model_id:
            result = self.roboflow_client.infer(detect_frame, model_id=self.roboflow_model_id)
            return result.get("predictions", [])

        result = self.roboflow_client.run_workflow(
            workspace_name=self.roboflow_workspace_name,
            workflow_id=self.roboflow_workflow_id,
            images={"image": detect_frame},
        )
        return result[0].get("predictions", {}).get("predictions", []) if result else []

    def _process_frame_roboflow(self, detect_frame):
        """Detect via Roboflow on the self-hosted inference server, instead of a
        local Ultralytics model. Roboflow returns fresh per-frame detections
        with no persistent track ID (unlike ByteTrack), so every detection gets
        a synthetic ID from a monotonically increasing counter -- never reused
        across frames, so `_select_primary_detection`'s "same ID still present"
        fast path can never falsely match an unrelated detection that happens to
        land at the same list position. Cross-frame continuity instead comes
        entirely from its position-matching against self.primary_trajectory,
        same as it would for a lost-then-reacquired ByteTrack ID.
        """
        predictions = self._fetch_roboflow_predictions(detect_frame)
        # self.conf is applied here explicitly -- Roboflow's own internal
        # threshold is a separate setting configured in its UI, not something
        # this call controls, so low-confidence noise isn't otherwise
        # guaranteed to be filtered out.
        predictions = [p for p in predictions if p.get("confidence", 0.0) >= self.conf]
        if not predictions:
            return self._handle_missed_detection(detect_frame)

        boxes = []
        confs = []
        for pred in predictions:
            cx, cy, w, h = pred["x"], pred["y"], pred["width"], pred["height"]
            boxes.append((cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2))
            confs.append(pred.get("confidence", 0.0))
        boxes = np.array(boxes)
        confs = np.array(confs)

        ids = list(range(self._roboflow_next_id, self._roboflow_next_id + len(boxes)))
        self._roboflow_next_id += len(boxes)

        selection = self._select_primary_detection(detect_frame, boxes, ids, confs)
        if selection is None:
            return self._handle_missed_detection(detect_frame)

        box, track_id = selection
        self.primary_track_id = track_id
        self.missed_frames = 0
        self._draw_primary_tracking(detect_frame, box, track_id, predicted=False)
        return detect_frame

    def _process_frame_ultralytics(self, detect_frame):
        results = self.model.track(
            detect_frame,
            persist=True,
            tracker=self.tracker_config,
            conf=self.conf,
            iou=self.iou,
            imgsz=self.imgsz,
            verbose=False,
        )

        if not results or len(results) == 0:
            return self._handle_missed_detection(detect_frame) if self.target_class_id is not None else detect_frame

        result = results[0]
        if result.boxes is None or result.boxes.id is None:
            return self._handle_missed_detection(detect_frame) if self.target_class_id is not None else detect_frame

        if self.target_class_id is not None:
            cls_ids = result.boxes.cls.int().cpu().tolist()
            valid_indices = [i for i, cls_id in enumerate(cls_ids) if cls_id == self.target_class_id]
            if not valid_indices:
                return self._handle_missed_detection(detect_frame)

            filtered_boxes = result.boxes.xyxy.cpu().numpy()[valid_indices]
            filtered_ids = result.boxes.id.int().cpu().numpy()[valid_indices].tolist()
            filtered_confs = result.boxes.conf.cpu().numpy()[valid_indices]

            selection = self._select_primary_detection(detect_frame, filtered_boxes, filtered_ids, filtered_confs)
            if selection is None:
                return self._handle_missed_detection(detect_frame)

            box, track_id = selection
            self.primary_track_id = track_id
            self.missed_frames = 0
            self._draw_primary_tracking(detect_frame, box, track_id, predicted=False)
            return detect_frame

        filtered_boxes = result.boxes.xyxy.cpu().numpy()
        filtered_ids = result.boxes.id.int().cpu().tolist()
        self._draw_tracking(detect_frame, filtered_boxes, filtered_ids)
        return detect_frame

    def run_video(self, source, output_path=None, show_window=True, raw_output_path=None, record_fps=None, flip_horizontal=False, flip_vertical=False, exposure=None):
        if isinstance(source, Path):
            video_source = str(source)
        elif isinstance(source, int):
            video_source = source
        elif isinstance(source, str):
            source_path = Path(source)
            if source_path.exists():
                video_source = str(source_path)
            elif source.isdigit():
                video_source = int(source)
            else:
                video_source = source
        else:
            raise TypeError(f"Unsupported source type: {type(source)}")

        cap = _open_camera(video_source)
        if not cap.isOpened():
            if isinstance(video_source, int):
                candidates = [idx for idx in range(0, 6) if idx != video_source]
                for idx in candidates:
                    print(f"[Camera Fallback] Source {video_source} failed; trying camera index {idx} instead.")
                    cap = _open_camera(idx)
                    if cap.isOpened():
                        video_source = idx
                        print(f"[Camera Fallback] Successfully opened camera index {idx}.")
                        break
                if not cap.isOpened():
                    raise FileNotFoundError(f"Unable to open source: {source}")
            else:
                raise FileNotFoundError(f"Unable to open source: {source}")

        # Configure USB camera for high-speed capture (ELP 120fps camera optimization)
        if isinstance(video_source, int):  # USB camera device
            # CRITICAL: Force MJPEG codec to achieve 120fps over USB 2.0
            # Without this, raw YUY2 format will throttle to 5-10fps due to bandwidth limits
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))

            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.target_width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.target_height)
            cap.set(cv2.CAP_PROP_FPS, self.target_fps)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce buffer for lower latency
            print(f"[Camera Config] Requesting {self.target_width}x{self.target_height} @ {self.target_fps}fps (MJPEG)")

            if exposure is not None:
                # EXPERIMENTAL: manual exposure to shorten shutter time and reduce
                # motion blur, at the cost of a darker image needing more light.
                # Verified on the ELP camera this project targets: cap.get() always
                # reads back a fixed value regardless of what was actually set (a
                # driver quirk, not a bug here) -- but the real captured brightness
                # DOES change measurably and monotonically with the requested value,
                # confirmed by direct brightness sampling. Don't trust the printed
                # "reports" values as ground truth; judge by the actual image instead.
                cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
                cap.set(cv2.CAP_PROP_EXPOSURE, exposure)
                print(f"[Camera Config] Requested manual exposure={exposure} (this camera's .get() readback is unreliable -- judge by the actual image brightness/blur, not the reported value)")

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
        print(f"[Camera Actual] {width}x{height} @ {fps}fps")

        # Recording writes video-encode (CPU) work on top of detection (GPU) work --
        # a high capture fps (120) is valuable for detection/motion-blur, but encoding
        # two full-resolution streams at that same rate is real, sustained CPU cost
        # that can starve the detection loop. Default the SAVED file to a lower fps
        # than capture; every frame is still detected on, just not every one written.
        effective_record_fps = record_fps if record_fps else min(fps, 30)
        if (output_path or raw_output_path) and effective_record_fps < fps:
            print(f"[Recording] Capturing/detecting at {fps}fps, writing video at {effective_record_fps}fps")

        if self.zoom_to_court and self.zoom_roi is None:
            self.zoom_roi = self._compute_zoom_roi(width, height)

        output_width, output_height = width, height
        if self.zoom_roi is not None:
            zx1, zy1, zx2, zy2 = self.zoom_roi
            output_width, output_height = zx2 - zx1, zy2 - zy1
            print(f"[Zoom] Cropping to calibrated court region: {output_width}x{output_height} (from ({zx1},{zy1}) to ({zx2},{zy2}))")

        writer = None
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            writer = cv2.VideoWriter(
                str(output_path),
                cv2.VideoWriter_fourcc(*"mp4v"),
                effective_record_fps,
                (output_width, output_height),
            )

        # Separate from `writer`: saves the frame BEFORE any boxes/labels/trajectory
        # lines are drawn on it, so this file is safe to upload to Roboflow for
        # annotation/training. `writer` above bakes the overlay in permanently and
        # is only meant for reviewing/demoing tracking results, not as training data.
        raw_writer = None
        if raw_output_path:
            raw_output_path = Path(raw_output_path)
            raw_output_path.parent.mkdir(parents=True, exist_ok=True)
            raw_writer = cv2.VideoWriter(
                str(raw_output_path),
                cv2.VideoWriter_fourcc(*"mp4v"),
                effective_record_fps,
                (width, height),
            )
            print(f"[Raw Recording] Saving unannotated footage to {raw_output_path} (Roboflow-ready)")

        window_name = "Project PickleVision - Single Camera Draft"
        if show_window:
            # WINDOW_NORMAL makes it resizable/draggable; the initial size is just a
            # display cap so a 1920x1080 capture doesn't overflow a laptop screen --
            # recording and detection still use the full captured (or zoomed) resolution.
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            display_width = min(output_width, 1280)
            display_height = int(display_width * output_height / output_width) if output_width else output_height
            cv2.resizeWindow(window_name, display_width, display_height)

        # Frame-pace the recording to wall-clock time instead of writing one frame
        # per loop iteration. Per-frame processing (detection/tracking/drawing) is
        # usually slower than the camera's nominal fps, so a naive 1-write-per-loop
        # would compress a longer real session into fewer frames than `fps` implies,
        # playing back sped-up/timelapsed. Duplicating the latest frame to catch up
        # to elapsed real time keeps recorded duration matching real duration.
        # Capped per-iteration, or falling far enough behind (e.g. the requested fps
        # is unrealistic for this camera/hardware) turns into a write-storm that
        # falls further behind with every duplicate write, freezing the app.
        record_start = time.time()
        frames_written = 0
        max_catchup_frames_per_iteration = max(1, int(effective_record_fps))

        try:
            while cap.isOpened():
                success, frame = cap.read()
                if not success:
                    break

                if flip_horizontal or flip_vertical:
                    flip_code = -1 if (flip_horizontal and flip_vertical) else (1 if flip_horizontal else 0)
                    frame = cv2.flip(frame, flip_code)

                # Must copy before process_frame() -- it draws directly onto the
                # array it's given (or, with zoom, onto a view sharing memory with
                # this same frame), so anything not copied first ends up annotated too.
                raw_frame = frame.copy() if raw_writer is not None else None

                annotated = self.process_frame(frame)

                if writer is not None or raw_writer is not None:
                    expected_frames = int((time.time() - record_start) * effective_record_fps)
                    catchup_target = min(expected_frames, frames_written + max_catchup_frames_per_iteration)
                    while frames_written <= catchup_target:
                        if writer is not None:
                            writer.write(annotated)
                        if raw_writer is not None:
                            raw_writer.write(raw_frame)
                        frames_written += 1

                if show_window:
                    cv2.imshow(window_name, annotated)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
        finally:
            cap.release()
            if writer is not None:
                writer.release()
            if raw_writer is not None:
                raw_writer.release()
            if show_window:
                cv2.destroyAllWindows()

        return self.events


def parse_args():
    parser = argparse.ArgumentParser(description="Project PickleVision: single-camera YOLOv8 tracking prototype for ELP 120fps USB camera")
    parser.add_argument("--source", type=str, default="0", help="Video file path or USB camera index (default: 0)")
    parser.add_argument("--model", type=str, default="yolov8n.pt", help="YOLOv8 model to load")
    parser.add_argument("--tracker", type=str, default="bytetrack.yaml", help="Tracking configuration")
    parser.add_argument("--conf", type=float, default=0.25, help="Detection confidence threshold")
    parser.add_argument("--target-class-id", type=int, default=32, help="Class ID to track (default 32 = COCO 'sports ball', for yolov8n.pt). A custom single-class Roboflow model almost always uses class 0 instead -- pass --target-class-id 0 when using one. Use -1 to track every detected class")
    parser.add_argument("--use-roboflow", action="store_true", help="Detect via a Roboflow Workflow on a self-hosted inference server instead of a local Ultralytics model -- for a custom-trained model when weights export isn't available on the Roboflow plan")
    parser.add_argument("--roboflow-api-url", type=str, default="http://localhost:9001", help="Self-hosted Roboflow inference server URL")
    parser.add_argument("--roboflow-api-key", type=str, default="ZceKVfYE1Cvm0jqDdA1F", help="Roboflow API key")
    parser.add_argument("--roboflow-workspace", type=str, default="franzs-workspace-utuz0", help="Roboflow workspace name")
    parser.add_argument("--roboflow-model-id", type=str, default="pickleball-prototype/3", help="Roboflow model ID as 'project-slug/version' -- calls the model directly (preferred: exact version, simple response). Pass an empty string to use --roboflow-workflow-id instead")
    parser.add_argument("--roboflow-workflow-id", type=str, default=None, help="Roboflow workflow ID -- only used if --roboflow-model-id is empty. Note a Workflow's model reference is pinned separately from the project's 'Current Model' setting and can silently lag behind after retraining")
    parser.add_argument("--iou", type=float, default=0.5, help="IoU threshold for NMS")
    parser.add_argument("--output", type=str, default=None, help="Optional annotated output video path (boxes/labels/trajectory baked in -- for review, not training)")
    parser.add_argument("--raw-output", type=str, default=None, help="Optional unannotated output video path, safe to upload to Roboflow for annotation/training")
    parser.add_argument("--record-fps", type=int, default=None, help="FPS to WRITE recorded video at (default: min(camera fps, 30)). Capture/detection still runs at full --fps; only the saved file rate is lowered, since encoding two full-res streams at 120fps is heavy CPU work")
    parser.add_argument("--show", action="store_true", default=True, help="Display annotated frames in real-time")
    parser.add_argument("--fps", type=int, default=30, help="Target camera FPS (default: 30; use 120 for ELP camera)")
    parser.add_argument("--width", type=int, default=640, help="Target camera width in pixels (default: 640; use 1920 for ELP camera)")
    parser.add_argument("--height", type=int, default=480, help="Target camera height in pixels (default: 480; use 1080 for ELP camera)")
    parser.add_argument("--court-corners", type=str, default=None, help="Court calibration: x1 y1 x2 y2 x3 y3 x4 y4 (TL TR BR BL)")
    parser.add_argument("--max-missed-frames", type=int, default=15, help="Frames to keep extrapolating the ball's position through a detection gap (e.g. motion blur) before dropping the track")
    parser.add_argument("--match-distance", type=float, default=250.0, help="Max pixel distance a new detection can be from the ball's last known position to be accepted as the same ball")
    parser.add_argument("--no-color-filter", action="store_true", help="Disable the optic yellow-green color check used to prefer the real ball over other round objects")
    parser.add_argument("--ball-color-min-ratio", type=float, default=0.20, help="Minimum fraction of a candidate box that must be ball-colored to pass the color filter (default 0.20 -- kept fairly loose since the ball's own dimples/holes mean even a correct box is never near-100%% solid color). Lower this if the real ball is being rejected; raise it if other yellow-ish objects (logos, skin, etc.) are being mistaken for the ball")
    parser.add_argument("--min-box-dimension", type=int, default=20, help="Reject candidate boxes smaller than this (pixels, in the detection frame) as implausibly small to be the actual ball -- catches things like a single dimple/hole on the ball's own surface")
    parser.add_argument("--min-aspect-ratio", type=float, default=0.6, help="Reject candidate boxes whose width:height ratio isn't roughly square (min(w,h)/max(w,h) below this) -- a round ball's box should be close to 1:1")
    parser.add_argument("--device", type=str, default=None, help="Inference device: 'cuda', 'cpu', or omit to auto-detect GPU")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference resolution the model resizes frames to (default 640). Raise to 960-1280 to detect a small/far-away ball better, at the cost of speed")
    parser.add_argument(
        "--calibrate",
        action="store_true",
        help="Interactively click the court's 4 corners on the live feed to generate a --court-corners string, then exit without tracking",
    )
    parser.add_argument("--calibration-output", type=str, default=None, help="Optional file path to save the calibrated --court-corners string to")
    parser.add_argument("--flip-horizontal", action="store_true", help="Flip the camera feed horizontally (fixes a mirrored image, e.g. left/right reversed) before detection, display, and recording")
    parser.add_argument("--flip-vertical", action="store_true", help="Flip the camera feed vertically (e.g. if the camera is mounted upside down) before detection, display, and recording")
    parser.add_argument("--exposure", type=float, default=None, help="EXPERIMENTAL: force manual camera exposure to reduce motion blur (lower = shorter shutter = less blur, but needs more light). Effect is highly camera/driver-dependent -- try values like -6, -8, -10")
    parser.add_argument("--court-length", type=float, default=44.0, help="Real-world length (ft) of the calibrated region: 44 for a full court, 22 to scope to just one half (baseline to net)")
    parser.add_argument("--court-width", type=float, default=20.0, help="Real-world width (ft) of the calibrated region (default 20, standard doubles court width)")
    parser.add_argument("--zoom", action="store_true", help="Digitally zoom: crop detection/display/recording to the calibrated court region (requires --court-corners)")
    parser.add_argument("--zoom-padding", type=float, default=0.15, help="Padding around the calibrated court corners when zoomed, as a fraction of the court's width/height (default 0.15)")
    return parser.parse_args()


def run_local_inference_example(model_id: str = "your-model-id", image_path: str = "path/to/image.jpg"):
    """Example Roboflow local inference call using the inference package."""
    model = get_model(model_id=model_id)
    results = model.infer(image_path)
    print(results)
    return results


def run_supervision_visualization_example(model_id: str = "rfdetr-medium", image_url: str = "https://media.roboflow.com/inference/people-walking.jpg"):
    """Example Roboflow inference + supervision visualization."""
    image = sv.load_image_from_url(image_url)

    model = get_model(model_id=model_id)
    results = model.infer(image)[0]

    detections = sv.Detections.from_inference(results)

    annotated_image = sv.BoxAnnotator().annotate(scene=image, detections=detections)
    annotated_image = sv.LabelAnnotator().annotate(scene=annotated_image, detections=detections)

    sv.plot_image(annotated_image)
    return annotated_image


def run_elp_self_hosted_inference(
    camera_index: int = 0,
    api_url: str = "http://localhost:9001",
    api_key: str = "ZceKVfYE1Cvm0jqDdA1F",
    workspace_name: str = "franzs-workspace-utuz0",
    workflow_id: str = "pickleball-prototype-vpickleball-prototype-1-yolo11n-t1-logic",
    target_width: int = 1920,
    target_height: int = 1080,
    target_fps: int = 120,
):
    """Open the ELP USB camera and run each frame through a Roboflow Workflow deployment.

    This targets a Roboflow *Workflow* (not a raw model), so it calls
    `run_workflow(workspace_name, workflow_id, ...)` rather than `infer(model_id=...)`.
    """
    client = InferenceHTTPClient.init(
        api_url=api_url,
        api_key=api_key,
    )

    cap = _open_camera(camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera index {camera_index}")

    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, target_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, target_height)
    cap.set(cv2.CAP_PROP_FPS, target_fps)

    print(f"[Camera] Opened index {camera_index}")
    print(f"[Camera] Requested: {target_width}x{target_height} @ {target_fps}fps")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to read frame from camera.")
                break

            results = client.run_workflow(
                workspace_name=workspace_name,
                workflow_id=workflow_id,
                images={"image": frame},
            )
            print(results)

            cv2.imshow("ELP Camera + Roboflow", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()

    return None


def _court_reference_lines(mapper: CourtMapper):
    """Standard court reference lines (baselines, sidelines, net, kitchen, centerline),
    projected from real-world court coordinates back into image space via the
    inverse homography -- used to visually sanity-check a calibration.
    """
    x_min = float(np.min(mapper.dst_points[:, 0]))
    x_max = float(np.max(mapper.dst_points[:, 0]))
    y_min = float(np.min(mapper.dst_points[:, 1]))
    y_max = float(np.max(mapper.dst_points[:, 1]))
    width = x_max - x_min
    center_x = x_min + width / 2.0

    segments = [
        ((x_min, y_min), (x_max, y_min)),  # near edge
        ((x_min, y_max), (x_max, y_max)),  # far edge
        ((x_min, y_min), (x_min, y_max)),  # sideline
        ((x_max, y_min), (x_max, y_max)),  # sideline
    ]

    if abs(mapper.court_length - 44.0) < 1.0:
        # Full-court calibration: net sits at the midpoint, kitchen 7ft each side.
        net_y = y_min + mapper.court_length / 2.0
        kitchen_near_y = net_y - 7.0
        kitchen_far_y = net_y + 7.0
        segments += [
            ((x_min, net_y), (x_max, net_y)),
            ((x_min, kitchen_near_y), (x_max, kitchen_near_y)),
            ((x_min, kitchen_far_y), (x_max, kitchen_far_y)),
            ((center_x, y_min), (center_x, kitchen_near_y)),
            ((center_x, kitchen_far_y), (center_x, y_max)),
        ]
    else:
        # Half-court calibration (baseline -> net): the far edge IS the net.
        net_y = y_max
        kitchen_y = net_y - 7.0
        segments += [
            ((x_min, net_y), (x_max, net_y)),
            ((x_min, kitchen_y), (x_max, kitchen_y)),
            ((center_x, y_min), (center_x, kitchen_y)),
        ]

    endpoints = np.array(segments, dtype=np.float32).reshape(-1, 1, 2)
    inv_h = np.linalg.inv(mapper.h_matrix)
    mapped = cv2.perspectiveTransform(endpoints, inv_h).reshape(-1, 2, 2)
    return [(tuple(pair[0]), tuple(pair[1])) for pair in mapped]


def calibrate_court_corners(source, save_path: str | None = None, court_width: float = 20.0, court_length: float = 44.0, flip_horizontal=False, flip_vertical=False):
    """Interactively click the court's 4 real-world corners on a live camera feed.

    Click order matters -- it must match CourtMapper's default destination
    rectangle (TOP-LEFT, TOP-RIGHT, BOTTOM-RIGHT, BOTTOM-LEFT, i.e. clockwise
    starting from whichever corner you treat as the origin).

    For a full court (court_length=44), TOP = far baseline, BOTTOM = near
    baseline (the one closest to the camera). For a half-court calibration
    (court_length=22, baseline-to-net only -- useful when the far half of the
    court isn't reliably visible from this camera angle), TOP = the net line,
    BOTTOM = the near baseline.

    Press 's' to save once 4 points are placed, 'r' to reset, 'q' to cancel.

    Returns the "x1 y1 x2 y2 x3 y3 x4 y4" string ready for --court-corners, or
    None if cancelled.
    """
    video_source = int(source) if isinstance(source, str) and source.isdigit() else source
    cap = _open_camera(video_source)
    if not cap.isOpened():
        raise FileNotFoundError(f"Unable to open source: {source}")

    clicked: list[tuple[int, int]] = []

    def on_click(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(clicked) < 4:
            clicked.append((x, y))

    window_name = "Court Calibration"
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, on_click)

    far_edge_label = "the NET line" if abs(court_length - 44.0) >= 1.0 else "the FAR baseline"
    print(f"Calibrating a {court_width:.0f}x{court_length:.0f} ft region.")
    print(f"Click in order: TOP-LEFT, TOP-RIGHT ({far_edge_label}), then BOTTOM-RIGHT, BOTTOM-LEFT (the NEAR baseline).")
    print("Keys: 'r' reset points | 's' save (once 4 are placed) | 'q' cancel")

    corner_str = None
    try:
        while True:
            success, frame = cap.read()
            if not success:
                print("Failed to read frame from camera.")
                break

            if flip_horizontal or flip_vertical:
                flip_code = -1 if (flip_horizontal and flip_vertical) else (1 if flip_horizontal else 0)
                frame = cv2.flip(frame, flip_code)

            display = frame.copy()
            for i, pt in enumerate(clicked):
                cv2.circle(display, pt, 6, (0, 0, 255), -1)
                cv2.putText(display, str(i + 1), (pt[0] + 8, pt[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            if len(clicked) == 4:
                pts = np.array(clicked, dtype=np.int32)
                cv2.polylines(display, [pts], isClosed=True, color=(0, 255, 0), thickness=2)
                cv2.putText(display, "Press 's' to save, 'r' to reset", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                try:
                    preview_mapper = CourtMapper(
                        src_points=np.array(clicked, dtype=np.float32),
                        court_width=court_width,
                        court_length=court_length,
                    )
                    for (lx1, ly1), (lx2, ly2) in _court_reference_lines(preview_mapper):
                        cv2.line(display, (int(lx1), int(ly1)), (int(lx2), int(ly2)), (255, 255, 0), 1)
                    cv2.putText(
                        display,
                        "Cyan = predicted net/kitchen/sidelines -- should align with the real lines",
                        (10, display.shape[0] - 15),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (255, 255, 0),
                        1,
                    )
                except np.linalg.LinAlgError:
                    cv2.putText(
                        display,
                        "Corners too close together / collinear -- press 'r' and reclick",
                        (10, display.shape[0] - 15),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 0, 255),
                        1,
                    )
            else:
                cv2.putText(display, f"Click corner {len(clicked) + 1}/4", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            cv2.imshow(window_name, display)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("r"):
                clicked.clear()
            if key == ord("s") and len(clicked) == 4:
                corner_str = " ".join(f"{x} {y}" for x, y in clicked)
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()

    if corner_str is None:
        print("Calibration cancelled -- no corners saved.")
        return None

    print(f"\n--court-corners \"{corner_str}\"")
    if save_path:
        Path(save_path).write_text(corner_str)
        print(f"Saved to {save_path}")

    return corner_str


def main():
    args = parse_args()

    source = args.source
    if isinstance(source, str) and source.isdigit():
        source = int(source)

    if args.calibrate:
        calibrate_court_corners(
            source,
            save_path=args.calibration_output,
            court_width=args.court_width,
            court_length=args.court_length,
            flip_horizontal=args.flip_horizontal,
            flip_vertical=args.flip_vertical,
        )
        return

    court_points = CourtMapper.parse_corners(args.court_corners) if args.court_corners else None
    court_mapper = (
        CourtMapper(src_points=court_points, court_width=args.court_width, court_length=args.court_length)
        if court_points is not None
        else None
    )

    tracker = PickleVisionTracker(
        model_name=args.model,
        tracker_config=args.tracker,
        conf=args.conf,
        iou=args.iou,
        court_mapper=court_mapper,
        target_fps=args.fps,
        target_width=args.width,
        target_height=args.height,
        max_missed_frames=args.max_missed_frames,
        max_match_distance=args.match_distance,
        require_ball_color=not args.no_color_filter,
        ball_color_min_ratio=args.ball_color_min_ratio,
        min_box_dimension=args.min_box_dimension,
        min_aspect_ratio=args.min_aspect_ratio,
        device=args.device,
        imgsz=args.imgsz,
        zoom_to_court=args.zoom,
        zoom_padding=args.zoom_padding,
        target_class_id=None if args.target_class_id == -1 else args.target_class_id,
        use_roboflow=args.use_roboflow,
        roboflow_api_url=args.roboflow_api_url,
        roboflow_api_key=args.roboflow_api_key,
        roboflow_workspace_name=args.roboflow_workspace,
        roboflow_model_id=args.roboflow_model_id,
        roboflow_workflow_id=args.roboflow_workflow_id,
    )
    print(f"[Device] Running inference on: {tracker.device}")

    events = tracker.run_video(
        source=source,
        output_path=args.output,
        show_window=args.show,
        raw_output_path=args.raw_output,
        record_fps=args.record_fps,
        flip_horizontal=args.flip_horizontal,
        flip_vertical=args.flip_vertical,
        exposure=args.exposure,
    )
    print(f"\n[Summary] Tracked {len(events)} candidate ball events.")


if __name__ == "__main__":
    main()