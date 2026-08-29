import argparse
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


@dataclass
class BallEvent:
    frame_index: int
    track_id: int
    centroid: tuple[float, float]
    velocity: float = 0.0
    landing_point: tuple[float, float] | None = None
    line_call: str = "UNKNOWN"
    timestamp: float = field(default_factory=time.time)


class CourtMapper:
    """Simple homography-based court calibration for a single camera.

    The default mapping assumes the full image is treated as a court rectangle in normalized coordinates.
    If a real court is visible, the user can pass four image corners using --court-corners to calibrate it.
    """

    def __init__(self, src_points=None, dst_points=None):
        if src_points is None:
            src_points = np.array([[0, 0], [1920, 0], [1920, 1080], [0, 1080]], dtype=np.float32)

        if dst_points is None:
            dst_points = np.array([[0, 0], [20, 0], [20, 44], [0, 44]], dtype=np.float32)

        self.src_points = np.array(src_points, dtype=np.float32)
        self.dst_points = np.array(dst_points, dtype=np.float32)
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
        device: str | None = None,
        target_class_id: int | None = 32,
        max_history: int = 30,
        court_mapper: CourtMapper | None = None,
        target_fps: int = 120,
        target_width: int = 1920,
        target_height: int = 1080,
    ) -> None:
        self.model = YOLO(model_name)
        if device:
            self.model.to(device)

        self.tracker_config = tracker_config
        self.conf = conf
        self.iou = iou
        self.target_class_id = target_class_id
        self.max_history = max_history
        self.track_history = defaultdict(list)
        self.events: list[BallEvent] = []
        self.frame_index = 0
        self.court_mapper = court_mapper or CourtMapper()
        self.target_fps = target_fps
        self.target_width = target_width
        self.target_height = target_height

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
        """Very lightweight placeholder for bounce/landing estimation.
        For the draft, we detect when the ball direction changes sharply or the speed drops.
        """
        if len(track_points) < 5:
            return None

        recent = track_points[-5:]
        x_vals = [p[0] for p in recent]
        y_vals = [p[1] for p in recent]

        dx = np.diff(x_vals)
        dy = np.diff(y_vals)

        direction_change = np.sum(np.sign(dx[:-1]) != np.sign(dx[1:])) > 0 or np.sum(np.sign(dy[:-1]) != np.sign(dy[1:])) > 0
        speed_drop = self._estimate_velocity(recent) < 4.0

        if direction_change or speed_drop:
            return recent[-1]

        return None

    def _classify_in_out(self, court_point):
        """Use homography-based court mapping to assign a line-call result."""
        if court_point is None:
            return "UNKNOWN"

        return self.court_mapper.classify(court_point)

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

        results = self.model.track(
            frame,
            persist=True,
            tracker=self.tracker_config,
            conf=self.conf,
            iou=self.iou,
            verbose=False,
        )

        if not results or len(results) == 0:
            return frame

        result = results[0]
        if result.boxes is None or result.boxes.id is None:
            return frame

        if self.target_class_id is not None:
            cls_ids = result.boxes.cls.int().cpu().tolist()
            valid_indices = [i for i, cls_id in enumerate(cls_ids) if cls_id == self.target_class_id]
            if not valid_indices:
                return frame

            filtered_boxes = result.boxes.xyxy.cpu().numpy()[valid_indices]
            filtered_ids = result.boxes.id.int().cpu().tolist()[valid_indices]
        else:
            filtered_boxes = result.boxes.xyxy.cpu().numpy()
            filtered_ids = result.boxes.id.int().cpu().tolist()

        self._draw_tracking(frame, filtered_boxes, filtered_ids)
        return frame

    def run_video(self, source, output_path=None, show_window=True):
        source_path = Path(source)
        video_source = str(source_path) if source_path.exists() else str(source)

        cap = cv2.VideoCapture(video_source)
        if not cap.isOpened():
            raise FileNotFoundError(f"Unable to open source: {source}")

        # Configure USB camera for high-speed capture (ELP 120fps camera optimization)
        if isinstance(video_source, int):  # USB camera device
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.target_width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.target_height)
            cap.set(cv2.CAP_PROP_FPS, self.target_fps)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce buffer for lower latency
            print(f"[Camera Config] Requesting {self.target_width}x{self.target_height} @ {self.target_fps}fps")

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
        print(f"[Camera Actual] {width}x{height} @ {fps}fps")

        writer = None
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            writer = cv2.VideoWriter(
                str(output_path),
                cv2.VideoWriter_fourcc(*"mp4v"),
                fps,
                (width, height),
            )

        try:
            while cap.isOpened():
                success, frame = cap.read()
                if not success:
                    break

                annotated = self.process_frame(frame)

                if writer is not None:
                    writer.write(annotated)

                if show_window:
                    cv2.imshow("Project PickleVision - Single Camera Draft", annotated)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
        finally:
            cap.release()
            if writer is not None:
                writer.release()
            if show_window:
                cv2.destroyAllWindows()

        return self.events


def parse_args():
    parser = argparse.ArgumentParser(description="Project PickleVision: single-camera YOLOv8 tracking prototype for ELP 120fps USB camera")
    parser.add_argument("--source", type=str, default="0", help="Video file path or USB camera index (default: 0 for ELP camera)")
    parser.add_argument("--model", type=str, default="yolov8n.pt", help="YOLOv8 model to load")
    parser.add_argument("--tracker", type=str, default="bytetrack.yaml", help="Tracking configuration")
    parser.add_argument("--conf", type=float, default=0.25, help="Detection confidence threshold")
    parser.add_argument("--iou", type=float, default=0.5, help="IoU threshold for NMS")
    parser.add_argument("--output", type=str, default=None, help="Optional annotated output video path")
    parser.add_argument("--show", action="store_true", default=True, help="Display annotated frames in real-time")
    parser.add_argument("--fps", type=int, default=120, help="Target camera FPS (default: 120 for ELP camera)")
    parser.add_argument("--width", type=int, default=1920, help="Target camera width in pixels")
    parser.add_argument("--height", type=int, default=1080, help="Target camera height in pixels")
    parser.add_argument("--court-corners", type=str, default=None, help="Court calibration: x1 y1 x2 y2 x3 y3 x4 y4 (TL TR BR BL)")
    return parser.parse_args()


def main():
    args = parse_args()

    court_points = CourtMapper.parse_corners(args.court_corners) if args.court_corners else None
    court_mapper = CourtMapper(src_points=court_points) if court_points is not None else None

    tracker = PickleVisionTracker(
        model_name=args.model,
        tracker_config=args.tracker,
        conf=args.conf,
        iou=args.iou,
        court_mapper=court_mapper,
        target_fps=args.fps,
        target_width=args.width,
        target_height=args.height,
    )

    source = args.source
    if not Path(source).exists() and source.isdigit():
        source = int(source)

    events = tracker.run_video(source=source, output_path=args.output, show_window=args.show)
    print(f"\n[Summary] Tracked {len(events)} candidate ball events.")


if __name__ == "__main__":
    main()