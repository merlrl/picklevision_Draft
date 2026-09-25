import argparse
import contextlib
import io
import os
import threading
import time

# Reference point for --stats "startup time" (launch -> first tracked frame),
# taken before the heavy imports below so their load time is included.
_PROCESS_START = time.perf_counter()

import warnings
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

# Without this, cv2.VideoCapture(...) on Windows can take 50-60s to open a USB
# camera because MSMF negotiates hardware color-conversion transforms for every
# advertised mode before returning. Must be set before any VideoCapture call.
os.environ.setdefault("OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS", "0")

# Where --roboflow-local caches downloaded Roboflow weights. inference's own
# default is "/tmp/cache", which on Windows lands in C:\tmp\cache -- keep it
# next to this script instead. Once cached, the model loads with no network.
# Must be set before importing inference, which reads it at import time.
_DEFAULT_MODEL_CACHE = Path(__file__).resolve().parent / "model_cache"
if not str(_DEFAULT_MODEL_CACHE).isascii() and os.environ.get("LOCALAPPDATA"):
    # onnxruntime's TensorRT provider can't create its engine cache folder
    # under a non-ASCII path (e.g. OneDrive's Documents folder on a
    # Japanese-locale Windows, which has a Japanese name): it fails with "The
    # system cannot find the path specified" and silently falls back to
    # plain CUDA, ~2x slower per frame.
    _DEFAULT_MODEL_CACHE = Path(os.environ["LOCALAPPDATA"]) / "picklevision" / "model_cache"
os.environ.setdefault("MODEL_CACHE_DIR", str(_DEFAULT_MODEL_CACHE))
# inference points Ultralytics' settings folder at a temp path that doesn't
# exist yet, so Ultralytics warns on every launch and falls back to creating
# an "Ultralytics/" folder in whatever directory tracker.py is run from. Put
# it back at Ultralytics' own Windows default (%APPDATA%\Ultralytics).
if os.environ.get("APPDATA"):
    os.environ.setdefault("YOLO_CONFIG_DIR", os.environ["APPDATA"])
# Only a single detection model is used from inference -- skip its version
# check (a network call, unwanted offline) and the unused core models that
# otherwise print a startup warning each. Only CUDA/CPU are ever available on
# this Windows setup, so don't also probe (and warn about) OpenVINO/CoreML.
os.environ.setdefault("DISABLE_VERSION_CHECK", "True")
os.environ.setdefault("CORE_MODEL_GAZE_ENABLED", "False")
os.environ.setdefault("CORE_MODEL_SAM_ENABLED", "False")
os.environ.setdefault("CORE_MODEL_SAM3_ENABLED", "False")
for _flag in (
    "PALIGEMMA_ENABLED", "FLORENCE2_ENABLED", "QWEN_2_5_ENABLED", "QWEN_3_ENABLED", "SMOLVLM2_ENABLED",
    "DEPTH_ESTIMATION_ENABLED", "MOONDREAM2_ENABLED", "CORE_MODEL_TROCR_ENABLED",
    "CORE_MODEL_GROUNDINGDINO_ENABLED", "CORE_MODEL_PE_ENABLED",
):
    os.environ.setdefault(_flag, "False")

# TensorRT (FP16), when its pip libraries (tensorrt-cu12-libs) are installed:
# benchmarked on an RTX 4060 Laptop at 6.7 -> 3.1 ms for the model itself,
# 12.5 -> 10.5 ms per full tracker frame. Its DLLs must be on PATH before
# onnxruntime loads its TensorRT provider. onnxruntime falls back to CUDA if
# TensorRT fails. Set PICKLEVISION_TENSORRT=0 to skip it.
_TENSORRT_ENABLED = False
if os.environ.get("PICKLEVISION_TENSORRT", "1") != "0":
    import importlib.util

    _trt_spec = importlib.util.find_spec("tensorrt_libs")
    if _trt_spec is not None and _trt_spec.submodule_search_locations:
        _trt_dir = list(_trt_spec.submodule_search_locations)[0]
        os.environ["PATH"] = _trt_dir + os.pathsep + os.environ.get("PATH", "")
        if hasattr(os, "add_dll_directory"):
            os.add_dll_directory(_trt_dir)
        _TENSORRT_ENABLED = True
os.environ.setdefault(
    "ONNXRUNTIME_EXECUTION_PROVIDERS",
    "[TensorrtExecutionProvider,CUDAExecutionProvider,CPUExecutionProvider]"
    if _TENSORRT_ENABLED
    else "[CUDAExecutionProvider,CPUExecutionProvider]",
)
# Deprecation notices from libraries inference imports internally, not from this code.
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
warnings.filterwarnings("ignore", message="Importing from timm.models.layers is deprecated")

import cv2
import numpy as np
import supervision as sv
import torch

# onnxruntime-gpu (what --roboflow-local runs on for the GPU) doesn't find the
# pip-installed NVIDIA CUDA/cuDNN DLLs on Windows by itself -- load them
# explicitly before inference creates its session. No-op on CPU-only onnxruntime.
#
# With the CUDA build of torch (a CUDA 13 build: there's no CUDA 12 build of
# torch 2.14), torch has already loaded its own, newer cuDNN by this point --
# same DLL names as onnxruntime's pinned cuDNN 9.7, so only one set can be
# loaded, and it has to be torch's: loading onnxruntime's first stops torch
# from importing at all. So onnxruntime's cuDNN is skipped then (it would only
# fail to load, noisily) and its CUDA provider runs on torch's cuDNN -- checked
# on an RTX 4050 Laptop: correct detections on both TensorRT and plain CUDA.
# That's also what preload_dlls' warning that torch isn't a CUDA 12 build is
# about, so it's silenced.
try:
    import onnxruntime

    if "CUDAExecutionProvider" in onnxruntime.get_available_providers():
        if torch.cuda.is_available():
            with contextlib.redirect_stdout(io.StringIO()):
                onnxruntime.preload_dlls(cudnn=False)
        else:
            onnxruntime.preload_dlls()
except (ImportError, AttributeError):
    pass

from inference import get_model
from inference_sdk import InferenceHTTPClient

try:
    from ultralytics import YOLO
except ModuleNotFoundError:
    YOLO = None

# Serializes calls into the --roboflow-local model, which dual-camera mode
# shares between both cameras' trackers running on two threads. The GPU runs
# one inference at a time regardless, so this costs nothing -- the rest of
# each tracker's per-frame work still overlaps the other camera's.
_LOCAL_MODEL_LOCK = threading.Lock()


def _open_camera(index_or_path):
    """Open a video source, using the MSMF backend for integer camera indices on Windows.

    Explicit CAP_MSMF avoids DSHOW's "can't capture by index" failure on some builds
    and, combined with OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS=0 above, opens in
    under a second instead of 50-60s.
    """
    if isinstance(index_or_path, int) and os.name == "nt":
        return cv2.VideoCapture(index_or_path, cv2.CAP_MSMF)
    return cv2.VideoCapture(index_or_path)


def _open_video_source(source, target_width, target_height, target_fps):
    """Open a video file or USB camera index, a camera configured the way
    run_video configures one (MJPEG at the target size/fps). Returns
    (cap, is_live).

    Unlike run_video, never falls back to a different camera index when this
    one fails to open: with two cameras connected, that fallback would just
    open the other camera a second time.
    """
    if isinstance(source, str) and source.isdigit() and not Path(source).exists():
        source = int(source)
    cap = _open_camera(source)
    if not cap.isOpened():
        raise FileNotFoundError(f"Unable to open source: {source}")
    is_live = isinstance(source, int)
    if is_live:
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, target_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, target_height)
        cap.set(cv2.CAP_PROP_FPS, target_fps)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return cap, is_live


def _flip_frame(frame, horizontal, vertical):
    """Apply --flip-horizontal/--flip-vertical (or a second camera's own) to a frame."""
    if not (horizontal or vertical):
        return frame
    return cv2.flip(frame, -1 if (horizontal and vertical) else (1 if horizontal else 0))


def _draw_label(frame, text, origin, color, scale=0.6, thickness=2):
    """cv2.putText on a black box, so it stays readable over any camera image.
    origin is the text's bottom-left corner, as for putText."""
    (text_w, text_h), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
    x, y = origin
    cv2.rectangle(frame, (x - 4, y - text_h - 6), (x + text_w + 4, y + baseline + 2), (0, 0, 0), -1)
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness)


class _FrameReader:
    """Reads (and MJPEG-decodes) frames on a background thread.

    Decoding is CPU work that otherwise runs in series with detection on the
    main loop -- ~8ms/frame at 1080p, a third of the whole frame budget. On a
    thread it overlaps with detection of the previous frame instead.

    live=True (a camera) keeps only the newest frame: if detection falls
    behind, stale frames are dropped so what's processed is always current.
    live=False (a video file) queues every frame, blocking the reader when
    the queue is full, so offline evaluation never skips a frame.
    """

    def __init__(self, cap, live, queue_size=4):
        import queue
        import threading

        self.cap = cap
        self.live = live
        self._queue = queue.Queue(maxsize=1 if live else queue_size)
        self._stopped = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        import queue

        while not self._stopped.is_set():
            try:
                ok, frame = self.cap.read()
            except cv2.error:
                # read() can throw instead of returning False -- seen on the
                # built-in webcam when asked for a size it doesn't support.
                # Treated as the source failing; otherwise this thread dies
                # and the tracking loop waits forever for its next frame.
                ok, frame = False, None
            # Arrival time rides along with the frame, so --stats can measure
            # latency from capture to tracked, queue wait included.
            item = (frame, time.perf_counter()) if ok else (None, None)
            if self.live:
                try:
                    self._queue.get_nowait()  # drop the stale frame, if any
                except queue.Empty:
                    pass
                self._queue.put(item)
            else:
                while not self._stopped.is_set():
                    try:
                        self._queue.put(item, timeout=0.1)
                        break
                    except queue.Full:
                        continue
            if not ok:
                return

    def read(self):
        """Next (frame, arrival_time), or (None, None) once the source is exhausted/failed."""
        return self._queue.get()

    def stop(self):
        self._stopped.set()
        self._thread.join(timeout=1.0)


class _FrameDisplay:
    """Shows the newest annotated frame from a background thread, at most max_fps.

    imshow + waitKey on Windows benchmarked at 5-14 ms per call -- more than
    the whole detection step -- and cut end-to-end throughput roughly in half
    when done on every frame of the main loop. Here tracking never waits on
    the window: it hands over its latest frame and moves on, and the window
    repaints at a steady rate (60 fps is already smoother than the eye needs
    for a preview). The window is created, drawn, and destroyed all on this
    one thread, as HighGUI requires.
    """

    def __init__(self, window_name, width, height, max_fps=60):
        import threading

        self._window_name = window_name
        self._size = (width, height)
        self._interval = 1.0 / max_fps
        self._latest = None
        self._lock = threading.Lock()
        self._stopped = threading.Event()
        self.quit_requested = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def show(self, frame):
        with self._lock:
            self._latest = frame

    def _run(self):
        cv2.namedWindow(self._window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self._window_name, *self._size)
        try:
            while not self._stopped.is_set():
                started = time.perf_counter()
                with self._lock:
                    frame, self._latest = self._latest, None
                if frame is not None:
                    cv2.imshow(self._window_name, frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    self.quit_requested.set()
                remaining = self._interval - (time.perf_counter() - started)
                if remaining > 0:
                    time.sleep(remaining)
        finally:
            cv2.destroyAllWindows()

    def stop(self):
        self._stopped.set()
        self._thread.join(timeout=2.0)


class _SessionStats:
    """--stats: performance and resource log for one tracking session.

    Records startup time (launch -> first tracked frame), and once a second:
    processed FPS, frame latency (camera arrival -> tracking done, avg and
    max), CPU and RAM use, and GPU utilization/temperature/power/VRAM via
    NVIDIA's NVML. Every sample goes to a CSV; a line is printed every 5 s and
    a summary at the end. CPU temperature isn't included -- Windows exposes no
    reliable non-admin API for it (use HWiNFO alongside if it's needed).
    """

    def __init__(self, csv_path):
        import csv

        import psutil

        self._psutil = psutil
        self._proc = psutil.Process()
        self._proc.cpu_percent(None)  # first call only primes the counters
        psutil.cpu_percent(None)
        self._nvml = self._gpu = None
        try:
            import pynvml

            pynvml.nvmlInit()
            self._nvml, self._gpu = pynvml, pynvml.nvmlDeviceGetHandleByIndex(0)
        except Exception:
            print("[Stats] NVIDIA GPU monitoring unavailable -- GPU columns will be empty")

        csv_path = Path(csv_path)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        self.csv_path = csv_path
        self._file = open(csv_path, "w", newline="")
        self._csv = csv.writer(self._file)
        self._csv.writerow([
            "time", "elapsed_s", "fps", "latency_avg_ms", "latency_max_ms", "cpu_system_pct",
            "cpu_tracker_pct", "ram_tracker_mb", "ram_system_pct", "gpu_util_pct", "gpu_temp_c",
            "gpu_power_w", "gpu_mem_mb",
        ])

        self.startup_s = None
        self._all_latencies: list[float] = []
        self._window: list[float] = []
        self._rows: list[dict] = []
        self._session_start = self._window_start = self._last_print = time.perf_counter()

    def frame_done(self, arrived_at):
        now = time.perf_counter()
        if self.startup_s is None:
            self.startup_s = now - _PROCESS_START
            print(f"[Stats] Startup time (launch -> first tracked frame): {self.startup_s:.1f}s")
            self._session_start = self._window_start = self._last_print = now
        latency = (now - arrived_at) * 1000
        self._window.append(latency)
        self._all_latencies.append(latency)
        if now - self._window_start >= 1.0:
            self._sample(now)

    def _gpu_reading(self):
        if self._gpu is None:
            return None, None, None, None
        n = self._nvml
        try:
            return (
                n.nvmlDeviceGetUtilizationRates(self._gpu).gpu,
                n.nvmlDeviceGetTemperature(self._gpu, n.NVML_TEMPERATURE_GPU),
                round(n.nvmlDeviceGetPowerUsage(self._gpu) / 1000, 1),
                round(n.nvmlDeviceGetMemoryInfo(self._gpu).used / 2**20),
            )
        except Exception:
            return None, None, None, None

    def _sample(self, now):
        window_s = now - self._window_start
        gpu_util, gpu_temp, gpu_power, gpu_mem = self._gpu_reading()
        row = {
            "fps": len(self._window) / window_s,
            "lat_avg": float(np.mean(self._window)),
            "lat_max": float(np.max(self._window)),
            "cpu_sys": self._psutil.cpu_percent(None),
            # Normalized to the whole CPU (psutil reports per-core, up to N*100%).
            "cpu_proc": self._proc.cpu_percent(None) / (self._psutil.cpu_count() or 1),
            "ram_proc": self._proc.memory_info().rss / 2**20,
            "ram_sys": self._psutil.virtual_memory().percent,
            "gpu_util": gpu_util, "gpu_temp": gpu_temp, "gpu_power": gpu_power, "gpu_mem": gpu_mem,
        }
        self._rows.append(row)
        self._csv.writerow([
            time.strftime("%Y-%m-%d %H:%M:%S"), round(now - self._session_start, 1), round(row["fps"], 1),
            round(row["lat_avg"], 2), round(row["lat_max"], 2), row["cpu_sys"], round(row["cpu_proc"], 1),
            round(row["ram_proc"]), row["ram_sys"], gpu_util, gpu_temp, gpu_power, gpu_mem,
        ])
        self._file.flush()
        if now - self._last_print >= 5.0:
            gpu = f" | GPU {gpu_util}% {gpu_temp}C {gpu_power}W" if gpu_temp is not None else ""
            print(
                f"[Stats] {row['fps']:.0f} fps | latency avg {row['lat_avg']:.1f} ms, max {row['lat_max']:.1f} ms"
                f" | CPU {row['cpu_sys']:.0f}% | RAM {row['ram_proc']:.0f} MB{gpu}"
            )
            self._last_print = now
        self._window = []
        self._window_start = now

    def close(self):
        now = time.perf_counter()
        if self._window and now - self._window_start > 0.2:
            self._sample(now)
        self._file.close()
        if self._all_latencies:
            lat = np.array(self._all_latencies)
            duration = now - self._session_start

            def column(key):
                return [r[key] for r in self._rows if r[key] is not None]

            temps = column("gpu_temp")
            print("\n[Stats] ===== Session summary =====")
            print(f"[Stats] Startup time: {self.startup_s:.1f}s")
            print(f"[Stats] Frames tracked: {len(lat)} in {duration:.0f}s (avg {len(lat) / duration:.1f} fps)")
            print(f"[Stats] Latency: avg {lat.mean():.1f} ms, 95th pct {np.percentile(lat, 95):.1f} ms, max {lat.max():.1f} ms")
            if self._rows:
                print(f"[Stats] CPU (system): avg {np.mean(column('cpu_sys')):.0f}%, peak {max(column('cpu_sys')):.0f}%")
                print(f"[Stats] RAM (tracker): peak {max(column('ram_proc')):.0f} MB")
            if temps:
                print(f"[Stats] GPU: avg {np.mean(column('gpu_util')):.0f}% util, temp avg {np.mean(temps):.0f}C / "
                      f"peak {max(temps)}C, power avg {np.mean(column('gpu_power')):.0f} W")
            print(f"[Stats] Per-second log: {self.csv_path}")
        if self._nvml is not None:
            try:
                self._nvml.nvmlShutdown()
            except Exception:
                pass


@dataclass
class BallEvent:
    frame_index: int
    track_id: int
    centroid: tuple[float, float]
    velocity: float = 0.0
    landing_point: tuple[float, float] | None = None
    line_call: str = "UNKNOWN"
    timestamp: float = field(default_factory=time.time)
    # Tags which camera produced this event: "A"/"B" in dual-camera mode
    # (--source2), where DualCameraTracker runs one PickleVisionTracker per end
    # camera and fuses their events into line calls on the full court.
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


def to_global_court_point(local_point, camera_end, half_length=22.0, court_width=20.0):
    """Convert a point from an END camera's own half-court calibration (local Y
    in [0, half_length], 0 = net) into one shared GLOBAL full-court coordinate
    system (global Y in [0, 2*half_length], 0 = end-A baseline, half_length =
    net, 2*half_length = end-B baseline).

    Global X follows camera A's local X (0 = the sideline on camera A's left);
    camera B's X is mirrored. The two end cameras face each other, so B's left
    is A's right: each calibration puts local X=0 at the left of ITS OWN image
    (--calibrate's TOP-LEFT is clicked as seen on screen), which is the
    opposite sideline for the two cameras. Centering both cameras on the
    centerline doesn't change that -- it only makes the mirror symmetric about
    the centerline, so passing X through unchanged was right for points ON
    the centerline and off by 2*(10 - x) ft everywhere else.
    """
    if camera_end not in ("A", "B"):
        raise ValueError(f"camera_end must be 'A' or 'B', got {camera_end!r}")
    local_x, local_y = local_point
    if camera_end == "A":
        return (local_x, half_length - local_y)
    return (court_width - local_x, half_length + local_y)


def check_camera_alignment(point_a_local, point_b_local, half_length=22.0, tolerance_ft=2.0, court_width=20.0):
    """Compare what the two end cameras each report, in their own local
    calibration, for what should be the SAME real-world point at the same
    moment (e.g. a person standing still while both cameras track them, or a
    synchronized ball position) -- both readings should land on nearly the
    same GLOBAL court coordinate if the two calibrations genuinely agree.

    Returns a dict with the two points converted to global coordinates, the
    per-axis and combined discrepancy, and an "aligned" verdict against
    `tolerance_ft`. A large X discrepancy points at the two cameras
    disagreeing on which side is "left" -- one feed flipped with
    --flip-horizontal and the other not, or one camera's corners clicked in
    mirrored order; a large Y discrepancy points at a scale/distance
    calibration issue instead -- report whichever axis is actually large,
    don't just report the combined number, since the axis tells you which
    problem to go fix.
    """
    global_a = to_global_court_point(point_a_local, "A", half_length, court_width)
    global_b = to_global_court_point(point_b_local, "B", half_length, court_width)
    dx = global_a[0] - global_b[0]
    dy = global_a[1] - global_b[1]
    distance = float(np.hypot(dx, dy))
    return {
        "global_a": global_a,
        "global_b": global_b,
        "dx": float(dx),
        "dy": float(dy),
        "distance_ft": distance,
        "aligned": distance <= tolerance_ft,
    }


class PickleVisionTracker:
    """Single-camera draft for Project PickleVision using YOLOv8 detection + tracking."""

    # Per-frame movement (px) up to which a step is treated as detector jitter
    # and fully smoothed; larger steps are smoothed proportionally less.
    SMOOTHING_JITTER_PX = 8.0
    # Fraction of ball_color_min_ratio a candidate still needs when the color
    # filter falls back to its looser stage (re-acquiring an existing lock).
    LOOSE_COLOR_FRACTION = 0.25
    # Static-spot suppression (see _static_candidates): detections are counted
    # per STATIC_CELL_PX grid cell with STATIC_DECAY per frame; a cell whose
    # count passes STATIC_THRESHOLD (~30 consecutive frames) is background.
    STATIC_CELL_PX = 24
    STATIC_DECAY = 0.97
    STATIC_THRESHOLD = 20.0
    # A static candidate is still accepted if it's this close to the tracked
    # ball's last position -- i.e. the ball being followed has come to rest.
    STATIC_KEEP_RADIUS_PX = 25.0
    # Gap prediction (see _fit_motion): how many recent real detections to
    # fit, the minimum needed to fit at all, and the polynomial degree in y.
    # Chosen by simulated-gap benchmark (noisy ballistic shots, 5-15 hidden
    # frames): 10-point gravity fit had the lowest error, ~6-10x below the old
    # two-point decayed extrapolation.
    PREDICT_FIT_POINTS = 10
    PREDICT_MIN_POINTS = 3
    PREDICT_Y_DEGREE = 2

    def __init__(
        self,
        model_name: str = "yolov8n.pt",
        tracker_config: str = "bytetrack.yaml",
        conf: float = 0.25,
        reacquire_conf: float = 0.1,
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
        roboflow_infer_size: int = 640,
        roboflow_local: bool = False,
        exclude_people: bool = True,
        smoothing: float = 0.5,
        bounce_min_dy: float = 3.0,
        shared_roboflow_model=None,
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
        #
        # roboflow_local runs the same Roboflow model in-process via the
        # inference package instead of calling a server (no Docker): weights
        # are downloaded once with the API key into MODEL_CACHE_DIR, then load
        # from there offline. Only model_id is supported locally, not Workflows.
        self.use_roboflow = use_roboflow or roboflow_local
        self.roboflow_local_model = None
        if self.use_roboflow:
            self.model = None
            if roboflow_local:
                if not roboflow_model_id:
                    raise ValueError("--roboflow-local requires --roboflow-model-id (Workflows aren't supported locally)")
                if shared_roboflow_model is not None:
                    # Dual-camera mode: the second camera's tracker runs the
                    # model the first one already loaded, instead of loading
                    # its own copy into VRAM (see _LOCAL_MODEL_LOCK).
                    self.roboflow_local_model = shared_roboflow_model
                else:
                    if _TENSORRT_ENABLED and not any(
                        Path(os.environ["MODEL_CACHE_DIR"]).glob(f"{roboflow_model_id}/**/*.engine")
                    ):
                        print(
                            "[TensorRT] First launch for this model: building an optimized engine for this GPU. "
                            "This takes about 4-5 minutes once; later launches reuse it."
                        )
                    self.roboflow_local_model = get_model(model_id=roboflow_model_id, api_key=roboflow_api_key)
                self.device = "roboflow-local"
            else:
                self.device = "roboflow-server"
                self.roboflow_client = InferenceHTTPClient.init(api_url=roboflow_api_url, api_key=roboflow_api_key)
            self.roboflow_workspace_name = roboflow_workspace_name
            self.roboflow_model_id = roboflow_model_id
            self.roboflow_workflow_id = roboflow_workflow_id
            self.roboflow_infer_size = roboflow_infer_size
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

        # A player wearing ball-colored clothing can pass both the color and
        # shape/size filters and get mistaken for the ball -- especially during
        # position-based re-acquisition or motion-blur prediction, which have no
        # other way to know a candidate/predicted point is sitting on a person.
        # Runs a small, fast local person detector alongside whichever backend
        # is doing ball detection (cheap: ~10-15ms on this GPU, per the earlier
        # ~90fps single-stream benchmark) to reject any such candidate outright.
        self.exclude_people = exclude_people and YOLO is not None
        self.person_model = None
        if self.exclude_people:
            person_device = "cuda" if torch.cuda.is_available() else "cpu"
            self.person_model = YOLO("yolov8n.pt")
            self.person_model.to(person_device)

        self.tracker_config = tracker_config
        self.conf = conf
        # Used only to reconfirm an ALREADY-tracked ball near its last known
        # position (see _confidence_floor) -- spatial distance-matching there
        # validates a weak detection, so a low-confidence-but-nearby candidate
        # during blur is more useful accepted than discarded, forcing a fall
        # back to pure motion-blur prediction (or a full lock reset) instead.
        # Fresh acquisition (no active lock, no spatial prior) still requires
        # the full `conf` -- see _confidence_floor.
        self.reacquire_conf = reacquire_conf
        self.iou = iou
        self.imgsz = imgsz
        self.target_class_id = target_class_id
        self.max_history = max_history
        self.track_history = defaultdict(list)
        self.events: list[BallEvent] = []
        self.frame_index = 0
        self.court_mapper = court_mapper or CourtMapper()
        # False when running on the default full-frame CourtMapper above,
        # whose "court" is just the whole image.
        self.court_calibrated = court_mapper is not None
        self.target_fps = target_fps
        self.target_width = target_width
        self.target_height = target_height
        self.camera_id = camera_id
        # Prepended to this tracker's console messages -- DualCameraTracker
        # sets it per camera, so each line says which camera it's about.
        self.log_prefix = ""

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

        # Detector boxes wobble a few pixels frame-to-frame even on a still
        # ball. Raw centroids made the drawn path scribble, and that noise fed
        # straight into velocity/prediction and bounce detection. smoothing is
        # an exponential moving average weight on the previous point (0 = raw,
        # closer to 1 = smoother but laggier). bounce_min_dy is the per-frame
        # vertical movement (px) below which a Y change counts as noise, not a
        # direction change -- otherwise every wobble registers as a "bounce".
        self.smoothing = smoothing
        self.bounce_min_dy = bounce_min_dy
        self._static_heat: dict[tuple[int, int], float] = {}
        # Whether the current lock has ever travelled more than
        # 2 * STATIC_KEEP_RADIUS_PX from where it started -- see _drop_static.
        self._lock_origin: tuple[float, float] | None = None
        self._lock_has_moved = False
        # Raw (unsmoothed) real detections of the current lock as
        # (frame_index, x, y) -- the input to _fit_motion.
        self._observations: list[tuple[int, float, float]] = []

        # Internal bookkeeping IDs (primary_track_id) can legitimately change
        # every single frame for the Roboflow backend -- IDs are deliberately
        # never reused there (see _process_frame_roboflow) to stop the "same ID
        # still present" fast path from ever falsely matching. That's correct
        # internally but confusing displayed on screen: a genuinely continuous
        # ball would show a new "Ball ID" every frame. display_track_id is the
        # user-facing identity instead -- it only advances when a NEW logical
        # lock begins (the "no active lock yet" branch in
        # _select_primary_detection), staying constant through every fast-path
        # or position-matched continuation of the same lock.
        self.display_track_id: int | None = None
        self._next_display_id = 1

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

    def _confidence_floor(self):
        """Minimum detection confidence to consider, given current lock state.

        An active lock has a spatial prior (primary_trajectory) to validate a
        candidate against in _select_primary_detection, so a weaker detection
        near the expected position is worth considering there. Fresh
        acquisition has no such prior, so it still requires the full `conf`.
        """
        return self.reacquire_conf if self.primary_trajectory else self.conf

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

        # If the ball was going down (+dy) and suddenly goes up (-dy), it bounced.
        # Steps smaller than bounce_min_dy are detector jitter, not movement, so
        # they're dropped before comparing directions.
        signs = np.sign(dy[np.abs(dy) >= self.bounce_min_dy])
        direction_change_y = np.any(signs[:-1] != signs[1:])

        # Speed drop is a fallback for when a ball rolls or loses momentum near the boundary.
        # Only the moment it slows counts -- a ball that's simply stationary (held,
        # resting) would otherwise register as a new "bounce" on every frame.
        speed_drop = self._estimate_velocity(recent) < 4.0 <= self._estimate_velocity(recent[:-1])

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

        return self.court_mapper.classify(self.frame_coords(court_point))

    def frame_coords(self, point):
        """Offset a point from the frame process_frame works on -- the cropped
        court region when zoomed -- back to original-frame pixels, the space
        the court calibration is in."""
        if self.zoom_roi is None:
            return point
        offset_x, offset_y, _, _ = self.zoom_roi
        return (point[0] + offset_x, point[1] + offset_y)

    def ball_position(self):
        """The locked ball's position this frame, in original-frame pixels,
        and whether it's a real detection (False = bridged by prediction
        through a detection gap). None when no ball is locked."""
        if not self.primary_trajectory:
            return None
        return self.frame_coords(self.primary_trajectory[-1]), self.missed_frames == 0

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

    def _detect_people(self, frame):
        """Bounding boxes of people in `frame`, used to reject ball candidates
        (or drifting motion-blur predictions) that land on a person -- clothing
        color/shape alone can't tell a player wearing ball-colored gear apart
        from the real ball.
        """
        if self.person_model is None:
            return []
        results = self.person_model.predict(frame, classes=[0], conf=0.3, verbose=False)
        if not results or results[0].boxes is None or len(results[0].boxes) == 0:
            return []
        return results[0].boxes.xyxy.cpu().numpy().tolist()

    @staticmethod
    def _point_in_any_box(point, boxes):
        px, py = point
        for x1, y1, x2, y2 in boxes:
            if x1 <= px <= x2 and y1 <= py <= y2:
                return True
        return False

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

    def _narrow_candidate_pool(self, frame, boxes, strict_color=False):
        """Progressively filter candidates by person-exclusion, shape, then color.

        Shape and color fall back to a looser stage whenever a stricter one
        eliminates everything -- so a real ball lacking a strong color match
        (e.g. mostly in shadow) still gets considered, rather than the filter
        silently doing nothing (the color fallback still needs a small
        fraction of ball color, see LOOSE_COLOR_FRACTION). strict_color disables the color
        fallback: used for fresh acquisition, which has no spatial prior, so a
        lone non-ball-colored false positive (a wall cable, a curtain edge) would
        otherwise pass by default and start a bogus lock. Person-exclusion is a hard
        filter with no such fallback: if every remaining candidate overlaps a
        detected person, none of them should be trusted as the ball, so an
        empty pool is the correct result, not something to loosen.
        """
        all_indices = np.arange(len(boxes))

        if self.exclude_people:
            person_boxes = self._detect_people(frame)
            if person_boxes:
                all_indices = np.array(
                    [i for i in all_indices if not self._point_in_any_box(self._get_ball_centroid(boxes[i]), person_boxes)],
                    dtype=int,
                )

        if len(all_indices) == 0:
            return all_indices

        shape_matches = np.array([i for i in all_indices if self._is_plausible_ball_shape(boxes[i])])
        pool = shape_matches if len(shape_matches) else all_indices

        if self.require_ball_color:
            color_ratios = np.array([self._ball_color_ratio(frame, boxes[i]) for i in pool])
            color_matches = pool[color_ratios >= self.ball_color_min_ratio]
            if len(color_matches) or strict_color:
                pool = color_matches
            else:
                # Even the looser fallback needs *some* ball color: otherwise
                # any non-ball junk the model flags near the last position gets
                # adopted whenever the real ball blurs out for a frame. A
                # shadowed real ball still keeps a few yellow pixels.
                pool = pool[color_ratios >= self.ball_color_min_ratio * self.LOOSE_COLOR_FRACTION]

        return pool

    def _static_candidates(self, boxes):
        """Flag detections sitting at a spot that has been detected nearly every frame.

        A ball in play keeps moving; something detected at the same spot for
        ~30 consecutive frames is background the model mistakes for a distant
        ball. Observed live: a small yellow object on the floor, detected in
        26/30 frames at up to 0.47 conf with 24-31% ball color -- indistinguishable
        from a far-away ball by color, shape, or confidence, only by never moving.
        Counts decay every call, so a spot stops being "static" once it's gone.
        """
        cell = self.STATIC_CELL_PX
        for key in list(self._static_heat):
            self._static_heat[key] *= self.STATIC_DECAY
            if self._static_heat[key] < 0.5:
                del self._static_heat[key]
        cells = [(int(cx // cell), int(cy // cell)) for cx, cy in (self._get_ball_centroid(b) for b in boxes)]
        for key in set(cells):
            self._static_heat[key] = self._static_heat.get(key, 0.0) + 1.0
        return np.array([self._static_heat[key] >= self.STATIC_THRESHOLD for key in cells])

    def _drop_static(self, candidate_pool, boxes, static):
        """Remove static-spot candidates, unless it's the tracked ball itself at rest.

        "Itself at rest" requires the lock to have moved at some point: a lock
        that has never moved is most likely on background that got grabbed
        before its spot was recognized as static (e.g. right at startup), and
        it's dropped here once that happens instead of being kept forever.
        """
        if not len(candidate_pool) or not static[candidate_pool].any():
            return candidate_pool
        last_point = np.array(self.primary_trajectory[-1]) if self.primary_trajectory and self._lock_has_moved else None

        def keep(i):
            if not static[i]:
                return True
            if last_point is None:
                return False
            return np.linalg.norm(np.array(self._get_ball_centroid(boxes[i])) - last_point) <= self.STATIC_KEEP_RADIUS_PX

        return np.array([i for i in candidate_pool if keep(i)], dtype=int)

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

        static = self._static_candidates(boxes)

        if self.primary_track_id in ids:
            idx = ids.index(self.primary_track_id)
            return boxes[idx], ids[idx]

        # Same asymmetry as _confidence_floor: an active lock's position match
        # validates a poorly-colored candidate (e.g. ball in shadow), but a
        # fresh lock has nothing else vouching for it, so color is mandatory.
        candidate_pool = self._narrow_candidate_pool(frame, boxes, strict_color=not self.primary_trajectory)
        candidate_pool = self._drop_static(candidate_pool, boxes, static)
        if len(candidate_pool) == 0:
            return None

        if self.primary_trajectory:
            last_point = np.array(self.primary_trajectory[-1])
            centroids = np.array([self._get_ball_centroid(b) for b in boxes])
            distances = np.linalg.norm(centroids - last_point, axis=1)
            best_idx = int(candidate_pool[np.argmin(distances[candidate_pool])])
            if distances[best_idx] <= self.max_match_distance:
                print(f"{self.log_prefix}[Ball Lock] Re-acquired via position match: box={boxes[best_idx]}, distance={distances[best_idx]:.0f}px")
                return boxes[best_idx], ids[best_idx]
            if not self.missed_frames:
                return None
            # The lock is running on prediction alone, which is exactly when
            # its position is least trustworthy -- after a fast direction change
            # the extrapolated point heads the wrong way, and the real ball ends
            # up beyond max_match_distance of it. Observed live: the ball in
            # plain view, ignored for the whole prediction window while the
            # marker drifted across the room. A detection confident enough to
            # start a fresh lock is trusted over the guess instead.
            candidate_pool = self._narrow_candidate_pool(frame, boxes, strict_color=True)
            candidate_pool = self._drop_static(candidate_pool, boxes, static)
            if confs is not None and len(confs):
                candidate_pool = candidate_pool[confs[candidate_pool] >= self.conf]
            if len(candidate_pool) == 0:
                return None
            # Drop the stale path so the trail doesn't draw a line back to the
            # wrong predicted spot. Same ball, so the displayed ID is kept.
            self.primary_trajectory = []
            switching = True
            print(f"{self.log_prefix}[Ball Lock] Prediction had drifted off the ball; switching to confident detection")
        else:
            switching = False

        # No active lock yet: acquire whichever plausible-shaped, ball-colored
        # detection the model is most confident about.
        if confs is not None and len(confs):
            best_idx = int(candidate_pool[np.argmax(confs[candidate_pool])])
        else:
            best_idx = int(candidate_pool[0])
        box = boxes[best_idx]
        # A genuinely new logical lock is starting -- this is the only place
        # display_track_id should advance (see its definition in __init__).
        if not switching:
            self.display_track_id = self._next_display_id
            self._next_display_id += 1
        print(f"{self.log_prefix}[Ball Lock] Acquired: box={box}, conf={confs[best_idx] if confs is not None and len(confs) else 'n/a'}, color_ratio={self._ball_color_ratio(frame, box):.2f}, display_id={self.display_track_id}")
        return box, ids[best_idx]

    def _fit_motion(self):
        """Predict the ball's position at the current frame from a motion fit.

        Fits the last PREDICT_FIT_POINTS *real* detections (never earlier
        predictions, so errors don't compound): x linear in time, y a
        polynomial of degree PREDICT_Y_DEGREE (2 = constant acceleration, i.e.
        gravity) once there are enough points to fit it. Averaging over several
        detections instead of differencing the last two keeps detector jitter
        out of the velocity estimate. Points before the most recent vertical
        direction reversal (a bounce or a hit) are dropped -- one smooth curve
        can't span one. Returns None if there isn't enough data to fit.
        """
        obs = self._observations[-self.PREDICT_FIT_POINTS:]
        if len(obs) < self.PREDICT_MIN_POINTS:
            return None
        t, xs, ys = (np.array(v, dtype=float) for v in zip(*obs))

        dy = np.diff(ys)
        moving = np.nonzero(np.abs(dy) >= self.bounce_min_dy)[0]
        for k in range(len(moving) - 1, 0, -1):
            if np.sign(dy[moving[k]]) != np.sign(dy[moving[k - 1]]):
                start = moving[k]
                t, xs, ys = t[start:], xs[start:], ys[start:]
                break
        distinct_times = len(np.unique(t))
        if distinct_times < 2:
            return None

        # Time relative to the newest detection keeps polyfit well-conditioned.
        tt = t - t[-1]
        target = self.frame_index - t[-1]
        y_degree = self.PREDICT_Y_DEGREE if distinct_times > self.PREDICT_Y_DEGREE + 2 else 1
        try:
            px = np.polyval(np.polyfit(tt, xs, 1), target)
            py = np.polyval(np.polyfit(tt, ys, y_degree), target)
        except (np.linalg.LinAlgError, ValueError):
            # Degenerate input must never take down a live session -- fall
            # back to the simple extrapolation instead.
            return None
        if not (np.isfinite(px) and np.isfinite(py)):
            return None
        return float(px), float(py)

    def _predict_primary_position(self):
        """Extrapolate the ball's position through a detection gap.

        Bridges brief detection gaps (typically motion blur at high ball speed,
        or occlusion by a player/paddle) so the trajectory and bounce logic
        don't reset on every single missed frame. Uses _fit_motion whenever the
        lock has enough real detections; the decayed two-point extrapolation
        below is only the fallback for a lock too new to fit.

        Decayed rather than a pure straight line: predicted points get appended
        back into primary_trajectory, so each successive prediction's own step
        is already `decay` times the previous one, geometrically shrinking the
        step size on its own (d, d*0.7, d*0.7^2, ...). A genuinely lost ball
        (stopped, bounced, left frame) settles near its last known position
        instead of flying off in a straight line for the full max_missed_frames
        window -- observed live drifting far enough to land on an unrelated
        person by the time that window ran out, under the old undamped version.
        """
        if len(self.primary_trajectory) < 2 or self.missed_frames > self.max_missed_frames:
            return None

        fitted = self._fit_motion()
        if fitted is not None:
            return fitted

        (x1, y1), (x2, y2) = self.primary_trajectory[-2], self.primary_trajectory[-1]
        decay = 0.7
        return (x2 + (x2 - x1) * decay, y2 + (y2 - y1) * decay)

    def _handle_missed_detection(self, frame):
        self.missed_frames += 1
        predicted_point = self._predict_primary_position()
        # A prediction outside the frame means the ball has left the view;
        # there's nothing left to bridge, so end the lock instead of drawing
        # a marker pinned to the edge.
        if predicted_point is not None and not (
            0 <= predicted_point[0] < frame.shape[1] and 0 <= predicted_point[1] < frame.shape[0]
        ):
            predicted_point = None
        if predicted_point is None:
            self.primary_track_id = None
            self.primary_trajectory = []
            self.display_track_id = None
            return frame

        if self.exclude_people and self._point_in_any_box(predicted_point, self._detect_people(frame)):
            # The decayed prediction has drifted onto a detected person -- treat
            # as genuinely lost rather than displaying a marker sitting on someone.
            self.primary_track_id = None
            self.primary_trajectory = []
            self.display_track_id = None
            return frame

        px, py = predicted_point
        predicted_box = (px - 10, py - 10, px + 10, py + 10)
        self._draw_primary_tracking(frame, predicted_box, predicted=True)
        return frame

    def _draw_primary_tracking(self, frame, box, predicted=False):
        x1, y1, x2, y2 = box
        center_x, center_y = self._get_ball_centroid(box)
        raw_x, raw_y = center_x, center_y
        # Predicted points are already derived from the smoothed trajectory.
        # Smoothing fades out as the step grows: small steps are detector
        # jitter and get the full weight, but a big step is real fast movement,
        # where averaging with the previous point would round off sharp
        # direction changes and leave the path lagging behind the ball.
        if self.primary_trajectory and not predicted:
            prev_x, prev_y = self.primary_trajectory[-1]
            step = np.hypot(center_x - prev_x, center_y - prev_y)
            weight = self.smoothing * min(1.0, self.SMOOTHING_JITTER_PX / step) if step > 0 else self.smoothing
            center_x = weight * prev_x + (1 - weight) * center_x
            center_y = weight * prev_y + (1 - weight) * center_y

        if not self.primary_trajectory:
            self._lock_origin = (center_x, center_y)
            self._lock_has_moved = False
            self._observations = []
        elif not predicted and not self._lock_has_moved:
            travelled = np.hypot(center_x - self._lock_origin[0], center_y - self._lock_origin[1])
            self._lock_has_moved = travelled > 2 * self.STATIC_KEEP_RADIUS_PX
        if not predicted:
            self._observations.append((self.frame_index, raw_x, raw_y))
            del self._observations[: -self.PREDICT_FIT_POINTS]

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
                    track_id=self.display_track_id,
                    centroid=(center_x, center_y),
                    velocity=velocity,
                    landing_point=landing_point,
                    line_call=call,
                    camera_id=self.camera_id,
                )
            )

        box_color = (0, 165, 255) if predicted else (0, 255, 0)
        label = f"Ball ID: {self.display_track_id}" + (" (predicted)" if predicted else "")
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

    def _infer_local(self, image, confidence):
        """Run the local Roboflow model directly on its ONNX session.

        Equivalent to roboflow_local_model.infer() for this model's
        preprocessing ("Fit (black edges) in" to a square input, RGB, /255) and
        its NMS-free output (N x [x1, y1, x2, y2, conf, class] in input-space
        pixels), replicating inference's own resize/padding/rounding steps --
        but faster: inference's generic path builds a non-contiguous input
        array (an extra copy before the GPU) and wraps every box in a pydantic
        object. Checked box-for-box against infer() on real camera frames and
        screenshots: identical boxes and confidences.

        Falls back to infer() for any model not matching that shape (e.g. after
        retraining with different preprocessing).
        """
        model = self.roboflow_local_model
        session = getattr(model, "onnx_session", None)
        if session is None or getattr(model, "resize_method", None) != "Fit (black edges) in":
            with _LOCAL_MODEL_LOCK:
                result = model.infer(image, confidence=confidence, iou_threshold=self.iou)[0]
            return [
                {"x": p.x, "y": p.y, "width": p.width, "height": p.height, "confidence": p.confidence}
                for p in result.predictions
            ]

        in_h, in_w = model.img_size_h, model.img_size_w
        h, w = image.shape[:2]
        # Same new-size arithmetic as inference's resize_image_keeping_aspect_ratio.
        if w / h >= in_w / in_h:
            new_w, new_h = in_w, int(in_w / (w / h))
        else:
            new_w, new_h = int(in_h * (w / h)), in_h
        resized = image if (new_w, new_h) == (w, h) else cv2.resize(image, (new_w, new_h))
        top, left = (in_h - new_h) // 2, (in_w - new_w) // 2
        # Buffers are reused across frames (the black padding never changes
        # while the frame size doesn't). Filling each RGB plane straight from
        # the uint8 BGR canvas is bit-identical to cv2.dnn.blobFromImage(...,
        # 1/255, swapRB=True) but ~4x faster (0.6 vs 2.5 ms benchmarked).
        key = (in_h, in_w, new_h, new_w)
        if getattr(self, "_infer_buffers_key", None) != key:
            self._infer_canvas = np.zeros((in_h, in_w, 3), np.uint8)
            self._infer_blob = np.empty((1, 3, in_h, in_w), np.float32)
            self._infer_buffers_key = key
        canvas, blob = self._infer_canvas, self._infer_blob
        canvas[top : top + new_h, left : left + new_w] = resized
        for plane, channel in enumerate((2, 1, 0)):  # BGR -> RGB
            np.multiply(canvas[:, :, channel], np.float32(1 / 255.0), out=blob[0, plane], casting="unsafe")

        with _LOCAL_MODEL_LOCK:
            out = session.run(None, {model.input_name: blob})[0][0]
        out = out[out[:, 4] > confidence]
        if not len(out):
            return []

        # Same inverse mapping as inference's undo_image_padding_for_predicted_boxes
        # + clip_boxes_coordinates (note: round() here vs int() above, as there).
        scale = min(in_h / h, in_w / w)
        pad_x = (in_w - round(w * scale)) / 2
        pad_y = (in_h - round(h * scale)) / 2
        boxes = out[:, :4].astype(np.float64)
        boxes[:, [0, 2]] = np.round(np.clip((boxes[:, [0, 2]] - pad_x) / scale, 0, w))
        boxes[:, [1, 3]] = np.round(np.clip((boxes[:, [1, 3]] - pad_y) / scale, 0, h))
        return [
            {"x": (x1 + x2) / 2, "y": (y1 + y2) / 2, "width": x2 - x1, "height": y2 - y1, "confidence": float(c)}
            for (x1, y1, x2, y2), c in zip(boxes, out[:, 4])
        ]

    def _fetch_roboflow_predictions(self, detect_frame):
        """Get the raw predictions list, preferring a direct model_id call.

        Direct model inference (self.roboflow_model_id, e.g. "project-slug/3")
        references an exact trained version and returns a plain, flat schema.
        The Workflow path (self.roboflow_workflow_id) is a fallback for when a
        Workflow's own custom logic is genuinely needed -- but a Workflow's
        "Project" block pins its own model version independently of whatever is
        set as "Current Model" on the Deployments page, so after retraining it
        can silently keep serving a stale version even though the UI looks updated.

        Sends a downscaled copy rather than the full captured resolution --
        benchmarked directly: a 1920x1080 frame over this HTTP round-trip caps
        out around 7 FPS (139ms/call) vs ~17 FPS (60ms/call) at 640x480. The
        model resizes internally anyway, so this cuts real end-to-end lag
        without a detection-quality cost. Predictions come back in the
        downscaled frame's coordinates, so they're rescaled to match
        detect_frame before returning, keeping every caller unaware this happened.
        """
        h, w = detect_frame.shape[:2]
        scale = min(1.0, self.roboflow_infer_size / max(h, w))
        send_frame = cv2.resize(detect_frame, (int(w * scale), int(h * scale))) if scale < 1.0 else detect_frame

        if self.roboflow_local_model is not None:
            # Ask for everything down to the lowest floor we might use --
            # _process_frame_roboflow applies the real, lock-dependent floor.
            predictions = self._infer_local(send_frame, confidence=min(self.conf, self.reacquire_conf))
        elif self.roboflow_model_id:
            result = self.roboflow_client.infer(send_frame, model_id=self.roboflow_model_id)
            predictions = result.get("predictions", [])
        else:
            result = self.roboflow_client.run_workflow(
                workspace_name=self.roboflow_workspace_name,
                workflow_id=self.roboflow_workflow_id,
                images={"image": send_frame},
            )
            predictions = result[0].get("predictions", {}).get("predictions", []) if result else []

        if scale < 1.0:
            inv_scale = 1.0 / scale
            for pred in predictions:
                pred["x"] *= inv_scale
                pred["y"] *= inv_scale
                pred["width"] *= inv_scale
                pred["height"] *= inv_scale

        return predictions

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
        # A confidence floor is applied here explicitly -- Roboflow's own
        # internal threshold is a separate setting configured in its UI, not
        # something this call controls, so low-confidence noise isn't
        # otherwise guaranteed to be filtered out. See _confidence_floor for
        # why this is lower while a lock is already active.
        predictions = [p for p in predictions if p.get("confidence", 0.0) >= self._confidence_floor()]
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
        self._draw_primary_tracking(detect_frame, box, predicted=False)
        return detect_frame

    def _process_frame_ultralytics(self, detect_frame):
        # See _confidence_floor -- lower while a lock is already active, so a
        # weak-but-spatially-consistent detection during blur can still be
        # used for reconfirmation instead of being discarded before we even
        # see it.
        results = self.model.track(
            detect_frame,
            persist=True,
            tracker=self.tracker_config,
            conf=self._confidence_floor(),
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
            self._draw_primary_tracking(detect_frame, box, predicted=False)
            return detect_frame

        filtered_boxes = result.boxes.xyxy.cpu().numpy()
        filtered_ids = result.boxes.id.int().cpu().tolist()
        self._draw_tracking(detect_frame, filtered_boxes, filtered_ids)
        return detect_frame

    def run_video(self, source, output_path=None, show_window=True, raw_output_path=None, record_fps=None, flip_horizontal=False, flip_vertical=False, stats_csv=None):
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

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
        print(f"[Camera Actual] {width}x{height} @ {fps}fps")

        # Recording writes video-encode (CPU) work on top of detection work -- a
        # high capture fps (120) is valuable for detection/motion-blur, but
        # encoding streams at a target far above what's actually achievable is
        # actively counterproductive: the frame-pacing catch-up logic below has
        # to write several duplicate frames per cycle to keep up, and each of
        # those writes costs real encoding time, slowing the next cycle and
        # needing even more catch-up -- a real feedback loop, benchmarked to
        # crash real throughput from ~10fps to ~2fps when the gap is large
        # (record_fps=30 target against a Roboflow HTTP round-trip's ~8-13fps
        # ceiling). The Roboflow backend defaults much lower than the local
        # Ultralytics one for exactly this reason -- every frame is still
        # detected on either way, just not every one written to the file.
        default_record_fps = 10 if self.use_roboflow else 30
        effective_record_fps = record_fps if record_fps else min(fps, default_record_fps)
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

        display = None
        if show_window:
            # WINDOW_NORMAL makes it resizable/draggable; the initial size is just a
            # display cap so a 1920x1080 capture doesn't overflow a laptop screen --
            # recording and detection still use the full captured (or zoomed) resolution.
            display_width = min(output_width, 1280)
            display_height = int(display_width * output_height / output_width) if output_width else output_height
            display = _FrameDisplay("Project PickleVision - Single Camera Draft", display_width, display_height)

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


        stats = _SessionStats(stats_csv) if stats_csv else None
        reader = _FrameReader(cap, live=isinstance(video_source, int))
        try:
            while True:
                frame, arrived_at = reader.read()
                if frame is None:
                    break

                if flip_horizontal or flip_vertical:
                    flip_code = -1 if (flip_horizontal and flip_vertical) else (1 if flip_horizontal else 0)
                    frame = cv2.flip(frame, flip_code)

                # Must copy before process_frame() -- it draws directly onto the
                # array it's given (or, with zoom, onto a view sharing memory with
                # this same frame), so anything not copied first ends up annotated too.
                raw_frame = frame.copy() if raw_writer is not None else None

                annotated = self.process_frame(frame)
                if stats is not None:
                    stats.frame_done(arrived_at)

                if writer is not None or raw_writer is not None:
                    expected_frames = int((time.time() - record_start) * effective_record_fps)
                    catchup_target = min(expected_frames, frames_written + max_catchup_frames_per_iteration)
                    while frames_written <= catchup_target:
                        if writer is not None:
                            writer.write(annotated)
                        if raw_writer is not None:
                            raw_writer.write(raw_frame)
                        frames_written += 1

                if display is not None:
                    display.show(annotated)
                    if display.quit_requested.is_set():
                        break
        finally:
            reader.stop()
            cap.release()
            if writer is not None:
                writer.release()
            if raw_writer is not None:
                raw_writer.release()
            if display is not None:
                display.stop()
            if stats is not None:
                stats.close()

        return self.events


@dataclass
class LineCall:
    """One bounce on the full court, merged from the two end cameras' flags of it (see DualCameraFusion)."""

    court_point: tuple[float, float]  # global court coordinates in ft (see to_global_court_point)
    call: str  # "IN" / "OUT", against the full court
    # CONFIRMED: both cameras put the ball at this court spot as it bounced --
    # it really was on the ground. SINGLE: only one camera had the ball in
    # view, so this is only as reliable as that camera's own bounce detector.
    # (A flag the other camera saw happen in the air isn't a call at all --
    # see DualCameraFusion.ignored_in_air.)
    status: str
    source_camera: str  # whose reading court_point is
    # What found it: "A"/"B" = that camera's own _detect_ball_contact,
    # "ground" = the two cameras' positions for the ball meeting on the ground.
    flagged_by: tuple[str, ...]
    frame_index: int
    # Closest the two cameras' court positions for the ball came while it was
    # flagged (ft), when both saw it -- a live read on how well the two
    # calibrations agree.
    camera_gap_ft: float | None = None
    timestamp: float = field(default_factory=time.time)


class DualCameraFusion:
    """Turns the two end cameras' bounce flags into line calls on the full court.

    Fusion rule (CLAUDE.md): the end camera whose own near half the ball
    landed in is the position source -- it has the closer, more detailed
    view. The other camera's reading is the fallback when that one didn't
    flag the bounce, e.g. a player blocked its view.

    Having two cameras also tells when the ball is on the ground. A ball on
    the ground maps to the same court point through both cameras'
    homographies; a ball in the air maps to two different points, each
    pushed away from its own camera (the homography follows the line of
    sight down to the ground) -- the higher the ball, the farther apart. So:

    - Where the OTHER camera sees the ball at the moment one camera flags a
      bounce tells a real bounce apart from a paddle hit or the top of an
      arc, which a direction change in one camera's image can't.
    - The gap between the two positions bottoming out near zero IS a bounce
      (add_ground_sample). It catches bounces _detect_ball_contact misses:
      a far-off ball traveling along the court barely moves up or down in
      an end camera's image around its bounce -- often less than the
      tracker's jitter threshold per frame at 120fps.

    One real bounce usually arrives as several flags: _detect_ball_contact
    keeps firing for a few frames after a direction change (and on jitter),
    and both cameras and the ground check may flag it. Flags within
    MERGE_WINDOW_S and MERGE_DISTANCE_FT of a bounce's first flag are merged
    into it, and it becomes a call once that window has closed.
    """

    MERGE_WINDOW_S = 0.3
    MERGE_DISTANCE_FT = 3.0
    # Once both cameras have put the ball on the ground together for longer
    # than this, it's rolling or at rest: detector jitter can still trip a
    # camera's own bounce detector, and the other camera would "confirm" it.
    GROUNDED_S = 0.25
    # Whether the ball has been in the air since the last bounce is only
    # known while both cameras see it; after this long without, it's unknown.
    AIRBORNE_MEMORY_S = 0.5

    def __init__(self, court_width=20.0, half_length=22.0, agreement_ft=3.0):
        self.court_width = court_width
        self.half_length = half_length
        self.agreement_ft = agreement_ft
        # The gap has to open up past this -- the ball clearly in the air --
        # between two bounces, so a rolling or resting ball (gap hovering
        # near zero with detector jitter) isn't a stream of bounces.
        self.airborne_gap_ft = 2 * agreement_ft
        self.calls: list[LineCall] = []
        # Bounce flags the other camera saw happen in the air: not calls.
        self.ignored_in_air = 0
        # One list of flags per bounce whose merge window is still open.
        self._pending: list[list[dict]] = []
        # Last 3 consecutive steps where both cameras detected the ball, as
        # (t, frame_index, gap_ft, point_a, point_b), for add_ground_sample.
        self._gaps: list[tuple] = []
        # Has the ball been clearly in the air since the last bounce? None =
        # unknown. A flag the other camera confirms only starts a new bounce
        # when this isn't False: otherwise the ball hasn't left the ground
        # since the last one (jitter right after a bounce, or rolling).
        self._airborne: bool | None = None
        self._both_seen_at: float | None = None
        self._grounded_since: float | None = None

    def home_camera(self, court_point):
        """The end camera whose own near half this court point is in."""
        return "A" if court_point[1] < self.half_length else "B"

    def classify(self, court_point):
        """IN/OUT against the full court (lines are in) -- not either camera's own half."""
        x, y = court_point
        return "IN" if 0.0 <= x <= self.court_width and 0.0 <= y <= 2 * self.half_length else "OUT"

    def add_ground_sample(self, t, frame_index, point_a, point_b):
        """Both cameras' court positions for the ball this step -- each a real
        detection, or None. Flags a bounce at the low point of the gap
        between them, once it's within agreement_ft of zero. Call it before
        add_flag for the same step, so the flags see the ball's latest state."""
        if point_a is None or point_b is None:
            # A missing step breaks the sequence a low point is read from.
            self._gaps.clear()
            self._grounded_since = None
            return
        if self._both_seen_at is None or t - self._both_seen_at > self.AIRBORNE_MEMORY_S:
            self._airborne = None
        self._both_seen_at = t
        gap = float(np.hypot(point_a[0] - point_b[0], point_a[1] - point_b[1]))
        self._gaps.append((t, frame_index, gap, point_a, point_b))
        del self._gaps[:-3]
        if gap > self.agreement_ft:
            self._grounded_since = None
        elif self._grounded_since is None:
            self._grounded_since = t
        if gap > self.airborne_gap_ft:
            self._airborne = True
        if len(self._gaps) < 3 or self._airborne is not True:
            return
        (_, _, before, _, _), (t_low, frame_low, low, low_a, low_b), (_, _, after, _, _) = self._gaps
        if low < before and low <= after and low <= self.agreement_ft:
            midpoint = ((low_a[0] + low_b[0]) / 2, (low_a[1] + low_b[1]) / 2)
            camera = self.home_camera(midpoint)
            self._add({
                "camera": camera, "source": "ground", "point": low_a if camera == "A" else low_b,
                "status": "CONFIRMED", "gap": low, "t": t_low, "frame_index": frame_low,
            })
            self._airborne = False

    def add_flag(self, camera, court_point, other_camera_point, t, frame_index):
        """A bounce flagged by `camera`'s own _detect_ball_contact at
        `court_point` (global ft) at time t. other_camera_point is where the
        other camera sees the ball at that moment -- a real detection, not a
        prediction -- or None if it doesn't."""
        if other_camera_point is None:
            status, gap = "SINGLE", None
        else:
            gap = float(np.hypot(court_point[0] - other_camera_point[0], court_point[1] - other_camera_point[1]))
            status = "CONFIRMED" if gap <= self.agreement_ft else "DISPUTED"
        if status == "CONFIRMED":
            rolling = self._grounded_since is not None and t - self._grounded_since > self.GROUNDED_S
            if self._airborne is False or rolling:
                return
            # Deliberately doesn't end the airborne state itself: a flag can
            # come a frame or two early, from jitter while the ball is already
            # low, and the ground check's low point just after it is the
            # precise moment of contact (it merges into the same bounce).
        self._add({"camera": camera, "source": camera, "point": court_point, "status": status, "gap": gap, "t": t, "frame_index": frame_index})

    def _add(self, flag):
        court_point, t = flag["point"], flag["t"]
        for bounce in self._pending:
            first = bounce[0]
            if t - first["t"] > self.MERGE_WINDOW_S:
                continue
            near = np.hypot(court_point[0] - first["point"][0], court_point[1] - first["point"][1]) <= self.MERGE_DISTANCE_FT
            # With only one camera seeing it, a flag's court point is the
            # ball's line of sight to the ground -- it slides along the court
            # as the ball falls -- so that camera's flags merge on time alone.
            same_view = flag["status"] == first["status"] == "SINGLE" and flag["camera"] == first["camera"]
            if near or same_view:
                bounce.append(flag)
                return
        self._pending.append([flag])

    def close_ready(self, t, close_all=False):
        """Turn every bounce whose merge window has closed by time t (or every
        pending one, at the end of a session) into a call. Returns the new calls."""
        new_calls, still_open = [], []
        for bounce in self._pending:
            if close_all or t - bounce[0]["t"] > self.MERGE_WINDOW_S:
                call = self._to_call(bounce)
                if call is None:
                    self.ignored_in_air += 1
                else:
                    new_calls.append(call)
            else:
                still_open.append(bounce)
        self._pending = still_open
        self.calls.extend(new_calls)
        return new_calls

    def _to_call(self, bounce):
        """The call for one merged bounce, or None if the other camera only
        ever saw the ball in the air for it."""
        statuses = {flag["status"] for flag in bounce}
        # One flag the other camera agreed with is enough: the rest come a
        # frame or two into the ball's rise, when the two cameras' readings
        # have already started to separate.
        if "CONFIRMED" in statuses:
            status = "CONFIRMED"
        elif "DISPUTED" in statuses:
            return None
        else:
            status = "SINGLE"
        home = self.home_camera((0.0, float(np.mean([flag["point"][1] for flag in bounce]))))
        if status == "SINGLE":
            # One camera's flags only: each court point is where its line of
            # sight through the ball meets the ground, pushed away from the
            # camera the higher the ball is -- so the flag nearest the camera
            # was taken closest to the ground. (Measured on simulated blocked
            # views with 2px jitter: 0.6ft median error, vs 7ft for the
            # earliest flag, which is often jitter during the fall.)
            best = min(bounce, key=lambda flag: flag["point"][1] if flag["camera"] == "A" else -flag["point"][1])
        else:
            # The home camera's reading, confirmed if possible -- from the
            # ground check if it caught this bounce (its low point is the
            # moment of contact; _detect_ball_contact fires a frame or more
            # after it), otherwise the earliest flag, the next closest.
            best = min(
                bounce,
                key=lambda flag: (flag["camera"] != home, flag["status"] != "CONFIRMED", flag["source"] != "ground", flag["t"]),
            )
        gaps = [flag["gap"] for flag in bounce if flag["gap"] is not None and flag["status"] == "CONFIRMED"]
        return LineCall(
            court_point=best["point"],
            call=self.classify(best["point"]),
            status=status,
            source_camera=best["camera"],
            flagged_by=tuple(sorted({flag["source"] for flag in bounce})),
            frame_index=bounce[0]["frame_index"],
            camera_gap_ft=min(gaps) if gaps else None,
        )


class DualCameraTracker:
    """Tracks with both end cameras at once -- CLAUDE.md's dual end-camera
    build. Each camera gets its own PickleVisionTracker (its own ball lock,
    prediction and person filter), calibrated for its own near half
    (--court-length 22). Their balls and bounce flags are mapped onto one
    full-court coordinate system (to_global_court_point) and the flags are
    fused into line calls (DualCameraFusion).

    Frames are taken in lockstep, one from each camera per step. From video
    files that pairs frame N with frame N; from live cameras it's each one's
    newest frame -- their clocks aren't synced, so a pair can be up to one
    frame period apart (~8ms at 120fps). Each step runs the two trackers on
    two threads, so one camera's CPU work (filtering, drawing) overlaps the
    other's GPU work: benchmarked on an RTX 4050 Laptop with rendered 720p
    footage and GPU torch, 70.5 -> 81.6 pairs/s with the person filter off,
    38.5 with it on (the default).
    """

    PANEL_HEIGHT = 360
    COURT_VIEW_HEIGHT = 300
    CAMERA_COLORS = {"A": (255, 170, 0), "B": (0, 150, 255)}  # BGR: blue, orange
    RECENT_CALLS_SHOWN = 12

    def __init__(self, tracker_a, tracker_b, court_width=20.0, half_length=22.0, agreement_ft=3.0):
        self.trackers = {"A": tracker_a, "B": tracker_b}
        for end, tracker in self.trackers.items():
            tracker.log_prefix = f"[Cam {end}] "
        self.court_width = court_width
        self.half_length = half_length
        # Court positions need both cameras calibrated -- an uncalibrated
        # tracker's "court" is just its whole frame.
        self.calibrated = tracker_a.court_calibrated and tracker_b.court_calibrated
        self.fusion = DualCameraFusion(court_width, half_length, agreement_ft)
        self.pair_index = 0
        self._events_seen = {"A": 0, "B": 0}

    def _court_point(self, end, frame_point):
        """A point in this camera's original-frame pixels -> global court ft."""
        local = self.trackers[end].court_mapper.map_point(frame_point)
        return to_global_court_point(local, end, self.half_length, self.court_width)

    def _update(self, t):
        """After both cameras' step: hand any new bounce flags to the fusion.
        Returns each camera's ball as (court point, is a real detection), or
        None, for the court view."""
        balls = {}
        for end, tracker in self.trackers.items():
            state = tracker.ball_position()
            balls[end] = None if state is None or not self.calibrated else (self._court_point(end, state[0]), state[1])
        # Only real detections count as a camera seeing the ball -- a
        # prediction bridging a gap is a guess, not evidence.
        detected = {end: ball[0] if ball is not None and ball[1] else None for end, ball in balls.items()}
        if self.calibrated:
            self.fusion.add_ground_sample(t, self.pair_index, detected["A"], detected["B"])
        for end, tracker in self.trackers.items():
            new_events = tracker.events[self._events_seen[end]:]
            self._events_seen[end] = len(tracker.events)
            if not self.calibrated:
                continue
            for event in new_events:
                court_point = self._court_point(end, tracker.frame_coords(event.landing_point))
                self.fusion.add_flag(end, court_point, detected["B" if end == "A" else "A"], t, self.pair_index)
        for call in self.fusion.close_ready(t):
            self._print_call(call)
        return balls

    @staticmethod
    def _print_call(call):
        x, y = call.court_point
        where = f"({x:.1f}, {y:.1f}) ft"
        if call.status == "CONFIRMED":
            print(
                f"[Call] {call.call} at {where} -- both cameras agree ({call.camera_gap_ft:.1f} ft apart), "
                f"position from camera {call.source_camera}"
            )
        else:
            print(f"[Call] {call.call} at {where} -- camera {call.source_camera} only (the other camera didn't have the ball)")

    def _compose(self, annotated, balls):
        """Both annotated feeds side by side, over a top-down court diagram."""
        panels = []
        for end in ("A", "B"):
            frame = annotated[end]
            height, width = frame.shape[:2]
            panel = cv2.resize(frame, (max(1, round(width * self.PANEL_HEIGHT / height)), self.PANEL_HEIGHT))
            state = self.trackers[end].ball_position()
            label = f"Camera {end}: " + ("no ball" if state is None else ("ball" if state[1] else "ball (predicted)"))
            _draw_label(panel, label, (10, 26), self.CAMERA_COLORS[end])
            panels.append(panel)
        top = np.hstack(panels)
        return np.vstack([top, self._draw_court_view(top.shape[1], balls)])

    def _draw_court_view(self, width, balls):
        """Top-down diagram of the full court -- end A on the left, end B on
        the right -- with each camera's ball (filled = detected, ring =
        predicted) and the recent line calls."""
        height = self.COURT_VIEW_HEIGHT
        view = np.full((height, width, 3), 32, np.uint8)
        text_area = 48
        length = 2 * self.half_length
        # Run-off shown past the baselines/sidelines (ft), so OUT calls stay on screen.
        run_off_y, run_off_x = 6.0, 4.0
        scale = min(width / (length + 2 * run_off_y), (height - text_area) / (self.court_width + 2 * run_off_x))
        left = (width - length * scale) / 2
        top = run_off_x * scale

        def px(court_point):
            # Court X (across) runs down the view, court Y (end A -> end B) left to right.
            return int(round(left + court_point[1] * scale)), int(round(top + court_point[0] * scale))

        def on_view(p):
            return 0 <= p[0] < width and 0 <= p[1] < height - text_area

        w, net, kitchen = self.court_width, self.half_length, 7.0
        white = (235, 235, 235)
        cv2.rectangle(view, px((0, 0)), px((w, length)), (100, 65, 30), -1)
        for a, b in (
            ((0, 0), (w, 0)), ((0, length), (w, length)),  # baselines
            ((0, 0), (0, length)), ((w, 0), (w, length)),  # sidelines
            ((0, net - kitchen), (w, net - kitchen)), ((0, net + kitchen), (w, net + kitchen)),  # kitchen lines
            ((w / 2, 0), (w / 2, net - kitchen)), ((w / 2, net + kitchen), (w / 2, length)),  # centerlines
        ):
            cv2.line(view, px(a), px(b), white, 1)
        cv2.line(view, px((-1, net)), px((w + 1, net)), (190, 190, 190), 3)  # net, posts just outside the sidelines
        # Out at the edge of the run-off, clear of calls just past the baselines.
        for end, y in (("A", 1.0 - run_off_y), ("B", length + run_off_y - 1.0)):
            label_x, label_y = px((w / 2, y))
            cv2.putText(view, end, (label_x - 6, label_y + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.CAMERA_COLORS[end], 2)

        for call in self.fusion.calls[-self.RECENT_CALLS_SHOWN:]:
            p = px(call.court_point)
            if on_view(p):
                color = (0, 200, 0) if call.call == "IN" else (0, 0, 255)
                cv2.circle(view, p, 6, color, -1 if call.status == "CONFIRMED" else 2)
        for end, ball in balls.items():
            if ball is not None and on_view(px(ball[0])):
                cv2.circle(view, px(ball[0]), 4, self.CAMERA_COLORS[end], -1 if ball[1] else 1)

        if not self.calibrated:
            status = "Not calibrated: --calibrate --court-length 22 each camera, then pass --court-corners / --court-corners2"
        elif self.fusion.calls:
            last = self.fusion.calls[-1]
            how = "both cameras agree" if last.status == "CONFIRMED" else f"camera {last.source_camera} only"
            status = f"Last call: {last.call} at ({last.court_point[0]:.1f}, {last.court_point[1]:.1f}) ft -- {how}"
            p = px(last.court_point)
            if on_view(p):
                cv2.putText(view, last.call, (p[0] + 8, p[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, white, 1)
        else:
            status = "Waiting for the first bounce"
        cv2.putText(view, status, (10, height - 28), cv2.FONT_HERSHEY_SIMPLEX, 0.5, white, 1)
        cv2.putText(
            view,
            "calls: filled = both cameras agree, ring = one camera only | small dots = each camera's ball (ring = predicted)",
            (10, height - 9),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (170, 170, 170),
            1,
        )
        return view

    def run(self, source_a, source_b, output_path=None, raw_output_a=None, raw_output_b=None, show_window=True,
            record_fps=None, flips=None, stats_csv=None):
        """Track both cameras until either source ends (or fails) or 'q' is
        pressed in the window. Returns the line calls.

        output_path records the combined view (both feeds + court diagram).
        raw_output_a/raw_output_b record each camera's unannotated frames,
        written in step, so the pair replays in sync as --source/--source2
        video files. flips maps "A"/"B" to (horizontal, vertical).
        """
        from concurrent.futures import ThreadPoolExecutor

        if str(source_a) == str(source_b):
            raise ValueError(f"--source and --source2 are the same ({source_a}) -- dual-camera mode needs two different cameras or videos")
        flips = flips or {}
        caps, live, readers, writers = {}, {}, {}, {}
        display = stats = None
        executor = ThreadPoolExecutor(max_workers=1)
        t = 0.0
        try:
            for end, source in (("A", source_a), ("B", source_b)):
                tracker = self.trackers[end]
                caps[end], live[end] = _open_video_source(source, tracker.target_width, tracker.target_height, tracker.target_fps)
                cap = caps[end]
                print(
                    f"[Camera {end}] {source}: {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x"
                    f"{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))} @ {int(cap.get(cv2.CAP_PROP_FPS)) or 30}fps"
                )
            fps = min(int(cap.get(cv2.CAP_PROP_FPS)) or 30 for cap in caps.values())

            # Same recording rules as run_video: a lower write rate than the
            # capture rate (see its comment on the encode feedback loop), and
            # frame-paced to wall-clock time so recordings play at real speed.
            default_record_fps = 10 if self.trackers["A"].use_roboflow else 30
            record_fps = record_fps if record_fps else min(fps, default_record_fps)
            outputs = {key: Path(path) for key, path in (("view", output_path), ("A", raw_output_a), ("B", raw_output_b)) if path}
            for path in outputs.values():
                path.parent.mkdir(parents=True, exist_ok=True)
            if outputs and record_fps < fps:
                print(f"[Recording] Capturing/detecting at {fps}fps, writing video at {record_fps}fps")
            record_start = time.time()
            frames_written = 0
            max_catchup_frames_per_iteration = max(1, int(record_fps))

            stats = _SessionStats(stats_csv) if stats_csv else None
            for end, cap in caps.items():
                readers[end] = _FrameReader(cap, live=live[end])
            # Merge windows need a clock that matches the footage: video files
            # can be processed faster or slower than they were recorded.
            frame_clock = not any(live.values())

            while True:
                frames, arrivals = {}, {}
                for end, reader in readers.items():
                    frames[end], arrivals[end] = reader.read()
                ended = [end for end, frame in frames.items() if frame is None]
                if ended:
                    print(f"[Dual] Camera {' and '.join(ended)} stopped delivering frames -- ending the session")
                    break
                self.pair_index += 1
                for end in frames:
                    frames[end] = _flip_frame(frames[end], *flips.get(end, (False, False)))
                # Copied before process_frame draws on the frames (see run_video).
                raw = {end: frames[end].copy() for end in ("A", "B") if end in outputs}

                future = executor.submit(self.trackers["A"].process_frame, frames["A"])
                annotated = {"B": self.trackers["B"].process_frame(frames["B"])}
                annotated["A"] = future.result()

                t = self.pair_index / fps if frame_clock else min(arrivals.values())
                balls = self._update(t)
                view = self._compose(annotated, balls)
                if stats is not None:
                    # One pair = one "frame" here: fps is pairs per second, and
                    # latency is from the older of the two frames' arrivals.
                    stats.frame_done(min(arrivals.values()))

                if outputs:
                    to_write = {"view": view, **raw}
                    if not writers:
                        for key, path in outputs.items():
                            size = (to_write[key].shape[1], to_write[key].shape[0])
                            writers[key] = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), record_fps, size)
                    expected_frames = int((time.time() - record_start) * record_fps)
                    catchup_target = min(expected_frames, frames_written + max_catchup_frames_per_iteration)
                    while frames_written <= catchup_target:
                        for key, writer in writers.items():
                            writer.write(to_write[key])
                        frames_written += 1

                if show_window:
                    if display is None:
                        display = _FrameDisplay("Project PickleVision - Dual Camera", view.shape[1], view.shape[0])
                    display.show(view)
                    if display.quit_requested.is_set():
                        break
        finally:
            executor.shutdown(wait=True)
            for reader in readers.values():
                reader.stop()
            for cap in caps.values():
                cap.release()
            for writer in writers.values():
                writer.release()
            if display is not None:
                display.stop()
            if stats is not None:
                stats.close()
            for call in self.fusion.close_ready(t, close_all=True):
                self._print_call(call)

        return self.fusion.calls


def parse_args():
    parser = argparse.ArgumentParser(description="Project PickleVision: YOLOv8 tracking prototype for ELP 120fps USB cameras -- one camera, or two end cameras with --source2")
    parser.add_argument("--source", type=str, default="0", help="Video file path or USB camera index (default: 0)")
    parser.add_argument("--model", type=str, default="yolov8n.pt", help="YOLOv8 model to load")
    parser.add_argument("--tracker", type=str, default="bytetrack.yaml", help="Tracking configuration")
    parser.add_argument("--conf", type=float, default=0.25, help="Detection confidence threshold required to acquire a FRESH lock (no ball currently tracked)")
    parser.add_argument("--reacquire-conf", type=float, default=0.1, help="Lower confidence threshold used only to reconfirm an ALREADY-tracked ball near its last known position -- spatial matching validates the weaker detection, so it's not treated as risky as accepting a fresh low-confidence detection blind")
    parser.add_argument("--target-class-id", type=int, default=32, help="Class ID to track (default 32 = COCO 'sports ball', for yolov8n.pt). A custom single-class Roboflow model almost always uses class 0 instead -- pass --target-class-id 0 when using one. Use -1 to track every detected class")
    parser.add_argument("--use-roboflow", action="store_true", help="Detect via a Roboflow Workflow on a self-hosted inference server instead of a local Ultralytics model -- for a custom-trained model when weights export isn't available on the Roboflow plan")
    parser.add_argument("--roboflow-local", action="store_true", help="Run the Roboflow model (--roboflow-model-id) in-process on this PC -- no inference server or Docker. Weights download once into ./model_cache using the API key, then load offline")
    parser.add_argument("--roboflow-api-url", type=str, default="http://localhost:9001", help="Self-hosted Roboflow inference server URL")
    parser.add_argument("--roboflow-api-key", type=str, default="ZceKVfYE1Cvm0jqDdA1F", help="Roboflow API key")
    parser.add_argument("--roboflow-workspace", type=str, default="franzs-workspace-utuz0", help="Roboflow workspace name")
    parser.add_argument("--roboflow-model-id", type=str, default="pickleball-prototype/3", help="Roboflow model ID as 'project-slug/version' -- calls the model directly (preferred: exact version, simple response). Pass an empty string to use --roboflow-workflow-id instead")
    parser.add_argument("--roboflow-workflow-id", type=str, default=None, help="Roboflow workflow ID -- only used if --roboflow-model-id is empty. Note a Workflow's model reference is pinned separately from the project's 'Current Model' setting and can silently lag behind after retraining")
    parser.add_argument("--roboflow-infer-size", type=int, default=640, help="Downscale the frame to this max dimension before sending to the Roboflow server (default 640). Benchmarked: 1920x1080 caps around 7 FPS over the network round-trip vs ~17 FPS at 640x480 -- lower this further for more speed at the cost of long-distance detection detail")
    parser.add_argument("--iou", type=float, default=0.5, help="IoU threshold for NMS")
    parser.add_argument("--output", type=str, default=None, help="Optional annotated output video path (boxes/labels/trajectory baked in -- for review, not training). With --source2: the combined view, both feeds plus the court diagram")
    parser.add_argument("--raw-output", type=str, default=None, help="Optional unannotated output video path, safe to upload to Roboflow for annotation/training. With --source2 this is --source's footage (see --raw-output2)")
    parser.add_argument("--record-fps", type=int, default=None, help="FPS to WRITE recorded video at (default: min(camera fps, 30)). Capture/detection still runs at full --fps; only the saved file rate is lowered, since encoding two full-res streams at 120fps is heavy CPU work")
    parser.add_argument("--show", action="store_true", default=True, help="Display annotated frames in real-time")
    parser.add_argument("--fps", type=int, default=120, help="Target camera FPS (default: 120, for the ELP camera; a camera that can't do it falls back to its own max)")
    parser.add_argument("--width", type=int, default=1280, help="Target camera width in pixels (default: 1280 -- 720p, the target for multi-camera headroom; 1920 for 1080p). Calibrate at the same size you track at")
    parser.add_argument("--height", type=int, default=720, help="Target camera height in pixels (default: 720; 1080 for 1080p)")
    parser.add_argument("--court-corners", type=str, default=None, help="Court calibration: x1 y1 x2 y2 x3 y3 x4 y4 (TL TR BR BL)")
    parser.add_argument("--max-missed-frames", type=int, default=15, help="Frames to keep extrapolating the ball's position through a detection gap (e.g. motion blur) before dropping the track")
    parser.add_argument("--stats", action="store_true", help="Log performance and resources: startup time, fps, frame latency, CPU/RAM, GPU utilization/temperature/power. Prints every 5s plus a summary at the end, and saves a per-second CSV (see --stats-csv). With --source2, fps counts camera pairs per second")
    parser.add_argument("--stats-csv", type=str, default=None, help="CSV path for --stats (default: logs/session_<date>_<time>_cam<source>.csv)")
    parser.add_argument("--smoothing", type=float, default=0.5, help="Trajectory smoothing, 0-1 (default 0.5). 0 = raw detector positions (jittery); higher = smoother path but lags a fast ball more")
    parser.add_argument("--bounce-min-dy", type=float, default=3.0, help="Per-frame vertical movement (px) below which a Y change is treated as detector jitter rather than a bounce (default 3)")
    parser.add_argument("--match-distance", type=float, default=250.0, help="Max pixel distance a new detection can be from the ball's last known position to be accepted as the same ball")
    parser.add_argument("--no-color-filter", action="store_true", help="Disable the optic yellow-green color check used to prefer the real ball over other round objects")
    parser.add_argument("--no-exclude-people", action="store_true", help="Disable rejecting ball candidates/predictions that land on a detected person (e.g. a player wearing ball-colored clothing). Runs a small local person detector alongside the main detection backend")
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
    parser.add_argument("--court-length", type=float, default=None, help="Real-world length (ft) of the calibrated region: 44 for a full court, 22 to scope to just one half (baseline to net). Default 44, or 22 with --source2 -- the two end cameras are each calibrated for their own half")
    parser.add_argument("--court-width", type=float, default=20.0, help="Real-world width (ft) of the calibrated region (default 20, standard doubles court width)")
    parser.add_argument("--zoom", action="store_true", help="Digitally zoom: crop detection/display/recording to the calibrated court region (requires --court-corners)")
    parser.add_argument("--zoom-padding", type=float, default=0.15, help="Padding around the calibrated court corners when zoomed, as a fraction of the court's width/height (default 0.15)")
    parser.add_argument(
        "--check-alignment",
        action="store_true",
        help="Dual end-camera setup tool (needs --source2), then exit without tracking. Live view of both cameras with center/level guides for aiming each one straight down the court's centerline. With --court-corners/--court-corners2 it also draws each calibrated court (the far half too, extrapolated past the net) with an aiming readout, and you click the same real-world point in both feeds to verify the two half-court calibrations agree",
    )
    parser.add_argument("--source2", type=str, default=None, help="Second end camera: video file path or USB camera index. Tracks both cameras at once and fuses them onto one full court -- calibrate each for its own half (--calibrate --court-length 22) and pass --court-corners for --source, --court-corners2 for this one. Also used by --check-alignment")
    parser.add_argument("--court-corners2", type=str, default=None, help="Second camera's half-court calibration (same format as --court-corners), for dual-camera tracking and --check-alignment")
    parser.add_argument("--flip-horizontal2", action="store_true", help="Flip the second camera's feed horizontally, for dual-camera tracking and --check-alignment")
    parser.add_argument("--flip-vertical2", action="store_true", help="Flip the second camera's feed vertically, for dual-camera tracking and --check-alignment")
    parser.add_argument("--raw-output2", type=str, default=None, help="With --source2: unannotated footage from --source2 (--raw-output saves --source's). Both are written frame-for-frame in step, so the pair replays in sync as --source/--source2 video files")
    parser.add_argument("--bounce-agreement-ft", type=float, default=3.0, help="With --source2: how close (ft) the two cameras' court positions for the ball must be when one flags a bounce for it to count as confirmed by both (default 3). A ball on the ground maps to the same court spot from both ends; one in the air doesn't, so a bounce the other camera disagrees with is ignored as a hit or the top of an arc")
    parser.add_argument("--alignment-tolerance-ft", type=float, default=2.0, help="Max discrepancy (ft) between the two cameras' reports of the same point before --check-alignment flags them as misaligned (default 2.0)")
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


def _court_reference_lines(mapper: CourtMapper, include_far_half: bool = False):
    """Standard court reference lines (baselines, sidelines, net, kitchen, centerline),
    projected from real-world court coordinates back into image space via the
    inverse homography -- used to visually sanity-check a calibration.

    Follows --calibrate's click order: the TOP edge (y_min) is the far
    baseline for a full court, or the net for a half court; the BOTTOM edge
    (y_max) is the near baseline either way. include_far_half adds, for a
    half-court calibration, the other half extrapolated past the net.
    """
    x_min = float(np.min(mapper.dst_points[:, 0]))
    x_max = float(np.max(mapper.dst_points[:, 0]))
    y_min = float(np.min(mapper.dst_points[:, 1]))
    y_max = float(np.max(mapper.dst_points[:, 1]))
    width = x_max - x_min
    center_x = x_min + width / 2.0

    segments = [
        ((x_min, y_min), (x_max, y_min)),  # far edge: far baseline, or the net for a half court
        ((x_min, y_max), (x_max, y_max)),  # near baseline
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
        # Half-court calibration (net -> near baseline): the far edge IS the
        # net, and this half's kitchen line is 7ft in front of it.
        net_y = y_min
        kitchen_y = net_y + 7.0
        segments += [
            ((x_min, net_y), (x_max, net_y)),
            ((x_min, kitchen_y), (x_max, kitchen_y)),
            ((center_x, kitchen_y), (center_x, y_max)),
        ]
        if include_far_half:
            # Where this calibration puts the other half's lines -- they
            # should land on the real ones too, since the other end's
            # camera takes over there.
            far_baseline_y = net_y - (y_max - y_min)
            far_kitchen_y = net_y - 7.0
            segments += [
                ((x_min, far_baseline_y), (x_max, far_baseline_y)),
                ((x_min, far_kitchen_y), (x_max, far_kitchen_y)),
                ((x_min, far_baseline_y), (x_min, net_y)),
                ((x_max, far_baseline_y), (x_max, net_y)),
                ((center_x, far_kitchen_y), (center_x, far_baseline_y)),
            ]

    endpoints = np.array(segments, dtype=np.float32).reshape(-1, 1, 2)
    inv_h = np.linalg.inv(mapper.h_matrix)
    mapped = cv2.perspectiveTransform(endpoints, inv_h).reshape(-1, 2, 2)
    return [(tuple(pair[0]), tuple(pair[1])) for pair in mapped]


def calibrate_court_corners(source, save_path: str | None = None, court_width: float = 20.0, court_length: float = 44.0, flip_horizontal=False, flip_vertical=False, target_width: int = 1920, target_height: int = 1080, target_fps: int = 120):
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

    # Must match run_video's camera config exactly -- calibrating against a
    # different resolution/FOV than the one actually used for tracking makes
    # every clicked corner meaningless, since USB cameras commonly change their
    # field of view (not just scale) between resolution/codec modes.
    if isinstance(video_source, int):
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, target_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, target_height)
        cap.set(cv2.CAP_PROP_FPS, target_fps)
        actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"[Camera Config] Requesting {target_width}x{target_height} @ {target_fps}fps (MJPEG) -> got {actual_width}x{actual_height}")

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


def _aim_readout(mapper: CourtMapper, frame_width: int):
    """How far a half-court-calibrated end camera is from looking straight
    down the court's centerline, from its calibration. Measured against the
    image's vertical center line, so it assumes the lens is centered on the
    sensor (true of practically every camera):

    - net_offset_px / baseline_offset_px: where the calibrated centerline
      crosses the net and the near baseline (+ = right of center) -- the
      cyan centerline against the magenta guide.
    - far_offset_px: where the court's long lines converge (their vanishing
      point). Only aiming (pan) moves it, not where the camera stands.
    - lateral_ft: how far the camera stands right (+) or left of the
      centerline -- standing off to one side is what slants the centerline.
    - tilt_deg: the net line's tilt (+ = its right end lower).

    A level camera centered on the centerline and aimed straight along it
    reads 0 on all of them.
    """
    x_min, x_max = float(np.min(mapper.dst_points[:, 0])), float(np.max(mapper.dst_points[:, 0]))
    y_min, y_max = float(np.min(mapper.dst_points[:, 1])), float(np.max(mapper.dst_points[:, 1]))
    center_x = (x_min + x_max) / 2.0
    inv_h = np.linalg.inv(mapper.h_matrix)
    court = np.array(
        [[center_x, y_min], [center_x, y_max], [x_min, y_min], [x_max, y_min], [x_min, y_max], [x_max, y_max]],
        dtype=np.float32,
    )
    net_center, baseline_center, net_left, net_right, baseline_left, baseline_right = cv2.perspectiveTransform(
        court.reshape(-1, 1, 2), inv_h
    ).reshape(-1, 2)
    vanishing = inv_h @ np.array([0.0, 1.0, 0.0])  # the court's long direction, at infinity
    far_u = vanishing[0] / vanishing[2] if abs(vanishing[2]) > 1e-9 else net_center[0]
    # The baseline is square to a straight-aimed camera, so its known width
    # gives the image scale there for turning the slant into feet.
    px_per_ft = np.hypot(*(baseline_right - baseline_left)) / (x_max - x_min)
    return {
        "net_offset_px": float(net_center[0] - frame_width / 2.0),
        "baseline_offset_px": float(baseline_center[0] - frame_width / 2.0),
        "far_offset_px": float(far_u - frame_width / 2.0),
        # The centerline's near end swings toward the side the line is on,
        # i.e. away from the side the camera stands on.
        "lateral_ft": float(-(baseline_center[0] - far_u) / px_per_ft),
        "tilt_deg": float(np.degrees(np.arctan2(net_right[1] - net_left[1], net_right[0] - net_left[0]))),
    }


def _aim_advice(readout, frame_width, pan_tolerance_frac=0.02, tilt_tolerance_deg=1.0, lateral_tolerance_ft=0.5):
    """The next physical adjustment to make, from _aim_readout (directions as
    seen from behind the camera), or None once it's aimed straight. One at a
    time, in this order: a pan also tilts the net and slants the centerline,
    and a roll slants the centerline too, so each reading is only
    trustworthy once the ones before it are fixed."""
    if abs(readout["far_offset_px"]) > frame_width * pan_tolerance_frac:
        return f"pan {'right' if readout['far_offset_px'] > 0 else 'left'}"
    if abs(readout["tilt_deg"]) > tilt_tolerance_deg:
        return f"rotate {'clockwise' if readout['tilt_deg'] > 0 else 'counter-clockwise'} ~{abs(readout['tilt_deg']):.1f} deg"
    if abs(readout["lateral_ft"]) > lateral_tolerance_ft:
        return f"move {'left' if readout['lateral_ft'] > 0 else 'right'} ~{abs(readout['lateral_ft']):.1f} ft"
    return None


def _draw_aim_guides(frame):
    """Aiming guides on an end camera's live feed: it's looking straight down
    the court once the court's centerline runs along the vertical center
    guide and the net and baselines lie level with the horizontal ones."""
    height, width = frame.shape[:2]
    color = (255, 0, 255)
    cv2.line(frame, (width // 2, 0), (width // 2, height), color, 1)
    for frac in (0.25, 0.5, 0.75):
        y = int(height * frac)
        for x in range(0, width, 24):  # dashed, to stay out of the court lines' way
            cv2.line(frame, (x, y), (min(x + 12, width), y), color, 1)


def check_dual_camera_alignment(
    source_a,
    source_b,
    corners_a: str | None = None,
    corners_b: str | None = None,
    court_width: float = 20.0,
    half_length: float = 22.0,
    tolerance_ft: float = 2.0,
    flip_horizontal_a: bool = False,
    flip_vertical_a: bool = False,
    flip_horizontal_b: bool = False,
    flip_vertical_b: bool = False,
    target_width: int = 1920,
    target_height: int = 1080,
    target_fps: int = 120,
):
    """Live setup tool for the 2 end cameras (one behind each baseline, facing
    each other, centered on the centerline), in two stages:

    1. Aim -- no calibration needed. Each feed shows a vertical center guide
       and dashed level guides: turn each camera until the court's centerline
       runs up the center guide and the net sits level.
    2. Verify -- needs both cameras calibrated for their own half (--calibrate
       --court-length <half_length>). Each calibrated court is drawn over its
       feed in cyan, including the far half extrapolated past the net, which
       should land on the real lines too, with a readout of how far the
       calibrated centerline is from the center guide and what to adjust
       (then recalibrate: a calibration describes where the camera was when
       its corners were clicked). Then have a person or cone stand at a spot
       visible to BOTH cameras and click it in each window: the two
       calibrations should put it at the same court position
       (to_global_court_point / check_camera_alignment). Check a few spots,
       including near a sideline -- a spot on the centerline can't reveal a
       left/right mix-up.

    Clicking either window replaces just that window's point. Press 'r' to
    clear both, 'q' to quit.
    """
    mappers = {
        end: CourtMapper(src_points=CourtMapper.parse_corners(corners), court_width=court_width, court_length=half_length)
        if corners
        else None
        for end, corners in (("A", corners_a), ("B", corners_b))
    }
    flips = {"A": (flip_horizontal_a, flip_vertical_a), "B": (flip_horizontal_b, flip_vertical_b)}
    windows = {"A": "Camera A (end)", "B": "Camera B (end)"}
    clicked: dict[str, list[tuple[int, int]]] = {"A": [], "B": []}

    caps = {}
    try:
        for end, source in (("A", source_a), ("B", source_b)):
            caps[end], _ = _open_video_source(source, target_width, target_height, target_fps)
    except FileNotFoundError:
        for cap in caps.values():
            cap.release()
        raise

    def on_click(end):
        def handler(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                clicked[end][:] = [(x, y)]

        return handler

    for end, window in windows.items():
        cv2.namedWindow(window)
        cv2.setMouseCallback(window, on_click(end))

    print(f"Camera alignment: half_length={half_length:.1f}ft, tolerance={tolerance_ft:.1f}ft")
    print("1) Aim: turn each camera until the court's centerline runs up the magenta center line and the net sits level.")
    if all(mapper is not None for mapper in mappers.values()):
        print("2) Verify: have a person/cone stand at one spot visible to BOTH cameras.")
        print(f"   Click it in '{windows['A']}', then the SAME spot in '{windows['B']}'. Try a spot near a sideline too.")
    else:
        missing = " and ".join(end for end, mapper in mappers.items() if mapper is None)
        print(f"   Camera {missing} not calibrated -- calibrate with --calibrate --court-length {half_length:g} for the calibration checks.")
    print("Keys: 'r' reset points | 'q' quit")

    try:
        while True:
            frames = {}
            for end, cap in caps.items():
                ok, frame = cap.read()
                if not ok:
                    break
                frames[end] = _flip_frame(frame, *flips[end])
            if len(frames) < 2:
                print("Failed to read from one of the cameras.")
                break

            result = None
            if all(clicked[end] and mappers[end] is not None for end in ("A", "B")):
                result = check_camera_alignment(
                    mappers["A"].map_point(clicked["A"][0]),
                    mappers["B"].map_point(clicked["B"][0]),
                    half_length=half_length,
                    tolerance_ft=tolerance_ft,
                    court_width=court_width,
                )

            for end, display in frames.items():
                _draw_aim_guides(display)
                mapper = mappers[end]
                if mapper is not None:
                    for (lx1, ly1), (lx2, ly2) in _court_reference_lines(mapper, include_far_half=True):
                        cv2.line(display, (int(lx1), int(ly1)), (int(lx2), int(ly2)), (255, 255, 0), 1)
                    aim = _aim_readout(mapper, display.shape[1])
                    advice = _aim_advice(aim, display.shape[1])
                    side = "right" if aim["lateral_ft"] > 0 else "left"
                    readout = [
                        f"Centerline {aim['net_offset_px']:+.0f}px at net, {aim['baseline_offset_px']:+.0f}px at baseline"
                        f" | net tilt {aim['tilt_deg']:+.1f} deg | camera {abs(aim['lateral_ft']):.1f} ft {side} of centerline",
                        "AIMED STRAIGHT" if advice is None else f"Next: {advice}, then recalibrate and check again",
                    ]
                    readout_color = (0, 200, 0) if advice is None else (0, 215, 255)
                else:
                    readout = ["Aim: centerline on the magenta line, net level", f"Not calibrated (--calibrate --court-length {half_length:g})"]
                    readout_color = (0, 215, 255)
                for i, line in enumerate(readout):
                    _draw_label(display, line, (10, 28 + 26 * i), readout_color)

                if clicked[end]:
                    point = clicked[end][0]
                    cv2.circle(display, point, 8, (0, 0, 255), -1)
                    if mapper is not None:
                        court_x, court_y = to_global_court_point(mapper.map_point(point), end, half_length, court_width)
                        _draw_label(display, f"court=({court_x:.1f}, {court_y:.1f})ft", (point[0] + 12, point[1] - 12), (0, 0, 255), 0.5, 1)
                elif all(mapper is not None for mapper in mappers.values()):
                    _draw_label(display, f"Click the reference point (camera {end})", (10, 28 + 26 * len(readout)), (0, 255, 255))

                if result is not None:
                    color = (0, 200, 0) if result["aligned"] else (0, 0, 255)
                    lines = [
                        f"{'ALIGNED' if result['aligned'] else 'MISALIGNED'} -- off by {result['distance_ft']:.2f} ft (tol {tolerance_ft:.1f})",
                        f"dx={result['dx']:+.2f}ft (across) dy={result['dy']:+.2f}ft (along the court)",
                    ]
                    if not result["aligned"]:
                        # A left/right mirror mix-up is off by 2*(10 - x) ft -- big,
                        # and growing toward the sidelines; a stale or sloppy
                        # calibration is off by a few feet anywhere.
                        lines.append(
                            "Mostly across: recalibrate (camera moved?); one feed flipped if it grows near the sidelines"
                            if abs(result["dx"]) > abs(result["dy"])
                            else "Mostly along the court: recalibrate (camera moved?); both need --court-length 22"
                        )
                    for i, line in enumerate(lines):
                        _draw_label(display, line, (10, display.shape[0] - 12 - 26 * (len(lines) - 1 - i)), color)

                cv2.imshow(windows[end], display)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("r"):
                clicked["A"].clear()
                clicked["B"].clear()
    finally:
        for cap in caps.values():
            cap.release()
        cv2.destroyAllWindows()


def _build_tracker(args, court_corners, camera_id="cam0", shared_roboflow_model=None):
    """A PickleVisionTracker configured from the command line, calibrated with
    court_corners (a --court-corners string), or uncalibrated if it's None."""
    court_points = CourtMapper.parse_corners(court_corners) if court_corners else None
    court_mapper = (
        CourtMapper(src_points=court_points, court_width=args.court_width, court_length=args.court_length)
        if court_points is not None
        else None
    )

    return PickleVisionTracker(
        model_name=args.model,
        tracker_config=args.tracker,
        conf=args.conf,
        reacquire_conf=args.reacquire_conf,
        iou=args.iou,
        court_mapper=court_mapper,
        target_fps=args.fps,
        target_width=args.width,
        target_height=args.height,
        max_missed_frames=args.max_missed_frames,
        max_match_distance=args.match_distance,
        require_ball_color=not args.no_color_filter,
        exclude_people=not args.no_exclude_people,
        ball_color_min_ratio=args.ball_color_min_ratio,
        min_box_dimension=args.min_box_dimension,
        min_aspect_ratio=args.min_aspect_ratio,
        device=args.device,
        imgsz=args.imgsz,
        camera_id=camera_id,
        zoom_to_court=args.zoom,
        zoom_padding=args.zoom_padding,
        target_class_id=None if args.target_class_id == -1 else args.target_class_id,
        use_roboflow=args.use_roboflow,
        roboflow_api_url=args.roboflow_api_url,
        roboflow_api_key=args.roboflow_api_key,
        roboflow_workspace_name=args.roboflow_workspace,
        roboflow_model_id=args.roboflow_model_id,
        roboflow_workflow_id=args.roboflow_workflow_id,
        roboflow_infer_size=args.roboflow_infer_size,
        roboflow_local=args.roboflow_local,
        smoothing=args.smoothing,
        bounce_min_dy=args.bounce_min_dy,
        shared_roboflow_model=shared_roboflow_model,
    )


def _warm_up(tracker, args):
    print(f"[Device] Running inference on: {tracker.device}")
    if tracker.roboflow_local_model is not None:
        # Warm up here so the one-time TensorRT engine build (or CUDA init)
        # happens before the camera opens, not as a multi-minute freeze on
        # the first live frame.
        tracker.roboflow_local_model.infer(np.zeros((args.height, args.width, 3), np.uint8))
        print(f"[Device] Model backend: {tracker.roboflow_local_model.onnx_session.get_providers()[0]}")


def _run_dual_camera(args, source):
    """--source2: track both end cameras at once (DualCameraTracker)."""
    if args.target_class_id == -1:
        raise SystemExit("--source2 follows one ball per camera, so --target-class-id -1 (track every class) isn't supported with it")
    missing = [flag for flag, corners in (("--court-corners", args.court_corners), ("--court-corners2", args.court_corners2)) if not corners]
    if missing:
        print(
            f"[Dual] No {' or '.join(missing)}: tracking both feeds, but court positions and line calls need both "
            "cameras calibrated (--calibrate --court-length 22 on each)"
        )

    tracker_a = _build_tracker(args, args.court_corners, camera_id="A")
    # One copy of the detection model serves both cameras (see _LOCAL_MODEL_LOCK).
    tracker_b = _build_tracker(args, args.court_corners2, camera_id="B", shared_roboflow_model=tracker_a.roboflow_local_model)
    _warm_up(tracker_a, args)

    dual = DualCameraTracker(
        tracker_a, tracker_b, court_width=args.court_width, half_length=args.court_length, agreement_ft=args.bounce_agreement_ft
    )
    calls = dual.run(
        source_a=source,
        source_b=args.source2,
        output_path=args.output,
        raw_output_a=args.raw_output,
        raw_output_b=args.raw_output2,
        show_window=args.show,
        record_fps=args.record_fps,
        flips={"A": (args.flip_horizontal, args.flip_vertical), "B": (args.flip_horizontal2, args.flip_vertical2)},
        stats_csv=(args.stats_csv or f"logs/session_{time.strftime('%Y%m%d_%H%M%S')}_dual.csv") if args.stats else None,
    )

    confirmed = [call for call in calls if call.status == "CONFIRMED"]
    print(
        f"\n[Summary] {len(calls)} line calls: {len(confirmed)} confirmed by both cameras, "
        f"{len(calls) - len(confirmed)} from one camera only (less reliable)."
    )
    if confirmed:
        gap = float(np.median([call.camera_gap_ft for call in confirmed]))
        print(f"[Summary] Median gap between the two cameras' positions on confirmed bounces: {gap:.2f} ft (how well the calibrations agree)")
    if dual.fusion.ignored_in_air:
        print(
            f"[Summary] {dual.fusion.ignored_in_air} bounce flags ignored: the other camera saw the ball in the air (a hit or the top "
            "of an arc). If real bounces are being ignored, check the calibrations agree (--check-alignment) or raise --bounce-agreement-ft."
        )


def main():
    args = parse_args()
    if args.court_length is None:
        # Each end camera in the dual setup is calibrated for its own half.
        args.court_length = 22.0 if (args.source2 is not None and not args.calibrate) else 44.0

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
            target_width=args.width,
            target_height=args.height,
            target_fps=args.fps,
        )
        return

    if args.check_alignment:
        if not args.source2:
            raise SystemExit("--check-alignment needs --source2, the second end camera (add --court-corners/--court-corners2 for the calibration checks)")
        check_dual_camera_alignment(
            source,
            args.source2,
            args.court_corners,
            args.court_corners2,
            court_width=args.court_width,
            half_length=args.court_length,
            tolerance_ft=args.alignment_tolerance_ft,
            flip_horizontal_a=args.flip_horizontal,
            flip_vertical_a=args.flip_vertical,
            flip_horizontal_b=args.flip_horizontal2,
            flip_vertical_b=args.flip_vertical2,
            target_width=args.width,
            target_height=args.height,
            target_fps=args.fps,
        )
        return

    if args.source2 is not None:
        _run_dual_camera(args, source)
        return

    tracker = _build_tracker(args, args.court_corners)
    _warm_up(tracker, args)

    events = tracker.run_video(
        source=source,
        output_path=args.output,
        show_window=args.show,
        raw_output_path=args.raw_output,
        record_fps=args.record_fps,
        flip_horizontal=args.flip_horizontal,
        flip_vertical=args.flip_vertical,
        stats_csv=(
            args.stats_csv
            or f"logs/session_{time.strftime('%Y%m%d_%H%M%S')}_cam{Path(str(args.source)).stem}.csv"
        ) if args.stats else None,
    )
    print(f"\n[Summary] Tracked {len(events)} candidate ball events.")


if __name__ == "__main__":
    main()
