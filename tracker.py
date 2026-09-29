import argparse
import contextlib
import io
import json
import os
import threading
import time

# Reference point for --stats "startup time" (launch -> first tracked frame),
# taken before the heavy imports below so their load time is included.
_PROCESS_START = time.perf_counter()

import warnings
from collections import defaultdict, deque
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


def _timestamps_path(video_path):
    """Where --record-raw saves a video's per-frame capture times: beside it,
    raw1.avi -> raw1.timestamps.csv."""
    path = Path(video_path)
    return path.with_name(path.stem + ".timestamps.csv")


def _load_timestamps(source):
    """Each frame's capture time (s) for a video --record-raw saved, or None
    (a live camera, or a video without them)."""
    if not isinstance(source, (str, Path)):
        return None
    path = _timestamps_path(source)
    if not path.exists():
        return None
    return np.loadtxt(path, delimiter=",", skiprows=1, ndmin=2)[:, 1]


def _delivered_fps(times):
    """The frame rate a camera really delivered, from its frames' capture
    times -- a video file's own fps is only the rate it was asked for (an ELP
    set to 120 delivers 60 or less when dim light lengthens its exposure).
    The average over the whole video: frames read without a driver timestamp
    are stamped on arrival, and some cameras hand them over in bursts."""
    if len(times) < 2 or times[-1] <= times[0]:
        return None
    return round((len(times) - 1) / float(times[-1] - times[0]))


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

    notify (a threading.Event) is set whenever a frame (or the end) arrives --
    so one loop can wait on several cameras at once (DualCameraTracker).
    """

    def __init__(self, cap, live, queue_size=4, notify=None):
        import queue
        import threading

        self.cap = cap
        self.live = live
        self._notify = notify
        self._queue = queue.Queue(maxsize=1 if live else queue_size)
        self._stopped = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        import queue

        while not self._stopped.is_set():
            try:
                ok, frame = self.cap.read()
            except Exception as error:
                # read() can throw instead of returning False -- seen (a
                # cv2.error) on the built-in webcam when asked for a size it
                # doesn't support. Treated as the source failing; otherwise
                # this thread dies and the tracking loop waits forever for
                # its next frame.
                print(f"[Camera] Reading a frame failed: {error!r}")
                ok, frame = False, None
            # Arrival time rides along with the frame, so --stats can measure
            # latency from capture to tracked, queue wait included -- and so
            # does its capture time: a camera's driver (Windows MSMF) stamps
            # each frame on the same clock as perf_counter. It can be 80-270 ms
            # before arrival (measured on a laptop webcam, varying frame to
            # frame), so two cameras' arrival times don't line their frames up.
            arrival = time.perf_counter()
            captured = arrival
            if ok and self.live:
                driver_time = self.cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
                if 0.0 <= arrival - driver_time < 2.0:
                    captured = driver_time
            item = (frame, arrival, captured) if ok else (None, None, None)
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
            if self._notify is not None:
                self._notify.set()
            if not ok:
                return

    def read(self):
        """Next (frame, arrival_time, capture_time), or (None, None, None) once
        the source is exhausted/failed. (A video file's capture_time is just
        its arrival; _SyncedFrames gives it the real one.)"""
        return self._queue.get()

    def poll(self):
        """Like read(), but None right away if no frame is waiting."""
        import queue

        try:
            return self._queue.get_nowait()
        except queue.Empty:
            return None

    def stop(self):
        self._stopped.set()
        self._thread.join(timeout=1.0)


class _SyncedFrames:
    """Each step's new frames from the two end cameras, in time order.

    Live cameras: whichever have delivered a frame since the last step --
    normally both, but a slower camera (an ELP drops from 120fps to 60 or
    less when dim light lengthens its exposure) is just processed less often,
    rather than holding the other one to its pace. Step time = arrival.

    Video files: frames in the order they were captured, by each file's
    capture times (saved beside it by --record-raw) or else frame number /
    fps; frames within half a frame period of each other are one step.
    Pairing frame N with frame N instead drifts apart -- two cameras never
    deliver exactly the same rate, and each drops frames of its own -- and
    breaks outright with cameras at different rates.
    """

    def __init__(self, caps, live, fps, timestamps):
        import threading

        self.live = live
        self._arrived = threading.Event()
        self.readers = {end: _FrameReader(cap, live=live[end], notify=self._arrived) for end, cap in caps.items()}
        self._fps = fps
        self._times = timestamps
        self._count = {end: 0 for end in caps}
        self._buffered = {}
        self._tolerance = 0.5 / max(fps.values())

    def next(self):
        """({camera: (frame, arrival, capture time)} for the cameras with a new
        frame, step time in s) -- or (None, [the cameras whose source ended])."""
        if any(self.live.values()):
            return self._next_live()
        return self._next_file()

    def _next_live(self):
        new, waited = {}, 0.0
        while not new:
            if not self._arrived.wait(timeout=1.0):
                waited += 1.0
                if waited == 3.0:
                    print("[Dual] No frames from either camera for 3 s -- still waiting (unplugged? 'q' to quit)")
                continue
            self._arrived.clear()
            for end, reader in self.readers.items():
                item = reader.poll()
                if item is not None:
                    new[end] = item
        ended = [end for end, (frame, _, _) in new.items() if frame is None]
        if ended:
            return None, ended
        return new, min(captured for _, _, captured in new.values())

    def _file_time(self, end):
        times, index = self._times.get(end), self._count[end]
        if times is not None and index < len(times):
            return float(times[index])
        return index / self._fps[end]

    def _next_file(self):
        for end, reader in self.readers.items():
            if end not in self._buffered:
                self._buffered[end] = (reader.read(), self._file_time(end))
                self._count[end] += 1
        ended = [end for end, (item, _) in self._buffered.items() if item[0] is None]
        if ended:
            return None, ended
        t = min(frame_time for _, frame_time in self._buffered.values())
        new = {}
        for end, ((frame, arrival, _), frame_time) in list(self._buffered.items()):
            if frame_time <= t + self._tolerance:
                new[end] = (frame, arrival, frame_time)
                del self._buffered[end]
        return new, t

    def stop(self):
        for reader in self.readers.values():
            reader.stop()


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


class _BackgroundVideoWriter:
    """cv2.VideoWriter on its own thread, for recording the tracker's view
    (--output, --raw-output) while it tracks.

    Encoding on the tracking loop is what made recording inside the tracker
    lag (on Franz's laptop, bad enough to fall back to screen recording):
    each frame waited for the encoder, and the combined dual-camera view
    (2560x1320) takes tens of ms to encode. Here write() only queues the
    frame; the thread encodes it. `copies` writes a frame more than once --
    the loop's frame pacing, so the video plays at real speed however fast
    tracking runs. If encoding falls ~2 s behind, frames are dropped (and
    counted) rather than slowing tracking down.
    """

    def __init__(self, path, fps, size, fourcc="mp4v"):
        import queue
        import threading

        self.path = Path(path)
        self._writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*fourcc), fps, size)
        if not self._writer.isOpened():
            raise SystemExit(f"Couldn't open {path} for writing")
        self._queue: "queue.Queue" = queue.Queue(maxsize=max(4, int(2 * fps)))
        self.written = self.dropped = 0
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def write(self, frame, copies=1):
        """Queue `frame` to be written `copies` times. The caller must not
        draw on it afterwards."""
        import queue

        try:
            self._queue.put_nowait((frame, copies))
        except queue.Full:
            self.dropped += copies

    def _run(self):
        while True:
            item = self._queue.get()
            if item is None:
                return
            frame, copies = item
            for _ in range(copies):
                self._writer.write(frame)
            self.written += copies

    def release(self):
        self._queue.put(None)
        self._thread.join()
        self._writer.release()
        dropped = f", {self.dropped} dropped (encoding fell behind)" if self.dropped else ""
        print(f"[Recording] {self.written} frames saved{dropped} -> {self.path}")


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
    # The frame the ball touched down in -- frame_index is when it was
    # recognized, a few frames later (the detector needs the rise too).
    landing_frame: int | None = None
    line_call: str = "UNKNOWN"
    timestamp: float = field(default_factory=time.time)
    # Tags which camera produced this event: "A"/"B" in dual-camera mode
    # (--source2), where DualCameraTracker runs one PickleVisionTracker per end
    # camera and fuses their events into line calls on the full court.
    camera_id: str = "cam0"


class LensModel:
    """One camera's lens distortion -- the barrel ("fisheye") bend of a
    wide-angle USB lens that makes straight court lines bow outward -- from
    --calibrate-lens.

    OpenCV's radial model (k1, k2) about the image center with a nominal focal
    length, fitted to the court's own painted lines (straight in reality), so
    no checkerboard is needed. That recovers the *shape* of the bend but not
    the true focal length, which is all straightening needs: straightened
    pixels use the same camera matrix, so they keep the raw image's scale and
    center (and a point on the center stays put).

    Only valid for the capture mode it was fitted in -- USB cameras change
    their field of view between resolutions.
    """

    _UNDISTORT_CRITERIA = (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 50, 1e-7)

    def __init__(self, k1, k2, width, height, focal=None):
        self.width, self.height = int(width), int(height)
        self.focal = float(focal) if focal else 0.75 * max(self.width, self.height)
        self.K = np.array(
            [[self.focal, 0.0, self.width / 2.0], [0.0, self.focal, self.height / 2.0], [0.0, 0.0, 1.0]], dtype=np.float64
        )
        self.D = np.array([k1, k2, 0.0, 0.0, 0.0], dtype=np.float64)

    @classmethod
    def load(cls, path):
        data = json.loads(Path(path).read_text())
        return cls(data["k1"], data["k2"], data["width"], data["height"], data.get("focal"))

    def save(self, path, **extra):
        data = {"width": self.width, "height": self.height, "k1": float(self.D[0]), "k2": float(self.D[1]), "focal": self.focal}
        Path(path).write_text(json.dumps({**data, **extra}, indent=2))

    def undistort(self, points):
        """Raw camera pixels -> straightened pixels, (N, 2) -> (N, 2)."""
        pts = np.asarray(points, dtype=np.float64).reshape(-1, 1, 2)
        return cv2.undistortPointsIter(pts, self.K, self.D, None, self.K, self._UNDISTORT_CRITERIA).reshape(-1, 2)

    def distort(self, points):
        """Straightened pixels -> raw camera pixels (the inverse of undistort)."""
        pts = np.asarray(points, dtype=np.float64).reshape(-1, 2)
        normalized = (pts - self.K[:2, 2]) / self.focal
        rays = np.hstack([normalized, np.ones((len(pts), 1))])
        image, _ = cv2.projectPoints(rays, np.zeros(3), np.zeros(3), self.K, self.D)
        return image.reshape(-1, 2)


def fit_lens_from_lines(lines, width, height):
    """Fit a LensModel that makes every clicked line straight.

    lines: point lists, each clicked along one line that's straight in reality
    (>= 3 points each). Returns (lens, rms_before_px, rms_after_px): how far,
    on average, the clicked points sit off a straight line through them, raw
    vs straightened.

    Each line's straightness is measured relative to its own length, so the
    fit can't cheat by shrinking everything toward the center. A line through
    the image center is straight under any radial bend and carries no
    information -- lines near the edges, where the bend is strongest, matter
    most. k2 is only fitted with enough lines to support it.
    """
    from scipy.optimize import least_squares

    lines = [np.asarray(line, dtype=np.float64) for line in lines if len(line) >= 3]
    use_k2 = len(lines) >= 4 and sum(len(line) for line in lines) >= 16

    def straightness(lens):
        residuals = []
        for raw in lines:
            pts = lens.undistort(raw) if lens is not None else raw
            if not np.all(np.isfinite(pts)):
                return np.full(sum(len(line) for line in lines), 1e3)
            centered = pts - pts.mean(axis=0)
            _, singular, vt = np.linalg.svd(centered, full_matrices=False)
            raw_spread = np.linalg.svd(raw - raw.mean(axis=0), compute_uv=False)[0]
            residuals.append((centered @ vt[1]) * (raw_spread / max(singular[0], 1e-9)))
        return np.concatenate(residuals)

    def lens_for(params):
        return LensModel(params[0], params[1] if use_k2 else 0.0, width, height)

    result = least_squares(
        lambda params: straightness(lens_for(params)),
        x0=np.zeros(2 if use_k2 else 1),
        bounds=([-1.0] * (2 if use_k2 else 1), [1.0] * (2 if use_k2 else 1)),
    )
    lens = lens_for(result.x)
    rms_before = float(np.sqrt(np.mean(straightness(None) ** 2)))
    rms_after = float(np.sqrt(np.mean(straightness(lens) ** 2)))
    return lens, rms_before, rms_after


class CourtMapper:
    """Simple homography-based court calibration for a single camera.

    The default mapping assumes the full image is treated as a court rectangle in normalized coordinates.
    If a real court is visible, the user can pass four image corners using --court-corners to calibrate it.

    With a lens model (--lens), every image point -- the clicked corners and
    each ball position -- is straightened before the homography, since a
    homography can only map straight lines to straight lines: without it, a
    wide-angle lens's bend makes the mapping exact at the 4 clicked corners
    and off everywhere between. Callers always pass and get raw camera pixels
    either way; src_points stay as clicked.
    """

    def __init__(self, src_points=None, dst_points=None, court_width=20.0, court_length=44.0, lens=None, view="end"):
        if src_points is None:
            src_points = np.array([[0, 0], [1920, 0], [1920, 1080], [0, 1080]], dtype=np.float32)

        if dst_points is None and view == "side":
            # A camera beside the court (--view side): the TOP edge clicked is
            # the far SIDELINE and the court's length runs across the image.
            # Same court coordinates as from an end (x across, y along) -- the
            # corners are just clicked in a different order.
            dst_points = np.array(
                [[court_width, court_length], [court_width, 0], [0, 0], [0, court_length]], dtype=np.float32
            )
        elif dst_points is None:
            # court_length=44 covers the full court; pass 22 to scope calibration to
            # just one half (baseline to net) -- useful when only one half is
            # reliably visible/accurate from a given camera angle.
            dst_points = np.array(
                [[0, 0], [court_width, 0], [court_width, court_length], [0, court_length]], dtype=np.float32
            )

        self.view = view
        self.src_points = np.array(src_points, dtype=np.float32)
        self.dst_points = np.array(dst_points, dtype=np.float32)
        self.court_width = court_width
        self.court_length = court_length
        self.lens = lens
        straight_src = lens.undistort(self.src_points).astype(np.float32) if lens is not None else self.src_points
        self.h_matrix = cv2.getPerspectiveTransform(straight_src, self.dst_points)

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
        if self.lens is not None:
            src_point = self.lens.undistort(src_point).astype(np.float32)
        mapped = cv2.perspectiveTransform(src_point[None, :, :], self.h_matrix)[0][0]
        return float(mapped[0]), float(mapped[1])

    def image_points(self, court_points):
        """Court coordinates (ft) -> raw camera pixels, (N, 2) -> (N, 2): the
        inverse of map_point, for drawing the court over the camera image."""
        pts = np.asarray(court_points, dtype=np.float32).reshape(-1, 1, 2)
        straight = cv2.perspectiveTransform(pts, np.linalg.inv(self.h_matrix)).reshape(-1, 2)
        return self.lens.distort(straight) if self.lens is not None else straight

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
    # Line calls: a new one within CALL_REPEAT_FRAMES of the last is the same
    # bounce; the on-screen banner stays up for CALL_BANNER_SECONDS.
    CALL_REPEAT_FRAMES = 15
    CALL_BANNER_SECONDS = 2.0
    # Gap prediction (see _fit_motion): how many recent real detections to
    # fit, the minimum needed to fit at all, and the polynomial degree in y.
    # Chosen by simulated-gap benchmark (noisy ballistic shots, 5-15 hidden
    # frames): 10-point gravity fit had the lowest error, ~6-10x below the old
    # two-point decayed extrapolation.
    PREDICT_FIT_POINTS = 10
    PREDICT_MIN_POINTS = 3
    PREDICT_Y_DEGREE = 2
    # Bounce detection (see _detect_ball_contact): real detections needed on
    # each side of the ball's low point at 120fps (scaled to the camera's
    # frame rate by set_frame_rate, min 3), how many standard errors the drop
    # and rise slopes must clear, the smallest sudden slowdown that counts as
    # a kink, and how far outside the calibrated court (ft) a landing may map
    # before it's the ball high in the air. Tuned on a simulated 120fps
    # session (12 landings incl. six within 0.5ft of a line, 1.5px jitter, 8%
    # missed detections, 2px calibration click error; 10 runs): 113/120
    # landings called right with 1 false call from the side of the court,
    # 102/120 with none from behind a baseline. The old detector: 22/60 right
    # and 694 false calls over 5 runs from behind a baseline.
    BOUNCE_SIDE_POINTS = 8
    BOUNCE_MIN_T = 2.5
    KINK_MIN_SLOWDOWN = 2.5  # px/frame at 120fps, see _detect_ball_contact
    LANDING_MARGIN_FT = 10.0

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
        # Only a real --court-corners calibration is drawn over the feed and
        # gives meaningful calls; the default mapper is a placeholder.
        self.court_calibrated = court_mapper is not None
        self._court_overlay = None
        # The latest line call, for run_video's on-screen banner and terminal
        # log: {"call", "court" (x, y ft), "frame_index", "time", "new"}.
        self.last_call = None
        self.target_fps = target_fps
        self.target_width = target_width
        self.target_height = target_height
        self.camera_id = camera_id
        # Prepended to this tracker's console messages -- DualCameraTracker
        # sets it per camera, so each line says which camera it's about.
        self.log_prefix = ""
        # See _plausible_landing: with a half-court calibration, ignore
        # landings past the net. DualCameraTracker turns it off.
        self.half_court_only = True

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
        # (frame_index, x, y, box half-height) -- the input to _fit_motion and
        # _detect_ball_contact.
        self._observations: list[tuple[int, float, float, float]] = []
        # The ball's radius in the image (its latest box's half-height).
        self._ball_radius_px = 0.0
        # Frame of the last low point reported as a bounce, per track, so
        # each bounce is reported once (see _detect_ball_contact).
        self._last_bounce_frames: dict = {}
        self.last_landing_frame: int | None = None
        # Frame-rate dependent settings -- see set_frame_rate.
        self.frame_rate = 120
        self.bounce_side_points = self.BOUNCE_SIDE_POINTS
        self.kink_min_slowdown = self.KINK_MIN_SLOWDOWN
        self.person_refresh_frames = 1
        # Live sources: when the last processed frames arrived (see track_processing_rate).
        self._processed_at: deque = deque()
        self._rate_checked_at = 0.0
        self._rate_started_at: float | None = None
        self._person_boxes: list = []
        self._person_boxes_frame: int | None = None
        # Live sources run the person filter on a worker thread (see _detect_people).
        self.async_person_filter = False
        self._person_worker = None
        self._person_pending = None

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

    def _detect_ball_contact(self, points, frames, key="primary", radii=None):
        """Detects a bounce by looking for a V-shape in the Y-axis -- or a kink.

        Physics: A true court bounce is the ball moving downward (increasing Y)
        then suddenly moving upward (decreasing Y). The top of an arc is the
        opposite shape (up, then down) and never a landing. A ball coming
        toward the camera can keep moving down the image through its bounce,
        just abruptly slower: that kink counts too (see below). X-axis changes
        don't make a bounce -- they're spin, curvature, lateral movement --
        but the ball being sent back the way it came does unmake one: only a
        paddle does that.

        `points` are REAL detections (raw, unsmoothed -- smoothing and
        prediction would round off the V) with their frame indices, and
        `radii` their boxes' half-heights (to put the landing at the ball's
        bottom). The
        ball's lowest point in the image must sit exactly bounce_side_points
        detections before the newest, with the ball at least bounce_min_dy px
        higher at both ends of that window, and lines fitted to the drop and
        the rise must both slope clearly beyond the jitter they leave
        (BOUNCE_MIN_T times their standard error). Measured across several
        frames rather than frame to frame, detector jitter (a pixel or two
        per frame) can't fake a V -- not even on a ball held still before a
        serve, the commonest false bounce without that slope test -- yet a
        far-off ball moving a pixel or two per frame at 120fps still
        registers. Each low point is reported once.

        This replaces a frame-to-frame direction-change check plus a "speed
        drop" fallback. On a simulated 120fps session (1.5px jitter, 8%
        missed detections) those raised ~11 false calls per real landing: any
        direction change counted, including every top of an arc, and a ball
        whose speed merely dipped under 4px/frame -- routine for a far-off
        ball at 120fps -- counted as a bounce.

        Returns the landing point -- where lines fitted to the drop and the
        rise cross, a sub-frame estimate of the moment of contact that's
        steadier than any single detection -- or None.
        """
        side = self.bounce_side_points
        if len(points) < 2 * side + 1:
            return None
        pts = np.asarray(points[-(2 * side + 1):], dtype=float)
        fr = np.asarray(frames[-(2 * side + 1):], dtype=float)
        ys = pts[:, 1]
        # Detections spread over a long gap could span a hit as well as a bounce.
        if fr[-1] - fr[0] > 6 * side:
            return None
        # A bounce changes the ball's velocity, never its position: a jump
        # within the window is the lock hopping between objects (or a new
        # ball put in play), which the fits below would read as a huge kink.
        steps = np.hypot(np.diff(pts[:, 0]), np.diff(pts[:, 1])) / np.maximum(np.diff(fr), 1)
        if steps.max() > max(4 * float(np.median(steps)), 10.0):
            return None
        t = fr - fr[side]
        try:
            fall, rise, standard_error = self._split_fit(t, ys, side, noise_floor=0.5)
            x_in, x_out, x_error = self._split_fit(t, pts[:, 0], side, noise_floor=0.5)
        except (np.linalg.LinAlgError, ValueError):
            return None
        # The ground can't turn the ball back the way it came; a paddle can.
        # Seen from the side of the court, where play runs across the image,
        # this is what tells most hits apart from bounces. Slopes, not single
        # points: on a ball dropping straight down, two jittery points alone
        # "reversed" often enough to throw away real bounces. And it must be a
        # fast turn-back: perspective alone swings an off-center ball's image
        # sideways and back at a straight-down bounce (its distance to the
        # camera turns back), but only by a fraction of its vertical speed --
        # a paddle sends it back the way it came at full speed.
        turn_back = min(abs(x_in[0]), abs(x_out[0]))
        if x_in[0] * x_out[0] < 0 and turn_back >= max(self.BOUNCE_MIN_T * x_error, 0.5 * min(abs(fall[0]), abs(rise[0]))):
            return None
        min_slope = self.BOUNCE_MIN_T * standard_error
        v_shape = int(np.argmax(ys)) == side and min(ys[side] - ys[0], ys[side] - ys[-1]) >= self.bounce_min_dy
        if v_shape:
            if fall[0] < min_slope or -rise[0] < min_slope:
                return None
        else:
            # A kink instead of a V: coming toward the camera, a ball keeps
            # moving down the image straight through its bounce, just
            # abruptly slower -- the ground's upward kick against its
            # approach. Gravity and perspective only ever bend the path the
            # other way, so a big enough, sudden enough slowdown centered here
            # is the bounce. (The V test alone missed every such bounce in
            # simulation: from behind a baseline, most of the shots coming
            # back.) It must be coming DOWN the image into it: counting upward
            # kicks of a ball already moving up too caught flat drives racing
            # away, but also 10-20x more paddle hits as false bounces.
            if fall[0] < min_slope:
                return None
            slowdown = fall[0] - rise[0]
            if slowdown < max(self.BOUNCE_MIN_T * np.sqrt(2) * standard_error, self.kink_min_slowdown):
                return None
            # The kink must be at (within a frame of) this window's middle --
            # not tighter, or with jitter the estimate can step right over a
            # half-frame window as it slides. Repeats are dropped below.
            if abs((rise[1] - fall[1]) / slowdown) > 1.0:
                return None
        low_frame = int(fr[side])
        # One bounce can pass both tests a frame apart; two real ones can't
        # be that close together.
        if low_frame <= self._last_bounce_frames.get(key, -1) + side:
            return None
        self._last_bounce_frames[key] = low_frame
        self.last_landing_frame = low_frame
        landing_x, landing_y = self._fit_landing(pts, t, fall, rise)
        if radii is not None:
            # The ball touches the ground at its bottom, not its center: a
            # ball's radius (~1.5in) further down the image. Mapping the center
            # put landings 3-6in too far from the camera, always the same way
            # -- the largest single error in simulation, and a biased one.
            landing_y += float(np.median(radii[-(2 * side + 1):]))
        return landing_x, landing_y

    @staticmethod
    def _split_fit(t, values, side, noise_floor):
        """Lines fitted to `values` up to and from the middle point (index
        `side`), and the standard error of their slopes -- from the jitter
        the lines leave unexplained, never taken as less than noise_floor."""
        before = np.polyfit(t[: side + 1], values[: side + 1], 1)
        after = np.polyfit(t[side:], values[side:], 1)
        residuals = np.concatenate([values[: side + 1] - np.polyval(before, t[: side + 1]), values[side:] - np.polyval(after, t[side:])])
        jitter = max(float(np.sqrt(np.sum(residuals**2) / max(len(residuals) - 4, 1))), noise_floor)
        spread = min(float(np.std(t[: side + 1])), float(np.std(t[side:]))) * np.sqrt(side + 1)
        return before, after, (jitter / spread if spread > 0 else np.inf)

    @staticmethod
    def _fit_landing(pts, t, fall, rise):
        """Where the lines fitted to the drop and the rise cross: the moment
        of contact, between frames. `t` is each point's frame offset from the
        low point."""
        side = int(np.argmin(np.abs(t)))
        low = (float(pts[side, 0]), float(pts[side, 1]))
        if fall[0] - rise[0] <= 1e-6:
            return low
        contact = (rise[1] - fall[1]) / (fall[0] - rise[0])
        if not t[0] <= contact <= t[-1]:
            return low
        along_x = np.polyfit(t, pts[:, 0], 1)
        return float(np.polyval(along_x, contact)), float(np.polyval(fall, contact))

    def _plausible_landing(self, point):
        """False for a "landing" mapping more than LANDING_MARGIN_FT outside
        the calibrated court: a V the ball made high in the air (e.g. a lob
        dropping onto a paddle), whose line of sight meets the ground far
        away. Always True uncalibrated, where there's no court to compare.

        With a half-court calibration (an end camera, net = the TOP edge) a
        single camera also doesn't call anything past the net: that half isn't
        its to call, and a ball in the air above it maps farther still. On real
        footage (ELP behind a baseline, near half calibrated) most wrong OUT
        calls were exactly this -- "-8 to -134 ft from the net". Dual-camera
        mode clears half_court_only: the other end camera covers that half.
        """
        if not self.court_calibrated:
            return True
        x, y = self.court_mapper.map_point(self.frame_coords(point))
        dst = self.court_mapper.dst_points
        margin = self.LANDING_MARGIN_FT
        y_min = float(np.min(dst[:, 1]))
        half_court = self.court_mapper.view == "end" and abs(self.court_mapper.court_length - 44.0) >= 1.0
        if half_court and self.half_court_only and y < y_min:
            return False
        return (
            float(np.min(dst[:, 0])) - margin <= x <= float(np.max(dst[:, 0])) + margin
            and y_min - margin <= y <= float(np.max(dst[:, 1])) + margin
        )

    def _court_position(self, point):
        """A point in the frame process_frame works on -> court (x, y) in ft."""
        return self.court_mapper.map_point(self.frame_coords(point))

    def _draw_court_and_call(self, frame):
        """run_video's single-camera overlay: the calibrated court's lines (bent
        with the lens, if there is one) and the latest line call as a banner
        for CALL_BANNER_SECONDS. Drawn after detection, never into its input."""
        if self.court_calibrated:
            if self._court_overlay is None:
                offset = np.array(self.zoom_roi[:2] if self.zoom_roi is not None else (0, 0), dtype=np.float64)
                self._court_overlay = [
                    (tuple(np.round(np.array(a) - offset).astype(int)), tuple(np.round(np.array(b) - offset).astype(int)))
                    for a, b in _court_reference_lines(self.court_mapper)
                ]
            for a, b in self._court_overlay:
                cv2.line(frame, a, b, (255, 255, 0), 1)

        call = self.last_call
        if call is None or time.perf_counter() - call["time"] > self.CALL_BANNER_SECONDS:
            return
        color = (0, 200, 0) if call["call"] == "IN" else (0, 0, 255)
        x_ft, y_ft = call["court"]
        if self.court_mapper.view == "side":
            # --view side's court coordinates: x=0 is the near sideline, y=0
            # the baseline at the right of the image.
            where = f"{x_ft:.1f} ft from the near sideline, {y_ft:.1f} ft from the right-hand baseline"
        else:
            from_edge = "the net" if abs(self.court_mapper.court_length - 44.0) >= 1.0 else "the far baseline"
            where = f"{x_ft:.1f} ft from the left sideline, {y_ft:.1f} ft from {from_edge}"
        _draw_label(frame, call["call"], (frame.shape[1] // 2 - 40, 60), color, scale=1.8, thickness=4)
        _draw_label(
            frame,
            where,
            (frame.shape[1] // 2 - 200, 95),
            color,
            scale=0.55,
            thickness=1,
        )

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
        """The locked ball's position this frame, in original-frame pixels --
        its bottom, the point that touches the ground when it bounces -- and
        whether it's a real detection (False = bridged by prediction through
        a detection gap). None when no ball is locked."""
        if not self.primary_trajectory:
            return None
        x, y = self.primary_trajectory[-1]
        return self.frame_coords((x, y + self._ball_radius_px)), self.missed_frames == 0

    def set_frame_rate(self, fps):
        """Scale the frame-count settings to the source's frame rate: the
        bounce detector's window (BOUNCE_SIDE_POINTS is for 120fps; a 30fps
        camera would otherwise need 8 frames = 267 ms after every bounce), and
        how often the person filter re-runs (~30 times a second)."""
        fps = fps or 30
        self.frame_rate = fps
        self.bounce_side_points = max(3, round(self.BOUNCE_SIDE_POINTS * fps / 120))
        self.kink_min_slowdown = self.KINK_MIN_SLOWDOWN * 120 / fps  # px per frame grows as frames get further apart
        self.person_refresh_frames = max(1, round(fps / 30))

    def track_processing_rate(self, now, camera_fps):
        """Live sources: time the frame-count settings (set_frame_rate) to
        the frames actually processed, not the camera's rate. A live camera's
        newest frame is taken each step and the rest dropped, so when
        processing can't keep up -- two 120fps cameras on one laptop GPU run
        ~60 pairs/s -- the detector sees every 2nd frame, and its 8-frame
        bounce window at "120fps" really spans twice as long. In simulation
        at 60 processed fps: 89/120 landings right timed for 120, 98 timed
        for 60; at 40: 70 vs 93. Re-checked twice a second from the last
        second's frames; re-timed on a change of more than 10%. The first
        1.5 s are skipped: startup (window, first GPU calls) runs slow.
        """
        times = self._processed_at
        if self._rate_started_at is None:
            self._rate_started_at = now
        times.append(now)
        while now - times[0] > 1.0:
            times.popleft()
        if now - self._rate_started_at < 1.5 or now - self._rate_checked_at < 0.5 or len(times) < 10:
            return
        self._rate_checked_at = now
        rate = min(float(camera_fps), (len(times) - 1) / (times[-1] - times[0]))
        if abs(rate - self.frame_rate) > 0.1 * self.frame_rate:
            if self.frame_rate == camera_fps or abs(rate - self.frame_rate) > 0.25 * self.frame_rate:
                print(f"{self.log_prefix}[Rate] Processing {rate:.0f} of the camera's {camera_fps} frames/s -- bounce detection timed for {rate:.0f}")
            self.set_frame_rate(round(rate))

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

        Boxes are reused for person_refresh_frames frames (~1/30 s, see
        set_frame_rate): people barely move in that time, and running the
        detector on every frame (~9 ms on an RTX 4050 Laptop) is what held
        processing below a 120fps camera's frame rate.

        With async_person_filter (live sources) the refresh runs on a worker
        thread and this returns the latest finished boxes without waiting --
        one refresh (~1/30 s) older, which people don't notice either. In
        dual-camera mode it was ~5 ms of every pair's critical path. Video
        files keep it synchronous, so re-running one gives the same result.
        """
        if self.person_model is None:
            return []
        pending = self._person_pending
        if pending is not None and pending.done():
            self._person_boxes, self._person_pending = pending.result(), None
        if self._person_boxes_frame is not None and self.frame_index - self._person_boxes_frame < self.person_refresh_frames:
            return self._person_boxes
        self._person_boxes_frame = self.frame_index
        if not self.async_person_filter:
            self._person_boxes = self._find_people(frame)
        elif self._person_pending is None:
            if self._person_worker is None:
                from concurrent.futures import ThreadPoolExecutor

                self._person_worker = ThreadPoolExecutor(max_workers=1)
            # A copy: the caller goes on to draw on this frame.
            self._person_pending = self._person_worker.submit(self._find_people, frame.copy())
        return self._person_boxes

    def _find_people(self, frame):
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

    def _snap_to_ball(self, frame, box):
        """The whole ball's box, when `box` is only part of it.

        Up close a pickleball is big enough that the model finds its holes --
        each a small round thing with a ball-colored rim, so shape and color
        both pass -- and a box on a hole puts the ball's center and bottom
        (its landing point) in the wrong place. Around the box, the
        ball-colored region's outer outline is the whole ball (the holes are
        inside it). If that outline contains the box's center and is a round
        blob clearly bigger than the box, its bounding box is used instead.
        A box already on the whole ball (the usual case on court, the ball a
        few pixels across) finds nothing bigger and is kept.
        """
        if not self.require_ball_color:
            return box
        x1, y1, x2, y2 = box
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        size = max(x2 - x1, y2 - y1)
        # A pickleball is ~8 hole-widths across, and from a hole at its rim
        # the far side is that far away: look 9 box-widths out.
        reach = min(max(9 * size, 40), 400)
        wx1, wy1 = max(int(cx - reach), 0), max(int(cy - reach), 0)
        wx2, wy2 = min(int(cx + reach), frame.shape[1]), min(int(cy + reach), frame.shape[0])
        if wx2 - wx1 < 4 or wy2 - wy1 < 4:
            return box
        # Blurred, and specks removed: with sensor noise, stray background
        # pixels in the ball's color otherwise link it to its surroundings.
        hsv = cv2.cvtColor(cv2.GaussianBlur(frame[wy1:wy2, wx1:wx2], (5, 5), 0), cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.ball_color_lower, self.ball_color_upper)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        # Bridge holes that break the ball's outline at its edge.
        kernel = max(3, int(size / 2) | 1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel, kernel)))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return box
        # The blob the box's center is in -- or, for a hole at the ball's
        # rim, whose thin edge can break into a notch, just outside of: within
        # one box-width.
        center = (float(cx - wx1), float(cy - wy1))
        depth, contour = max(((cv2.pointPolygonTest(c, center, True), c) for c in contours), key=lambda pair: pair[0])
        if depth < -size:
            return box
        bx, by, bw, bh = cv2.boundingRect(contour)
        touches_edge = bx == 0 or by == 0 or bx + bw >= wx2 - wx1 or by + bh >= wy2 - wy1
        round_blob = min(bw, bh) / max(bw, bh) >= 0.7 and cv2.contourArea(contour) >= 0.55 * bw * bh
        if not touches_edge and round_blob and bw * bh >= 2.0 * (x2 - x1) * (y2 - y1):
            return (float(wx1 + bx), float(wy1 + by), float(wx1 + bx + bw), float(wy1 + by + bh))
        return box

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
        t, xs, ys = (np.array(v, dtype=float) for v in list(zip(*obs))[:3])

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
            self._observations.append((self.frame_index, raw_x, raw_y, (y2 - y1) / 2.0))
            self._ball_radius_px = (y2 - y1) / 2.0
            del self._observations[: -max(self.PREDICT_FIT_POINTS, 2 * self.bounce_side_points + 1)]

        self.primary_trajectory.append((center_x, center_y))
        if len(self.primary_trajectory) > self.max_history:
            self.primary_trajectory.pop(0)

        velocity = self._estimate_velocity(self.primary_trajectory)
        landing_point = None
        if not predicted:
            landing_point = self._detect_ball_contact(
                [(o[1], o[2]) for o in self._observations],
                [o[0] for o in self._observations],
                radii=[o[3] for o in self._observations],
            )
            if landing_point is not None and not self._plausible_landing(landing_point):
                landing_point = None
        if landing_point is not None:
            call = self._classify_in_out(landing_point)
            self.events.append(
                BallEvent(
                    frame_index=self.frame_index,
                    track_id=self.display_track_id,
                    centroid=(center_x, center_y),
                    velocity=velocity,
                    landing_point=landing_point,
                    landing_frame=self.last_landing_frame,
                    line_call=call,
                    camera_id=self.camera_id,
                )
            )
            # One bounce trips _detect_ball_contact on a few consecutive frames;
            # announce it once.
            if self.last_call is None or self.frame_index - self.last_call["frame_index"] > self.CALL_REPEAT_FRAMES:
                self.last_call = {
                    "call": call,
                    "court": self._court_position(landing_point),
                    "frame_index": self.frame_index,
                    "time": time.perf_counter(),
                    "new": True,
                }

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
            # One centroid per frame this track was seen in.
            frames = list(range(self.frame_index - len(track) + 1, self.frame_index + 1))
            landing_point = self._detect_ball_contact(track, frames, key=track_id)
            if landing_point is not None and not self._plausible_landing(landing_point):
                landing_point = None
            if landing_point is not None:
                call = self._classify_in_out(landing_point)
                self.events.append(
                    BallEvent(
                        frame_index=self.frame_index,
                        track_id=track_id,
                        centroid=(center_x, center_y),
                        velocity=velocity,
                        landing_point=landing_point,
                        landing_frame=self.last_landing_frame,
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
        self._draw_primary_tracking(detect_frame, self._snap_to_ball(detect_frame, box), predicted=False)
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
            self._draw_primary_tracking(detect_frame, self._snap_to_ball(detect_frame, box), predicted=False)
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
        times = None if isinstance(video_source, int) else _load_timestamps(video_source)
        delivered = _delivered_fps(times) if times is not None else None
        if delivered and abs(delivered - fps) > 0.1 * fps:
            print(f"[Camera Actual] ...but it really delivered {delivered}fps when recorded (from its capture times)")
            fps = delivered
        self.set_frame_rate(fps)

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
        # (Encoding now also runs on its own thread, _BackgroundVideoWriter;
        # the local Roboflow model is fast enough for 30.)
        default_record_fps = 10 if self.device == "roboflow-server" else 30
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
            writer = _BackgroundVideoWriter(output_path, effective_record_fps, (output_width, output_height))

        # Separate from `writer`: saves the frame BEFORE any boxes/labels/trajectory
        # lines are drawn on it, so this file is safe to upload to Roboflow for
        # annotation/training. `writer` above bakes the overlay in permanently and
        # is only meant for reviewing/demoing tracking results, not as training data.
        raw_writer = None
        if raw_output_path:
            raw_output_path = Path(raw_output_path)
            raw_output_path.parent.mkdir(parents=True, exist_ok=True)
            raw_writer = _BackgroundVideoWriter(raw_output_path, effective_record_fps, (width, height))
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
        record_start = time.perf_counter()
        frames_written = 0
        max_catchup_frames_per_iteration = max(1, int(effective_record_fps))


        stats = _SessionStats(stats_csv) if stats_csv else None
        reader = _FrameReader(cap, live=isinstance(video_source, int))
        self.async_person_filter = reader.live
        try:
            while True:
                frame, arrived_at, _ = reader.read()
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
                if reader.live:
                    self.track_processing_rate(arrived_at, fps)
                if stats is not None:
                    stats.frame_done(arrived_at)
                self._draw_court_and_call(annotated)
                if self.last_call is not None and self.last_call["new"]:
                    self.last_call["new"] = False
                    x_ft, y_ft = self.last_call["court"]
                    print(f"[Call] {self.last_call['call']} -- court position x={x_ft:.1f} ft, y={y_ft:.1f} ft (frame {self.last_call['frame_index']})")

                if writer is not None or raw_writer is not None:
                    due = min(
                        int((time.perf_counter() - record_start) * effective_record_fps) + 1 - frames_written,
                        max_catchup_frames_per_iteration,
                    )
                    if due > 0:
                        # Queued without copying: both are new arrays every frame,
                        # and nothing draws on them after this.
                        if writer is not None:
                            writer.write(annotated, copies=due)
                        if raw_writer is not None:
                            raw_writer.write(raw_frame, copies=due)
                        frames_written += due

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
    # Roughly how far behind its baseline each end camera stands (ft) --
    # only used to weight the two cameras' readings (see _ground_point), so
    # being a few feet off barely matters.
    CAMERA_SETBACK_FT = 15.0

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
            point = self._ground_point(low_a, low_b)
            camera = self.home_camera(point)
            self._add({
                "camera": camera, "source": "ground", "point": point,
                "status": "CONFIRMED", "gap": low, "t": t_low, "frame_index": frame_low,
            })
            self._airborne = False

    def _ground_point(self, point_a, point_b):
        """The landing spot from both cameras' readings at the ground check's
        low point. That sample is the frame closest to contact, but rarely
        at it: the ball is still up to a frame's fall above the ground, and
        each camera maps a raised ball away from itself along its line of
        sight -- ~7 ft per ft of height -- so the two readings straddle the
        true spot. It sits closer to the nearer camera's reading, in
        proportion to the two cameras' distances. In simulation (a dink
        landing 3 ft past the net, cameras at 120fps + 60fps) the home
        camera's reading alone was 1.1-1.4 ft off; this cancels most of it.
        """
        y = (point_a[1] + point_b[1]) / 2
        distance_a = max(1.0, y + self.CAMERA_SETBACK_FT)
        distance_b = max(1.0, 2 * self.half_length - y + self.CAMERA_SETBACK_FT)
        weight_a = distance_b / (distance_a + distance_b)
        return (
            weight_a * point_a[0] + (1 - weight_a) * point_b[0],
            weight_a * point_a[1] + (1 - weight_a) * point_b[1],
        )

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


class CaptureTimeAligner:
    """Feeds DualCameraFusion in capture-time order, pairing each camera's
    ball position with where the OTHER camera saw the ball at that same
    instant -- interpolated between that camera's frames just before and
    after it.

    The fusion's ground check compares two cameras' court positions for the
    ball, and a ball in flight moves ~0.5 ft per 120fps frame: positions
    even a few ms apart differ by more than the calibrations do. Frames
    processed together aren't captured together -- the cameras aren't
    synchronized, one can run slower (dim light) or be skipped a step, and
    each driver hands its frames over 80-270 ms after capture, varying frame
    to frame (a laptop webcam, measured) -- so pairing by processing step
    compares the ball at two different moments. Each sample waits here
    until the other camera has a frame after it (a frame period or so), then
    goes to the fusion.

    A bounce flag is paired the same way, at its landing frame's capture
    time: it's raised a few frames after the landing (the detector needs the
    rise too), when the other camera's current view shows the ball already
    back up in the air.
    """

    # Interpolate only across a gap this short in the other camera's frames
    # (~8 frames at 120fps): longer, and the ball's path between isn't a line.
    MAX_BRACKET_S = 0.07
    # A camera whose newest frame is this far behind the other's is stalled
    # or unplugged: samples stop waiting for it.
    STALE_S = 0.3
    HISTORY_S = 2.0

    def __init__(self, fusion):
        self.fusion = fusion
        # Per camera, every processed frame as (capture time, court point of
        # a real detection, or None), in time order.
        self.history: dict[str, list] = {"A": [], "B": []}
        # Each camera's frame_index -> capture time, for its bounce flags' landing frames.
        self.frame_times: dict[str, dict[int, float]] = {"A": {}, "B": {}}
        self._pending: list = []
        self._order = 0
        self.emitted_until = float("-inf")

    def add_frame(self, end, frame_index, t, court_point):
        """One processed frame of camera `end`, captured at t: its court
        point for the ball (a real detection), or None."""
        history = self.history[end]
        if history and t <= history[-1][0]:
            return  # a repeated or out-of-order capture time: nothing new
        history.append((t, court_point))
        while history and history[0][0] < t - self.HISTORY_S:
            history.pop(0)
        times = self.frame_times[end]
        times[frame_index] = t
        for old in [index for index in times if index < frame_index - 1000]:
            del times[old]
        self._push(t, 0, end, court_point, frame_index)

    def add_flag(self, end, landing_frame, court_point, fallback_t):
        """A bounce flag from camera `end`'s own detector, landing in its
        frame landing_frame at court_point (global ft)."""
        t = self.frame_times[end].get(landing_frame, fallback_t)
        self._push(t, 1, end, court_point, landing_frame)

    def _push(self, t, kind, end, court_point, frame_index):
        import heapq

        # Samples before flags at the same instant: the fusion expects the
        # ball's latest state before a flag (see add_ground_sample).
        heapq.heappush(self._pending, (t, kind, self._order, end, court_point, frame_index))
        self._order += 1

    def _other_at(self, end, t, force):
        """(ready, court point): the other camera's ball at time t, or None
        if it had no real detection around then. ready is False while that
        camera has no frame at or after t yet."""
        history = self.history["B" if end == "A" else "A"]
        newest = max((h[-1][0] for h in self.history.values() if h), default=t)
        if not history or history[-1][0] < t:
            stalled = not history or history[-1][0] < newest - self.STALE_S
            return (force or stalled), None
        if t < history[0][0]:
            return True, None
        after = next(i for i, (frame_t, _) in enumerate(history) if frame_t >= t)
        t1, p1 = history[after]
        if t1 - t < 1e-4:
            return True, p1
        t0, p0 = history[after - 1]
        if p0 is None or p1 is None or t1 - t0 > self.MAX_BRACKET_S:
            return True, None
        w = (t - t0) / (t1 - t0)
        return True, (p0[0] + w * (p1[0] - p0[0]), p0[1] + w * (p1[1] - p0[1]))

    def flush(self, force=False):
        """Hand the fusion everything that can be paired now (force: all of
        it, at the end of a session). Returns the time up to which the
        fusion has been fed."""
        import heapq

        while self._pending:
            t, kind, _, end, court_point, frame_index = self._pending[0]
            ready, other = self._other_at(end, t, force)
            if not ready:
                break
            heapq.heappop(self._pending)
            self.emitted_until = max(self.emitted_until, t)
            if kind == 0:
                if court_point is None:
                    self.fusion.add_ground_sample(t, frame_index, None, None)
                else:
                    a, b = (court_point, other) if end == "A" else (other, court_point)
                    self.fusion.add_ground_sample(t, frame_index, a, b)
            else:
                self.fusion.add_flag(end, court_point, other, t, frame_index)
        return self.emitted_until


class DualCameraTracker:
    """Tracks with both end cameras at once -- CLAUDE.md's dual end-camera
    build. Each camera gets its own PickleVisionTracker (its own ball lock,
    prediction and person filter), calibrated for its own near half
    (--court-length 22). Their balls and bounce flags are mapped onto one
    full-court coordinate system (to_global_court_point) and the flags are
    fused into line calls (DualCameraFusion).

    Frames are taken in time order (_SyncedFrames): live, each camera's
    newest frame -- their clocks aren't synced, so the two can be up to one
    frame period apart (~8ms at 120fps); from video files, by the capture
    times --record-raw saved beside them. Each step runs the two trackers on
    two threads, so one camera's CPU work (filtering, drawing) overlaps the
    other's GPU work. The detection model is shared, and the GPU is the
    limit: two 120fps cameras on an RTX 4050 Laptop (720p, person filter on)
    run ~90 pairs/s live -- so the detector sees ~3 of every 4 frames and is
    timed for that (track_processing_rate).
    """

    COURT_VIEW_HEIGHT = 300  # at a 360-pixel-high feed; scaled with the feeds (see _compose)
    # The combined view is only built this often unless it's being recorded:
    # ~3 ms a pair, and the window doesn't need more.
    DISPLAY_FPS = 30
    CAMERA_COLORS = {"A": (255, 170, 0), "B": (0, 150, 255)}  # BGR: blue, orange
    RECENT_CALLS_SHOWN = 12

    def __init__(self, tracker_a, tracker_b, court_width=20.0, half_length=22.0, agreement_ft=3.0):
        self.trackers = {"A": tracker_a, "B": tracker_b}
        for end, tracker in self.trackers.items():
            tracker.log_prefix = f"[Cam {end}] "
            tracker.half_court_only = False
        self.court_width = court_width
        self.half_length = half_length
        # Court positions need both cameras calibrated -- an uncalibrated
        # tracker's "court" is just its whole frame.
        self.calibrated = tracker_a.court_calibrated and tracker_b.court_calibrated
        self.fusion = DualCameraFusion(court_width, half_length, agreement_ft)
        self.aligner = CaptureTimeAligner(self.fusion)
        self.pair_index = 0
        self._events_seen = {"A": 0, "B": 0}

    def _court_point(self, end, frame_point):
        """A point in this camera's original-frame pixels -> global court ft."""
        local = self.trackers[end].court_mapper.map_point(frame_point)
        return to_global_court_point(local, end, self.half_length, self.court_width)

    def _update(self, captured):
        """After a step: hand the fusion (through the aligner) each processed
        camera's ball and any new bounce flags. captured maps each camera
        processed this step to its frame's capture time. Returns each
        camera's ball as (court point, is a real detection), or None, for
        the court view."""
        balls = {}
        for end, tracker in self.trackers.items():
            state = tracker.ball_position()
            balls[end] = None if state is None or not self.calibrated else (self._court_point(end, state[0]), state[1])
        for end, t in captured.items():
            tracker = self.trackers[end]
            new_events = tracker.events[self._events_seen[end]:]
            self._events_seen[end] = len(tracker.events)
            if not self.calibrated:
                continue
            # Only real detections count as a camera seeing the ball -- a
            # prediction bridging a gap is a guess, not evidence.
            ball = balls[end]
            self.aligner.add_frame(end, tracker.frame_index, t, ball[0] if ball is not None and ball[1] else None)
            for event in new_events:
                court_point = self._court_point(end, tracker.frame_coords(event.landing_point))
                landing_frame = event.landing_frame if event.landing_frame is not None else event.frame_index
                self.aligner.add_flag(end, landing_frame, court_point, t)
        if self.calibrated:
            for call in self.fusion.close_ready(self.aligner.flush()):
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
        """Both annotated feeds side by side, over a top-down court diagram.
        The feeds keep the cameras' own resolution (the smaller one's height,
        if they differ) -- shrinking them to fit a laptop screen here made
        the window blurry once maximized; the window scales it instead."""
        panel_height = min(frame.shape[0] for frame in annotated.values())
        ui = panel_height / 360  # text and marks were sized for 360-pixel-high panels
        panels = []
        for end in ("A", "B"):
            frame = annotated[end]
            height, width = frame.shape[:2]
            panels.append(frame if height == panel_height else cv2.resize(frame, (max(1, round(width * panel_height / height)), panel_height)))
        top = np.hstack(panels)  # a copy: labels drawn on it don't touch the trackers' frames
        for end, x in (("A", 0), ("B", panels[0].shape[1])):
            state = self.trackers[end].ball_position()
            label = f"Camera {end}: " + ("no ball" if state is None else ("ball" if state[1] else "ball (predicted)"))
            _draw_label(top, label, (x + round(10 * ui), round(26 * ui)), self.CAMERA_COLORS[end], scale=0.6 * ui, thickness=max(2, round(2 * ui)))
        return np.vstack([top, self._draw_court_view(top.shape[1], balls, ui)])

    def _draw_court_view(self, width, balls, ui=1.0):
        """Top-down diagram of the full court -- end A on the left, end B on
        the right -- with each camera's ball (filled = detected, ring =
        predicted) and the recent line calls. ui scales text and marks."""
        height = round(self.COURT_VIEW_HEIGHT * ui)
        view = np.full((height, width, 3), 32, np.uint8)
        text_area = round(48 * ui)
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

        def size(value):
            return max(1, round(value * ui))

        font = cv2.FONT_HERSHEY_SIMPLEX
        w, net, kitchen = self.court_width, self.half_length, 7.0
        white = (235, 235, 235)
        cv2.rectangle(view, px((0, 0)), px((w, length)), (100, 65, 30), -1)
        for a, b in (
            ((0, 0), (w, 0)), ((0, length), (w, length)),  # baselines
            ((0, 0), (0, length)), ((w, 0), (w, length)),  # sidelines
            ((0, net - kitchen), (w, net - kitchen)), ((0, net + kitchen), (w, net + kitchen)),  # kitchen lines
            ((w / 2, 0), (w / 2, net - kitchen)), ((w / 2, net + kitchen), (w / 2, length)),  # centerlines
        ):
            cv2.line(view, px(a), px(b), white, size(1), cv2.LINE_AA)
        cv2.line(view, px((-1, net)), px((w + 1, net)), (190, 190, 190), size(3))  # net, posts just outside the sidelines
        # Out at the edge of the run-off, clear of calls just past the baselines.
        for end, y in (("A", 1.0 - run_off_y), ("B", length + run_off_y - 1.0)):
            label_x, label_y = px((w / 2, y))
            cv2.putText(view, end, (label_x - size(6), label_y + size(6)), font, 0.7 * ui, self.CAMERA_COLORS[end], size(2), cv2.LINE_AA)

        for call in self.fusion.calls[-self.RECENT_CALLS_SHOWN:]:
            p = px(call.court_point)
            if on_view(p):
                color = (0, 200, 0) if call.call == "IN" else (0, 0, 255)
                cv2.circle(view, p, size(6), color, -1 if call.status == "CONFIRMED" else size(2), cv2.LINE_AA)
        for end, ball in balls.items():
            if ball is not None and on_view(px(ball[0])):
                cv2.circle(view, px(ball[0]), size(4), self.CAMERA_COLORS[end], -1 if ball[1] else size(1), cv2.LINE_AA)

        if not self.calibrated:
            status = "Not calibrated: --calibrate --court-length 22 each camera, then pass --court-corners / --court-corners2"
        elif self.fusion.calls:
            last = self.fusion.calls[-1]
            how = "both cameras agree" if last.status == "CONFIRMED" else f"camera {last.source_camera} only"
            status = f"Last call: {last.call} at ({last.court_point[0]:.1f}, {last.court_point[1]:.1f}) ft -- {how}"
            p = px(last.court_point)
            if on_view(p):
                cv2.putText(view, last.call, (p[0] + size(8), p[1] - size(8)), font, 0.5 * ui, white, size(1), cv2.LINE_AA)
        else:
            status = "Waiting for the first bounce"
        cv2.putText(view, status, (size(10), height - size(28)), font, 0.5 * ui, white, size(1), cv2.LINE_AA)
        cv2.putText(
            view,
            "calls: filled = both cameras agree, ring = one camera only | small dots = each camera's ball (ring = predicted)",
            (size(10), height - size(9)),
            font,
            0.4 * ui,
            (170, 170, 170),
            size(1),
            cv2.LINE_AA,
        )
        return view

    def run(self, source_a, source_b, output_path=None, raw_output_a=None, raw_output_b=None, show_window=True,
            record_fps=None, flips=None, stats_csv=None):
        """Track both cameras until either source ends (or fails) or 'q' is
        pressed in the window. Returns the line calls.

        Frames are taken in time order (_SyncedFrames): each step processes
        the cameras with a new frame -- both, normally, on two threads.
        output_path records the combined view (both feeds + court diagram).
        raw_output_a/raw_output_b record each camera's unannotated frames,
        written in step (for lag-free footage to replay, use --record-raw).
        flips maps "A"/"B" to (horizontal, vertical).
        """
        from concurrent.futures import ThreadPoolExecutor

        if str(source_a) == str(source_b):
            raise ValueError(f"--source and --source2 are the same ({source_a}) -- dual-camera mode needs two different cameras or videos")
        flips = flips or {}
        caps, live, writers, camera_fps, timestamps = {}, {}, {}, {}, {}
        frames_in = display = stats = None
        executor = ThreadPoolExecutor(max_workers=1)
        t = 0.0
        try:
            for end, source in (("A", source_a), ("B", source_b)):
                tracker = self.trackers[end]
                caps[end], live[end] = _open_video_source(source, tracker.target_width, tracker.target_height, tracker.target_fps)
                cap = caps[end]
                camera_fps[end] = int(cap.get(cv2.CAP_PROP_FPS)) or 30
                timestamps[end] = None if live[end] else _load_timestamps(source)
                delivered = _delivered_fps(timestamps[end]) if timestamps[end] is not None else None
                note = ""
                if delivered and abs(delivered - camera_fps[end]) > 0.1 * camera_fps[end]:
                    note = f" (really delivered {delivered}fps, from its capture times)"
                    camera_fps[end] = delivered
                elif timestamps[end] is not None:
                    note = " (synced by its capture times)"
                tracker.set_frame_rate(camera_fps[end])
                print(
                    f"[Camera {end}] {source}: {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x"
                    f"{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))} @ {int(cap.get(cv2.CAP_PROP_FPS)) or 30}fps{note}"
                )
            if not any(live.values()) and (timestamps["A"] is None) != (timestamps["B"] is None):
                print("[Dual] Only one of the two videos has capture times -- syncing both by frame number / fps instead")
                timestamps = {"A": None, "B": None}
            fps = min(camera_fps.values())

            # Same recording rules as run_video: a lower write rate than the
            # capture rate, frame-paced to wall-clock time so recordings play
            # at real speed, encoded on a background thread.
            default_record_fps = 10 if self.trackers["A"].device == "roboflow-server" else 30
            record_fps = record_fps if record_fps else min(fps, default_record_fps)
            outputs = {key: Path(path) for key, path in (("view", output_path), ("A", raw_output_a), ("B", raw_output_b)) if path}
            for path in outputs.values():
                path.parent.mkdir(parents=True, exist_ok=True)
            if outputs and record_fps < fps:
                print(f"[Recording] Capturing/detecting at {fps}fps, writing video at {record_fps}fps")
            record_start = time.perf_counter()
            frames_written = 0
            max_catchup_frames_per_iteration = max(1, int(record_fps))

            stats = _SessionStats(stats_csv) if stats_csv else None
            frames_in = _SyncedFrames(caps, live, camera_fps, timestamps)
            for end in caps:
                self.trackers[end].async_person_filter = live[end]
            annotated, raw = {}, {}
            composed_at = 0.0

            while True:
                new, t_or_ended = frames_in.next()
                if new is None:
                    print(f"[Dual] Camera {' and '.join(t_or_ended)} stopped delivering frames -- ending the session")
                    break
                t = t_or_ended
                self.pair_index += 1
                frames = {end: _flip_frame(frame, *flips.get(end, (False, False))) for end, (frame, _, _) in new.items()}
                # Copied before process_frame draws on the frames (see run_video).
                raw.update({end: frames[end].copy() for end in frames if end in outputs})

                if len(frames) == 2:
                    future = executor.submit(self.trackers["A"].process_frame, frames["A"])
                    annotated["B"] = self.trackers["B"].process_frame(frames["B"])
                    annotated["A"] = future.result()
                else:
                    for end, frame in frames.items():
                        annotated[end] = self.trackers[end].process_frame(frame)
                for end, (_, _, captured) in new.items():
                    if live[end]:
                        self.trackers[end].track_processing_rate(captured, camera_fps[end])

                balls = self._update({end: captured for end, (_, _, captured) in new.items()})
                if len(annotated) < 2:
                    continue  # both cameras' first frames are needed to show anything
                now = time.perf_counter()
                # Recorded frames are due at record_fps of wall-clock time (a
                # step can owe several, written as copies, if it ran long).
                record_due = 0
                if outputs and all(end in raw for end in ("A", "B") if end in outputs):
                    record_due = min(int((now - record_start) * record_fps) + 1 - frames_written, max_catchup_frames_per_iteration)
                view = None
                if (record_due > 0 and "view" in outputs) or (show_window and now - composed_at >= 1.0 / self.DISPLAY_FPS):
                    view, composed_at = self._compose(annotated, balls), now
                if stats is not None:
                    # One step = one "frame" here: fps is steps per second, and
                    # latency is from the older new frame's arrival.
                    stats.frame_done(min(arrival for _, arrival, _ in new.values()))

                if record_due > 0:
                    to_write = {"view": view, **raw}
                    if not writers:
                        for key, path in outputs.items():
                            size = (to_write[key].shape[1], to_write[key].shape[0])
                            writers[key] = _BackgroundVideoWriter(path, record_fps, size)
                        print(f"[Recording] Writing at {record_fps}fps: " + ", ".join(str(path) for path in outputs.values()))
                    # Safe to queue without copying: each step's frames, raw copies
                    # and composed view are new arrays, never drawn on again.
                    for key, writer in writers.items():
                        writer.write(to_write[key], copies=record_due)
                    frames_written += record_due

                if show_window and view is not None:
                    if display is None:
                        # Opens at a size that fits a laptop screen; maximize it for
                        # the full resolution.
                        window_width = min(view.shape[1], 1600)
                        display = _FrameDisplay(
                            "Project PickleVision - Dual Camera", window_width, round(view.shape[0] * window_width / view.shape[1])
                        )
                    display.show(view)
                if display is not None and display.quit_requested.is_set():
                    break
        finally:
            executor.shutdown(wait=True)
            if frames_in is not None:
                frames_in.stop()
            for cap in caps.values():
                cap.release()
            for writer in writers.values():
                writer.release()
            if display is not None:
                display.stop()
            if stats is not None:
                stats.close()
            if self.calibrated:
                self.aligner.flush(force=True)
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
    parser.add_argument("--view", choices=["end", "side"], default="end", help="Where the camera watches from: 'end' (behind a baseline, the default) or 'side' (beside the court, e.g. level with the net). Calibrate and track with the same value. For one camera, 'side' calls bounces better: in simulation 95%% of landings right vs 85%% from an end (99%% vs 88%% with carefully clicked calibration corners), since from there a bounce always shows as a down-then-up in the image. Single camera only")
    parser.add_argument("--record-raw", type=str, default=None, metavar="FILE.avi", help="Just record --source's camera (and --source2's: FILE_A.avi, FILE_B.avi), every frame at full rate, nothing else (no detection, no overlays), then exit. Saves each frame's capture time beside it (.timestamps.csv) so two cameras replay in sync. Doesn't lag like --raw-output; replay with --source FILE.avi (--source2 FILE_B.avi)")
    parser.add_argument("--record-seconds", type=float, default=None, help="With --record-raw: stop after this many seconds (default: until 'q' or Ctrl+C)")
    parser.add_argument("--list-cameras", action="store_true", help="List the camera indices that open, with the size and real frame rate each delivers at --width/--height/--fps, then exit -- the ELP is the one that keeps up 120fps")
    parser.add_argument("--calibrate-lens", action="store_true", help="Interactively measure the camera lens's fisheye/barrel bend by clicking points along straight court lines, save it (--lens-output), then exit. Use the result with --lens")
    parser.add_argument("--lens-output", type=str, default=None, help="Where --calibrate-lens saves the lens file (default: lens_cam<source>.json)")
    parser.add_argument("--lens", type=str, default=None, help="Lens file from --calibrate-lens for --source's camera: straightens its wide-angle bend before court mapping, so calibration and line calls match the real lines across the whole image. Use it for --calibrate as well as tracking, at the same --width/--height")
    parser.add_argument("--lens2", type=str, default=None, help="Lens file from --calibrate-lens for --source2's camera")
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

    # With a lens model a straight court line is curved in the camera image,
    # so each one comes back as short pieces along its length (drawn the same
    # way by callers) that follow the bend of the real painted line.
    samples = 2 if mapper.lens is None else 25
    t = np.linspace(0.0, 1.0, samples)[:, None]
    pieces = []
    for start, end in segments:
        court_pts = np.array(start) + (np.array(end) - np.array(start)) * t
        image_pts = mapper.image_points(court_pts)
        pieces += [(tuple(a), tuple(b)) for a, b in zip(image_pts[:-1], image_pts[1:])]
    return pieces


# Arrow keys as cv2.waitKeyEx reports them: Windows, then Linux (GTK).
_ARROW_NUDGES = {
    2424832: (-1, 0), 2555904: (1, 0), 2490368: (0, -1), 2621440: (0, 1),
    65361: (-1, 0), 65363: (1, 0), 65362: (0, -1), 65364: (0, 1),
}


def _draw_magnifier(display, frame, center, zoom=8, half=16):
    """An 8x close-up of the pixels around `center` with a crosshair on the
    exact pixel, in a top corner of `display` (whichever is farther from it)."""
    height, width = frame.shape[:2]
    cx, cy = int(round(center[0])), int(round(center[1]))
    patch = np.zeros((2 * half + 1, 2 * half + 1, 3), np.uint8)
    x0, y0 = cx - half, cy - half
    sx0, sy0, sx1, sy1 = max(x0, 0), max(y0, 0), min(cx + half + 1, width), min(cy + half + 1, height)
    if sx1 > sx0 and sy1 > sy0:
        patch[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = frame[sy0:sy1, sx0:sx1]
    big = cv2.resize(patch, None, fx=zoom, fy=zoom, interpolation=cv2.INTER_NEAREST)
    mid = half * zoom + zoom // 2
    cv2.line(big, (mid, 0), (mid, big.shape[0] - 1), (0, 0, 255), 1)
    cv2.line(big, (0, mid), (big.shape[1] - 1, mid), (0, 0, 255), 1)
    size = big.shape[0]
    ox = width - size - 10 if cx < width / 2 else 10
    if size + 10 <= height and size + 10 <= width:
        display[10:10 + size, ox:ox + size] = big
        cv2.rectangle(display, (ox - 1, 9), (ox + size, 10 + size), (255, 255, 255), 1)


def _draw_loupe(display, frame, cursor, zoom=4, radius=24):
    """A magnified view of the unannotated frame around the mouse, with a
    crosshair, for clicking a line's exact edge or corner. Drawn in the top
    right corner, or the top left when the mouse is over that corner."""
    if cursor is None:
        return
    height, width = frame.shape[:2]
    x, y = cursor
    if not (0 <= x < width and 0 <= y < height):
        return
    x0, y0 = max(0, x - radius), max(0, y - radius)
    patch = frame[y0 : min(height, y + radius + 1), x0 : min(width, x + radius + 1)]
    big = cv2.resize(patch, None, fx=zoom, fy=zoom, interpolation=cv2.INTER_NEAREST)
    cx, cy = (x - x0) * zoom + zoom // 2, (y - y0) * zoom + zoom // 2
    cv2.line(big, (cx, 0), (cx, big.shape[0]), (0, 0, 255), 1)
    cv2.line(big, (0, cy), (big.shape[1], cy), (0, 0, 255), 1)
    big_h, big_w = big.shape[:2]
    left = 10 if (x > width - big_w - 30 and y < big_h + 30) else width - big_w - 10
    if big_h + 10 > height or left < 0:
        return
    display[10 : 10 + big_h, left : left + big_w] = big
    cv2.rectangle(display, (left, 10), (left + big_w, 10 + big_h), (255, 255, 255), 1)


def calibrate_court_corners(source, save_path: str | None = None, court_width: float = 20.0, court_length: float = 44.0, flip_horizontal=False, flip_vertical=False, target_width: int = 1920, target_height: int = 1080, target_fps: int = 120, view: str = "end", lens: "LensModel | None" = None):
    """Interactively click the court's 4 real-world corners on a live camera feed.

    With lens (--lens, from --calibrate-lens) the preview is lens-corrected:
    the green outline and cyan lines curve to follow the real painted lines.
    The saved corners are still the raw pixels clicked, so they're used with
    the same --lens when tracking.

    Click order matters -- it must match CourtMapper's destination rectangle
    (TOP-LEFT, TOP-RIGHT, BOTTOM-RIGHT, BOTTOM-LEFT as seen on screen, i.e.
    clockwise starting from whichever corner you treat as the origin).

    From an end (view="end"), for a full court (court_length=44), TOP = far
    baseline, BOTTOM = near baseline (the one closest to the camera). For a
    half-court calibration (court_length=22, baseline-to-net only -- useful
    when the far half of the court isn't reliably visible from this camera
    angle), TOP = the net line, BOTTOM = the near baseline. From beside the
    court (view="side"), TOP = the far sideline, BOTTOM = the near sideline.

    Click the OUTSIDE corner of the painted lines: the lines are part of the
    court. Calibration precision is most of what's left of line-call error:
    in simulation, clicks off by 2px got 89% of calls within 0.5ft of a line
    right from the side, 0.7px clicks 98%. Hence the magnifier -- an 8x
    close-up with a crosshair on the exact pixel under the mouse -- and the
    arrow keys, which nudge the last point placed by 1px.

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

    clicked: list[list[int]] = []
    focus = {"point": None}

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_MOUSEMOVE:
            focus["point"] = (x, y)
        elif event == cv2.EVENT_LBUTTONDOWN and len(clicked) < 4:
            clicked.append([x, y])
            focus["point"] = (x, y)

    window_name = "Court Calibration"
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, on_mouse)

    if view == "side":
        top_edge, bottom_edge = "the FAR sideline", "the NEAR sideline"
    else:
        top_edge = "the NET line" if abs(court_length - 44.0) >= 1.0 else "the FAR baseline"
        bottom_edge = "the NEAR baseline"
    print(f"Calibrating a {court_width:.0f}x{court_length:.0f} ft region, camera at the {view} of the court.")
    print("Lens correction: " + ("ON" if lens is not None else "off (add --lens from --calibrate-lens for a wide-angle lens)"))
    print(f"Click in order: TOP-LEFT, TOP-RIGHT ({top_edge}), then BOTTOM-RIGHT, BOTTOM-LEFT ({bottom_edge}).")
    print("Click the OUTSIDE corner of the painted lines (the lines are in). The magnifier shows the exact pixel;")
    print("arrow keys nudge the last point 1px. Keys: 'r' reset points | 's' save (once 4 are placed) | 'q' cancel")

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
            if lens is not None and (frame.shape[1], frame.shape[0]) != (lens.width, lens.height):
                raise SystemExit(
                    f"--lens was fitted at {lens.width}x{lens.height} but this feed is {frame.shape[1]}x{frame.shape[0]} "
                    "-- use the same --width/--height, or redo --calibrate-lens at this resolution"
                )

            display = frame.copy()
            for i, pt in enumerate(clicked):
                # Small, so it doesn't hide the corner it marks.
                cv2.circle(display, tuple(pt), 3, (0, 0, 255), -1)
                cv2.putText(display, str(i + 1), (pt[0] + 8, pt[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            if len(clicked) == 4:
                if lens is None:
                    pts = np.array(clicked, dtype=np.int32)
                    cv2.polylines(display, [pts], isClosed=True, color=(0, 255, 0), thickness=1)
                cv2.putText(display, "Press 's' to save, 'r' to reset", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                try:
                    preview_mapper = CourtMapper(
                        src_points=np.array(clicked, dtype=np.float32),
                        court_width=court_width,
                        court_length=court_length,
                        lens=lens,
                        view=view,
                    )
                    if lens is not None:
                        # The outline between the clicked corners, bent like the real lines.
                        corners = preview_mapper.dst_points
                        t = np.linspace(0.0, 1.0, 25)[:, None]
                        for a, b in zip(corners, np.roll(corners, -1, axis=0)):
                            edge = preview_mapper.image_points(a + (b - a) * t).astype(np.int32)
                            cv2.polylines(display, [edge], isClosed=False, color=(0, 255, 0), thickness=2)
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
                cv2.putText(display, f"Click corner {len(clicked) + 1}/4 (outside corner of the lines)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            if focus["point"] is not None:
                _draw_magnifier(display, frame, focus["point"])

            cv2.imshow(window_name, display)
            key = cv2.waitKeyEx(1)
            if key in _ARROW_NUDGES and clicked:
                dx, dy = _ARROW_NUDGES[key]
                clicked[-1][0] += dx
                clicked[-1][1] += dy
                focus["point"] = tuple(clicked[-1])
                continue
            key &= 0xFF
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

    print(f"\n{'--view side ' if view == 'side' else ''}--court-corners \"{corner_str}\"")
    if save_path:
        Path(save_path).write_text(corner_str)
        print(f"Saved to {save_path}" + (" -- track with --view side too" if view == "side" else ""))

    return corner_str


_LENS_LINE_COLORS = [(0, 200, 255), (255, 0, 255), (0, 255, 0), (255, 128, 0), (128, 0, 255), (0, 128, 255), (255, 255, 255)]

# --calibrate-lens walks through these in order (an end camera's view of its
# half), then takes optional extra lines. The bend is strongest near the
# image edges, which the sidelines and baseline reach.
_LENS_LINE_PLAN = ["the NEAR BASELINE", "the LEFT SIDELINE", "the RIGHT SIDELINE", "the KITCHEN LINE", "the CENTERLINE"]
_LENS_POINTS_PER_LINE = 5


def calibrate_lens(source, save_path, flip_horizontal=False, flip_vertical=False, target_width=1280, target_height=720, target_fps=120):
    """--calibrate-lens: measure a wide-angle lens's bend from straight court lines.

    Guided and mouse-only: the window names the line to click next (near
    baseline, both sidelines, kitchen line, centerline), and after
    _LENS_POINTS_PER_LINE clicks spread along it moves on to the next by
    itself. From the 2nd line on, the bend is fitted and each line's fitted
    straight line is drawn back over the feed in cyan, bent the way this lens
    bends it: it should run along the real paint. From the 3rd line on the
    result is saved automatically after every line, so closing the window
    keeps it. After the planned lines, any other straight line (a fence rail,
    a wall edge -- ideally near the image edges) can be added the same way.

    Optional keys (calibration window focused): right-click or 'n' ends a
    line early (3+ points), 'u' undo, 'r' reset, 'v' straightened view,
    'q' close. Returns the saved LensModel, or None if nothing was saved.
    """
    cap, _ = _open_video_source(source, target_width, target_height, target_fps)
    lines: list[list[tuple[int, int]]] = [[]]
    cursor: list[tuple[int, int]] = []
    state = {"straight": False, "fit": None, "fit_key": None, "maps": None, "saved": None, "quit_armed": False}

    def next_line():
        if len(lines[-1]) >= 3:
            lines.append([])

    def on_mouse(event, x, y, flags, param):
        cursor[:] = [(x, y)]
        if state["straight"]:
            return
        if event == cv2.EVENT_LBUTTONDOWN:
            state["quit_armed"] = False
            lines[-1].append((x, y))
            if len(lines[-1]) >= _LENS_POINTS_PER_LINE:
                next_line()
        elif event == cv2.EVENT_RBUTTONDOWN:
            next_line()

    window = "Lens Calibration"
    cv2.namedWindow(window)
    cv2.setMouseCallback(window, on_mouse)
    print(f"Click {_LENS_POINTS_PER_LINE} points spread along each line the window names -- it moves to the next line by itself.")
    print(f"From the 3rd line on it saves automatically to {save_path}; close the window (or 'q') when the cyan lines follow the paint.")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Failed to read frame from camera.")
                break
            frame = _flip_frame(frame, flip_horizontal, flip_vertical)
            height, width = frame.shape[:2]

            usable = [line for line in lines if len(line) >= 3]
            fit_key = tuple(tuple(line) for line in usable)
            if len(usable) < 2:
                state["fit"] = state["fit_key"] = None
                state["straight"] = False
            elif fit_key != state["fit_key"]:
                state["fit"] = fit_lens_from_lines(usable, width, height)
                state["fit_key"], state["maps"] = fit_key, None
                if len(usable) >= 3:
                    lens, before, after = state["fit"]
                    lens.save(
                        save_path,
                        method="court lines (plumb-line fit)",
                        lines=len(usable),
                        points=sum(len(line) for line in usable),
                        rms_before_px=round(before, 2),
                        rms_after_px=round(after, 2),
                    )
                    state["saved"] = state["fit"]
            fit = state["fit"]
            lens = fit[0] if fit is not None else None
            straight_view = state["straight"] and lens is not None

            if straight_view:
                if state["maps"] is None:
                    state["maps"] = cv2.initUndistortRectifyMap(lens.K, lens.D, None, lens.K, (width, height), cv2.CV_16SC2)
                display = cv2.remap(frame, *state["maps"], cv2.INTER_LINEAR)
            else:
                display = frame.copy()

            for i, line in enumerate(lines):
                if not line:
                    continue
                color = _LENS_LINE_COLORS[i % len(_LENS_LINE_COLORS)]
                pts = np.array(line, dtype=np.float64)
                shown = lens.undistort(pts) if straight_view else pts
                for x, y in shown:
                    cv2.circle(display, (int(round(x)), int(round(y))), 4, color, -1)
                if lens is not None and len(line) >= 3:
                    # The straight line through this line's points, drawn as
                    # the lens bends it (or straight, in the straightened view).
                    straight = lens.undistort(pts)
                    center = straight.mean(axis=0)
                    direction = np.linalg.svd(straight - center)[2][0]
                    along = (straight - center) @ direction
                    span = np.linspace(along.min(), along.max(), 40)[:, None]
                    segment = center + span * direction
                    curve = segment if straight_view else lens.distort(segment)
                    cv2.polylines(display, [curve.astype(np.int32)], False, (255, 255, 0), 1)

            index, current = len(lines) - 1, len(lines[-1])
            planned = len(_LENS_LINE_PLAN)
            if index < planned:
                step = f"STEP {index + 1} of {planned}: click {_LENS_POINTS_PER_LINE} points spread along {_LENS_LINE_PLAN[index]}"
            else:
                step = f"All {planned} lines done! Optional extra: any other straight line (fence rail, wall edge)"
            status = [(f"{step}  ({current}/{_LENS_POINTS_PER_LINE} points)", (0, 255, 255))]
            if index > 0 and current == 0:
                previous = _LENS_LINE_PLAN[index - 1] if index - 1 < planned else "that line"
                status.append((f"{previous[4:].capitalize() if previous.startswith('the ') else previous} done!", (0, 255, 0)))
            if fit is not None:
                _, before, after = fit
                status.append((f"Lens bend: {before:.1f} px off straight -> {after:.1f} px corrected. Cyan lines should follow the paint.", (0, 255, 255)))
            if state["quit_armed"]:
                status.append(("NOT SAVED YET -- press 'q' again to quit WITHOUT saving, or keep clicking", (0, 0, 255)))
            elif state["saved"] is not None:
                done_note = "all lines done -- close the window" if index >= planned else "keep going for the best result, or close the window"
                status.append((f"SAVED to {save_path} -- {done_note}", (0, 255, 0)))
            else:
                remaining = 3 - len(usable)
                status.append((f"NOT SAVED YET -- finish {remaining} more line{'s' if remaining != 1 else ''} (it saves by itself after line 3)", (0, 128, 255)))
            for i, (text, color) in enumerate(status):
                _draw_label(display, text, (10, 28 + 26 * i), color)
            if straight_view:
                _draw_label(display, "STRAIGHTENED VIEW -- press 'v' to go back to clicking", (10, height - 15), (255, 255, 0))
            else:
                _draw_loupe(display, frame, cursor[0] if cursor else None)

            cv2.imshow(window, display)
            key = cv2.waitKey(1) & 0xFF
            if cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                break
            if key in (ord("q"), ord("s")):
                # Quitting before anything is saved needs a second press: it's
                # easy to take the next line's "0/5" for "finished".
                if state["saved"] is not None or state["quit_armed"]:
                    break
                state["quit_armed"] = True
            elif key != 255:
                state["quit_armed"] = False
            if key == ord("n"):
                next_line()
            if key == ord("u"):
                if lines[-1]:
                    lines[-1].pop()
                elif len(lines) > 1:
                    lines.pop()  # back into the previous line
                    lines[-1].pop()
            if key == ord("r"):
                lines[:] = [[]]
                state["saved"] = None
            if key == ord("v") and lens is not None:
                state["straight"] = not state["straight"]
    finally:
        cap.release()
        cv2.destroyAllWindows()

    if state["saved"] is None:
        print("Lens calibration closed before 3 lines were done -- nothing saved.")
        return None
    lens, before, after = state["saved"]
    print(f"Lens bend: {before:.1f} px off straight -> {after:.1f} px after correction (k1 {lens.D[0]:+.4f}, k2 {lens.D[1]:+.4f})")
    if before < 1.0:
        print("Note: the lines were barely bent -- if they were all near the image center, add lines near the edges.")
    print(f"Saved to {save_path}. Use it with --lens {save_path} (court calibration, tracking; --lens2 for --source2's camera).")
    return lens


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

    With a lens model these are measured in the straightened image, which is
    what the geometry needs; lens bend is radial about the image center, so
    offsets from the center guide mean the same thing there.
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
    lens_a: "LensModel | None" = None,
    lens_b: "LensModel | None" = None,
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
        end: CourtMapper(src_points=CourtMapper.parse_corners(corners), court_width=court_width, court_length=half_length, lens=lens)
        if corners
        else None
        for end, corners, lens in (("A", corners_a, lens_a), ("B", corners_b, lens_b))
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


class _RawRecorder:
    """One camera of --record-raw: a capture thread reading every frame the
    camera delivers, stamping its capture time, and a writer thread encoding
    it (MJPEG .avi) and its time (.timestamps.csv beside it). The queue
    between them absorbs encoding hiccups, so capture never waits on disk."""

    def __init__(self, cap, path, fps, start):
        import queue
        import threading

        self.cap, self.path, self.start = cap, path, start
        self.size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        self.fps = fps
        self._writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), fps, self.size)
        if not self._writer.isOpened():
            raise SystemExit(f"Couldn't open {path} for writing")
        self._times = open(_timestamps_path(path), "w")
        self._times.write("frame,seconds\n")
        self._queue: "queue.Queue" = queue.Queue(maxsize=fps * 4)  # up to ~4 s of backlog
        self.latest = None
        self.captured = self.written = self.dropped = 0
        self.failed = False
        self._stopped = threading.Event()
        self._threads = [threading.Thread(target=self._capture, daemon=True), threading.Thread(target=self._write, daemon=True)]
        for thread in self._threads:
            thread.start()

    def _capture(self):
        import queue

        while not self._stopped.is_set():
            try:
                ok, frame = self.cap.read()
            except cv2.error:
                ok = False
            if not ok:
                self.failed = True
                break
            stamp = time.perf_counter() - self.start
            self.captured += 1
            self.latest = frame
            try:
                self._queue.put_nowait((frame, stamp))
            except queue.Full:
                self.dropped += 1  # encoding fell ~4 s behind; keep capturing live
        self._queue.put(None)

    def _write(self):
        while True:
            item = self._queue.get()
            if item is None:
                return
            frame, stamp = item
            self._writer.write(frame)
            self._times.write(f"{self.written},{stamp:.6f}\n")
            self.written += 1

    def stop(self):
        self._stopped.set()
        for thread in self._threads:
            thread.join()
        self._writer.release()
        self._times.close()
        self.cap.release()


def record_raw(sources, path, target_width, target_height, target_fps, show_window=True, seconds=None):
    """--record-raw: save every frame the camera(s) deliver, and nothing else
    -- footage the tracker can be re-run on later (--source <file>, and
    --source2 for the second camera's), as many times as needed.

    Recording through the tracker (--raw-output) lags: every frame is also
    detected and re-encoded on the same machine, and it's frame-paced to the
    wall clock, so behind schedule it writes duplicate frames. Here nothing
    else runs (see _RawRecorder). Each video gets its frames' capture times
    beside it (.timestamps.csv): the tracker replays two cameras' videos in
    sync by those (_SyncedFrames), and times detection to the rate the camera
    really delivered. sources is a list of one or two cameras; with two, path
    raw1.avi saves raw1_A.avi and raw1_B.avi. Stops on 'q' in the preview,
    Ctrl+C, or after `seconds`.
    """
    path = Path(path)
    if path.suffix.lower() != ".avi":
        path = path.with_suffix(".avi")
        print(f"[Record] MJPEG needs an .avi file -- saving to {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    ends = ["A", "B"][: len(sources)]
    paths = {end: path.with_name(f"{path.stem}_{end}.avi") if len(sources) > 1 else path for end in ends}
    caps = {}
    try:
        for end, source in zip(ends, sources):
            caps[end], _ = _open_video_source(source, target_width, target_height, target_fps)
    except Exception:
        for cap in caps.values():
            cap.release()
        raise
    start = time.perf_counter()
    recorders = {end: _RawRecorder(caps[end], paths[end], int(caps[end].get(cv2.CAP_PROP_FPS)) or 30, start) for end in ends}
    for end, source in zip(ends, sources):
        rec = recorders[end]
        label = f"Camera {end} ({source})" if len(sources) > 1 else f"Camera {source}"
        print(f"[Record] {label}: {rec.size[0]}x{rec.size[1]} @ {rec.fps}fps -> {rec.path}")
    print("[Record] Recording -- 'q' in the preview or Ctrl+C to stop")
    display = None
    if show_window:
        width = 1280 if len(sources) > 1 else min(recorders["A"].size[0], 1280)
        display = _FrameDisplay("Recording -- 'q' to stop", width, 360 if len(sources) > 1 else 720, max_fps=30)
    try:
        while seconds is None or time.perf_counter() - start < seconds:
            if any(rec.failed for rec in recorders.values()):
                print("[Record] " + ", ".join(f"camera {end}" for end, rec in recorders.items() if rec.failed) + " stopped delivering frames.")
                break
            if display is not None:
                latest = [rec.latest for rec in recorders.values()]
                if all(frame is not None for frame in latest):
                    display.show(np.hstack([cv2.resize(frame, (640, 360)) for frame in latest]) if len(latest) > 1 else latest[0])
                if display.quit_requested.is_set():
                    break
            time.sleep(1 / 30)
    except KeyboardInterrupt:
        pass
    finally:
        elapsed = time.perf_counter() - start
        for rec in recorders.values():
            rec.stop()
        if display is not None:
            display.stop()
    for end, rec in recorders.items():
        label = f"Camera {end}" if len(sources) > 1 else "Camera"
        print(f"[Record] {label}: {rec.captured} frames in {elapsed:.1f}s ({rec.captured / max(elapsed, 1e-9):.0f} fps delivered), "
              f"{rec.written} saved, {rec.dropped} dropped -> {rec.path}")
    if len(sources) > 1:
        print(f"[Record] Track them later with: --source \"{paths['A']}\" --source2 \"{paths['B']}\"")
    else:
        print(f"[Record] Track it later with: --source \"{path}\"")


def list_cameras(target_width, target_height, target_fps, max_index=6):
    """--list-cameras: each camera index that opens, with the size it gives
    and the frame rate it really delivers at the requested mode (measured,
    not what the driver reports) -- the ELP is the one keeping up 120fps."""
    print(f"Checking camera indices 0-{max_index - 1} at {target_width}x{target_height} @ {target_fps}fps (MJPEG)...")
    # OpenCV warns for every index with no camera behind it -- expected here.
    log_level = cv2.getLogLevel()
    cv2.setLogLevel(2)  # errors only
    found = 0
    for index in range(max_index):
        cap = _open_camera(index)
        if not cap.isOpened():
            cap.release()
            continue
        found += 1
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, target_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, target_height)
        cap.set(cv2.CAP_PROP_FPS, target_fps)
        width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frames, measured = 0, 0.0
        try:
            for _ in range(5):  # the first frames after opening come slowly
                cap.read()
            start = time.perf_counter()
            while time.perf_counter() - start < 1.5:
                ok, _ = cap.read()
                if not ok:
                    break
                frames += 1
            measured = frames / (time.perf_counter() - start)
        except cv2.error:
            pass
        print(f"  --source {index}: {width}x{height}, delivering ~{measured:.0f} fps")
        cap.release()
    cv2.setLogLevel(log_level)
    if not found:
        print("  No cameras opened.")


def _load_lens(path, args, flag="--lens"):
    """The LensModel saved by --calibrate-lens at path, or None if no path.
    Refuses one fitted at a different resolution than --width/--height: USB
    cameras change their field of view between modes, so it wouldn't fit."""
    if not path:
        return None
    lens = LensModel.load(path)
    if (lens.width, lens.height) != (args.width, args.height):
        raise SystemExit(
            f"{flag} {path} was fitted at {lens.width}x{lens.height}, but --width/--height is {args.width}x{args.height} "
            "-- use the same resolution, or redo --calibrate-lens at this one"
        )
    print(f"[Lens] {flag} {path}: correcting lens bend (k1 {lens.D[0]:+.4f}, k2 {lens.D[1]:+.4f})")
    return lens


def _build_tracker(args, court_corners, camera_id="cam0", shared_roboflow_model=None, lens=None):
    """A PickleVisionTracker configured from the command line, calibrated with
    court_corners (a --court-corners string), or uncalibrated if it's None."""
    court_points = CourtMapper.parse_corners(court_corners) if court_corners else None
    court_mapper = (
        CourtMapper(src_points=court_points, court_width=args.court_width, court_length=args.court_length, lens=lens, view=args.view)
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


def _warm_up(tracker, args, *others):
    print(f"[Device] Running inference on: {tracker.device}")
    if tracker.roboflow_local_model is not None:
        # Warm up here so the one-time TensorRT engine build (or CUDA init)
        # happens before the camera opens, not as a multi-minute freeze on
        # the first live frame.
        tracker.roboflow_local_model.infer(np.zeros((args.height, args.width, 3), np.uint8))
        print(f"[Device] Model backend: {tracker.roboflow_local_model.onnx_session.get_providers()[0]}")
    for each in (tracker, *others):
        if each.person_model is not None:
            # Its first call sets up CUDA kernels (~1 s): otherwise the first
            # second of tracking runs at a third of its speed.
            each._find_people(np.zeros((args.height, args.width, 3), np.uint8))


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

    tracker_a = _build_tracker(args, args.court_corners, camera_id="A", lens=_load_lens(args.lens, args))
    # One copy of the detection model serves both cameras (see _LOCAL_MODEL_LOCK).
    tracker_b = _build_tracker(
        args,
        args.court_corners2,
        camera_id="B",
        shared_roboflow_model=tracker_a.roboflow_local_model,
        lens=_load_lens(args.lens2, args, "--lens2"),
    )
    _warm_up(tracker_a, args, tracker_b)

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

    if args.record_raw:
        sources = [source] if args.source2 is None else [source, int(args.source2) if str(args.source2).isdigit() else args.source2]
        record_raw(sources, args.record_raw, args.width, args.height, args.fps, show_window=args.show, seconds=args.record_seconds)
        return

    if args.list_cameras:
        list_cameras(args.width, args.height, args.fps)
        return

    if args.view == "side" and (args.source2 is not None or args.check_alignment):
        raise SystemExit("--view side is for a single camera; the two-camera modes use end cameras")
    if args.view == "side" and abs(args.court_length - 44.0) >= 1.0:
        raise SystemExit("--view side calibrates the whole court -- leave --court-length at 44")

    if args.calibrate_lens:
        calibrate_lens(
            source,
            save_path=args.lens_output or f"lens_cam{Path(str(args.source)).stem}.json",
            flip_horizontal=args.flip_horizontal,
            flip_vertical=args.flip_vertical,
            target_width=args.width,
            target_height=args.height,
            target_fps=args.fps,
        )
        return

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
            view=args.view,
            lens=_load_lens(args.lens, args),
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
            lens_a=_load_lens(args.lens, args),
            lens_b=_load_lens(args.lens2, args, "--lens2"),
        )
        return

    if args.source2 is not None:
        _run_dual_camera(args, source)
        return

    tracker = _build_tracker(args, args.court_corners, lens=_load_lens(args.lens, args))
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
    ins = sum(event.line_call == "IN" for event in events)
    if tracker.court_calibrated:
        print(f"\n[Summary] {len(events)} bounces called: {ins} IN, {len(events) - ins} OUT.")
    else:
        print(f"\n[Summary] {len(events)} bounces detected -- calibrate (--court-corners) for IN/OUT calls.")


if __name__ == "__main__":
    main()
