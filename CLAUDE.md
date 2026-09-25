# Project PickleVision

Computer-vision pickleball line-calling/analytics system (Cpe-401 thesis project). Team: Merl, Franz, Arjhon. All tracking logic lives in the single file `tracker.py`.

## Setup

Follow `requirements.txt`'s header comment exactly — install order matters (`inference` drags in a CPU-only `onnxruntime` that must be overwritten last with the GPU build). No CUDA Toolkit/cuDNN system install needed; the NVIDIA libraries come from pip. First launch of `--roboflow-local` with TensorRT available builds an optimized engine for that specific GPU (~4-5 min, one time only, then ~17s on later launches) — don't panic if it looks stuck.

Roboflow model: `pickleball-prototype/3` (workspace `franzs-workspace-utuz0`), weights cache into `./model_cache` after first fetch (offline after that).

## Hardware / camera architecture

**Finalized 3-camera plan** (from the thesis panel, not written in `Cpe-401_Grp3_Proposal Paper.pdf`): two cameras at opposite baseline ends of the court facing each other (the ELP 120fps USB camera + one more), plus one overhead camera mounted above mid-court looking straight down. The three roles are NOT interchangeable:

- **End cameras (2)** — best source for *when* a bounce happens (Y-axis trajectory-reversal logic in `_detect_ball_contact`, needs an elevated side-on angle). Each covers its own near half only (baseline to net) via `--court-length 22` half-court calibration — higher detail per half than one camera straining across the full 44ft, plus free redundancy (a player blocking one end's view is unlikely to block the other's).
- **Overhead camera (1, not yet acquired)** — best source for accurate ground-plane (x,y) for the actual IN/OUT call (near-orthographic homography, minimal distortion), but Z-blind: cannot tell ball height, so cannot detect bounce timing on its own. Needs a pole/truss/ceiling mount, not a tripod — a real infrastructure ask the end cameras don't have.
- Fusion rule (not yet implemented): whichever end camera currently has the ball on its own near half is the timing/position source for that half; the overhead camera, once added, is the line-call authority regardless of which end camera flagged the bounce.

**Coordinate unification across the 2 end cameras** (math already implemented — see `to_global_court_point`/`check_camera_alignment` in `tracker.py`): each end camera calibrates its own half independently (local Y in [0, 22], 0 = net). To get one global court Y in [0, 44] (0 = end-A baseline, 22 = net, 44 = end-B baseline): `global_y_A = 22 - local_y_A`, `global_y_B = 22 + local_y_B`. X is passed through unchanged — **only valid if both cameras are calibrated against the same fixed physical reference**, which is why both end cameras are mounted centered on the court's width, aligned with the centerline (this is the actual physical mounting decision, and it's what makes "left" consistent between the two cameras instead of mirrored).

**Compute:** this dev laptop has an RTX 3050 (4GB VRAM); Franz's laptop has an RTX 4060 (measured 3 concurrent 720p streams at 120+ FPS each, well above what's needed). Benchmarked on the RTX 3050 with the real Roboflow model end-to-end: 1080p = 41.3 FPS, 720p = 85.2 FPS single-stream — 720p is the resolution to target for multi-camera headroom on this machine.

**Open idea, deliberately held:** run the 2 end cameras at 720p but the overhead camera at 1080p (it covers 2x the court area of one end camera's half, so needs the extra resolution for comparable detail-per-foot — and it's the camera whose accuracy matters most for the actual line call). Current code doesn't support per-camera resolution yet (`--width`/`--height`/`--fps` are single global flags shared by every camera). Before building this: have Franz benchmark a mixed sequential stream (2x720p + 1x1080p) on his RTX 4060 — don't assume from the RTX 3050's uniform-resolution numbers, since part of the 1080p-vs-720p gap may be CPU-side frame capture/resize rather than pure GPU model cost.

**Dual end-camera build:** the alignment math and a live diagnostic tool both already exist (see below) but the actual dual-camera *tracking* fusion loop is NOT yet wired into `run_video` — that's the next real step once both end cameras and their calibrations are confirmed good via the alignment tool.

## The 3 CLI commands

1. **Live tracking**: `python tracker.py --source 0 --roboflow-local --court-corners "..." --court-length 22`
2. **Calibrate homography** (interactive, click 4 court corners): `python tracker.py --source 0 --calibrate --court-length 22 --calibration-output corners_a.txt`
3. **Dual end-camera alignment check** (verify both end cameras' calibrations agree on the same real-world point): `python tracker.py --check-alignment --source 0 --court-corners "$(cat corners_a.txt)" --source2 1 --court-corners2 "$(cat corners_b.txt)" --court-length 22`

## Known gotchas

- `OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS=0` must be set (it is, at the top of `tracker.py`) or `cv2.VideoCapture` on Windows takes 50-60s to open a USB camera.
- A phone running Camo (camo.com) works fine as a temporary second webcam for testing — it shows up as a normal `cv2.VideoCapture` index, no code changes needed. Just watch for USB-vs-WiFi latency and free-tier resolution/fps caps vs. the real ELP camera's 120fps MJPEG capability.
- Never attempt to bypass Roboflow's weights-export paywall — `--roboflow-local`/`get_model()` is the legitimate offline path already in use.
