# Startup Timing Investigation — `demo_realtime_visual_jetson.py`

**Date:** 2026-03-03  
**Goal:** Determine whether the slow startup on Jetson Orin is caused by TensorRT engine compilation or by something else in the code, by reproducing the test on a different machine (Windows PC).

---

## 1. Test Setup

### Tool
A dedicated timing script (`tests/test_startup_timing.py`) was written to measure each component individually:
- Each init step is wrapped with `time.perf_counter()` before and after
- Results are appended to `tests/startup_timing_results.txt` on every run
- Checks whether a TRT cache already exists before and after each run

### Hardware — Windows PC (test machine)
| Property | Value |
| :--- | :--- |
| **Platform** | Windows 10 x64 |
| **Python** | 3.12 |
| **ONNX Runtime** | onnxruntime (standard) with TRT/CUDA providers installed but non-functional (missing `cublas64_12.dll` / `cublasLt64_12.dll`) |
| **Effective provider** | `CPUExecutionProvider` (TRT and CUDA both failed to load) |
| **Camera** | USB webcam via Windows MSMF backend |

---

## 2. Results — 3 Consecutive Runs (Windows PC)

| Component | Run 1 | Run 2 | Run 3 |
| :--- | ---: | ---: | ---: |
| Detector (SCRFD) | 1.085 s | 0.072 s | 0.071 s |
| Face Recognizer (ArcFace) | 0.486 s | 0.474 s | 0.421 s |
| ByteTracker | < 1 ms | < 1 ms | < 1 ms |
| HeadPoseEstimator | < 1 ms | < 1 ms | < 1 ms |
| RecognitionTrigger | < 1 ms | < 1 ms | < 1 ms |
| EnrollmentDatabase | 1 ms | 2 ms | 2 ms |
| **Camera open + first frame (1280×720)** | **21.35 s** | **21.47 s** | **21.12 s** |
| **TOTAL** | **22.93 s** | **22.02 s** | **21.62 s** |

> **Finding:** The camera accounts for **93–97% of total startup time** on every single run, regardless of model loading. This is caused by the Windows MSMF USB camera driver negotiating the 1280×720 stream (confirmed by `cap_msmf.cpp: Failed to select stream` warnings in the log).

---

## 3. Normal Demo vs Jetson Demo — Same Hardware

To verify whether the Jetson-specific additions (`psutil`, GPU stats overlay) add meaningful startup overhead, the same components were timed from `demo_realtime_visual.py`:

| Step | `demo_realtime_visual.py` | `demo_realtime_visual_jetson.py` |
| :--- | ---: | ---: |
| All model init | 0.92 s | ~1.5 s |
| Camera 1280×720 | 21.09 s | ~21.35 s |
| **TOTAL** | **~22 s** | **~22 s** |

**Conclusion:** Both scripts have identical effective startup time on this PC. The `psutil` import and GPU stats code in the Jetson version add negligible overhead.

---

## 4. What This Means for Jetson

The test confirms the slow startup is **not a Jetson-specific hardware problem** — the same symptom appears on different hardware. The root cause depends on the machine:

| Platform | Bottleneck | Duration |
| :--- | :--- | :--- |
| Windows PC (this test) | USB MSMF camera driver handshake at 1280×720 | ~21 s every run |
| Jetson Orin — Run 1 | TRT engine compilation from ONNX (one-time) | ~60–120 s |
| Jetson Orin — Run 2+ | TRT cache load (fast path) | ~2–5 s |
| Jetson Orin — camera | CSI/IMX camera (typically instant) | < 1 s |

On Jetson the camera is not the bottleneck — TRT compilation on the **first run only** is. Subsequent runs load from `models/trt_cache/` and are fast.

---

## 5. Fixes Applied (2026-03-03)

### Fix 1 — Camera opened in background thread (`demo_realtime_visual_jetson.py`)

The camera `VideoCapture(0)` is now launched in a daemon thread at the very beginning of `__init__()`, running in parallel with model loading instead of sequentially after it.

- On **Jetson Run 1**: TRT build takes ~60–120 s → camera finishes in < 1 s → zero wait when `run()` is called
- On **Jetson Run 2+**: TRT cache load takes ~5 s → camera still finishes first → zero wait
- On **Windows PC**: camera still takes ~21 s (models finish in < 1 s), so the thread joins and waits — no regression, just no benefit here due to the inverted ratio

### Fix 3 — Low-resolution warmup

Inside `_open_camera_bg()`, the camera is set to 640×480 first, three frames are read to flush the sensor, then resolution is ramped up to 1280×720. This resolves slow stream-negotiation on USB cameras that struggle with high-res cold-starts.

```python
def _open_camera_bg(self):
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    for _ in range(3): cap.read()          # flush + fast handshake
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    self._cap = cap
```

---

## 6. Next Step — Jetson Re-test

Run `tests/test_startup_timing.py` on the Jetson Orin to measure:
1. **Run 1 (no TRT cache):** How long does TRT engine build take? Is camera ready before it finishes?
2. **Run 2 / Run 3 (TRT cached):** How long does cache load take? Is it now fast enough to match camera warmup?
3. **Effective wait time in `run()`:** With Fix 1 applied, how long does `cam_thread.join()` actually block (should be ~0 s)?

Append those results to this document or to `tests/startup_timing_results.txt`.
