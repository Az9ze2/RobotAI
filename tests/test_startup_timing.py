"""
Startup Timing Test for demo_realtime_visual_jetson.py
=======================================================
Measures initialization time for each pipeline component.
Run this script MULTIPLE TIMES (1st, 2nd, 3rd) to check if
TensorRT cache reduces startup time after the first run.

Usage:
    python tests/test_startup_timing.py

Expected behaviour:
  Run 1  → TRT builds engine from ONNX  →  slow  (~30-120s)
  Run 2+ → TRT loads cached engine      →  fast  (~2-10s)

If Run 2 and Run 3 are still slow, the bottleneck is NOT the
TRT engine-build step — it is elsewhere in the init code.
"""

import sys
import time
import os
from pathlib import Path
from datetime import datetime

# ── path setup ────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

LOG_FILE = Path(__file__).parent / "startup_timing_results.txt"

# ── helper ────────────────────────────────────────────────────
def ts():
    """Return current wall-clock time as a readable string."""
    return datetime.now().strftime("%H:%M:%S.%f")[:-3]

def section(label: str):
    line = f"\n{'='*60}\n  {label}\n{'='*60}"
    print(line)
    return line

results = []

def timed(label: str, fn):
    """Run fn(), print and record how long it took. Return the result."""
    print(f"\n[{ts()}]  ▶  {label} ...")
    t0 = time.perf_counter()
    result = fn()
    elapsed = time.perf_counter() - t0
    msg = f"[{ts()}]  ✔  {label}  →  {elapsed:.3f}s"
    print(msg)
    results.append((label, elapsed))
    return result


# ── check TRT cache ───────────────────────────────────────────
cache_dir = Path("models/trt_cache")
cache_files = list(cache_dir.glob("*")) if cache_dir.exists() else []
cache_status = (
    f"TRT cache dir: {cache_dir.resolve()}\n"
    f"Cache files ({len(cache_files)}): "
    + (", ".join(f.name for f in cache_files) if cache_files else "NONE — first run will be slow")
)
print(section("Environment Check"))
print(cache_status)

import onnxruntime as ort
providers = ort.get_available_providers()
print(f"ORT available providers: {providers}")

# ── time each component ───────────────────────────────────────
print(section("Component Initialization Timing"))

# 1. Detector (most likely TRT bottleneck)
from vision.detector_factory import create_scrfd_detector
detector = timed(
    "Detector (SCRFD + TensorRT)",
    lambda: create_scrfd_detector(
        model_path="models/buffalo_l/det_10g.onnx",
        confidence_threshold=0.5,
        nms_threshold=0.4,
        input_size=(640, 640),
        device="cuda"
    )
)
active_ep = detector.session.get_providers()[0]
print(f"   → Active execution provider: {active_ep}")

# 2. Face Recognizer (also uses ONNX/TRT)
from vision.recognizer import FaceRecognizer
recognizer_ok = False
def _load_recognizer():
    global recognizer_ok
    try:
        r = FaceRecognizer(
            model_path="models/arcface_r100_v1_fp16.onnx",
            device="cuda"
        )
        recognizer_ok = True
        return r
    except Exception as e:
        print(f"   → Recognizer unavailable: {e}")
        return None

recognizer = timed("Face Recognizer (ArcFace + TensorRT)", _load_recognizer)

# 3. Tracker (pure Python/numpy, should be instant)
from vision.tracker import ByteTracker
timed(
    "ByteTracker",
    lambda: ByteTracker(track_thresh=0.5, track_buffer=30, match_thresh=0.4, min_box_area=100)
)

# 4. Head pose (pure numpy, should be instant)
from vision.head_pose import HeadPoseEstimator
timed(
    "HeadPoseEstimator",
    lambda: HeadPoseEstimator(yaw_threshold=25, pitch_threshold=15, roll_threshold=30)
)

# 5. Recognition trigger
from vision.recognition_trigger import RecognitionTrigger
timed(
    "RecognitionTrigger",
    lambda: RecognitionTrigger(cooldown_seconds=5.0, require_attention=True)
)

# 6. Database
from vision.database import EnrollmentDatabase
db = timed(
    "EnrollmentDatabase",
    lambda: EnrollmentDatabase("data/enrollments.json")
)

# 7. Camera open
import cv2
def _open_camera():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    ret, _ = cap.read()
    cap.release()
    return ret

timed("Camera open + first frame read", _open_camera)

# ── summary ───────────────────────────────────────────────────
print(section("TIMING SUMMARY"))
total = sum(e for _, e in results)
rows = []
for label, elapsed in results:
    pct = elapsed / total * 100 if total > 0 else 0
    row = f"  {elapsed:7.3f}s  ({pct:5.1f}%)  {label}"
    print(row)
    rows.append(row)
print(f"\n  {'─'*50}")
print(f"  {total:7.3f}s  TOTAL init time\n")

# Re-check TRT cache after run
cache_files_after = list(cache_dir.glob("*")) if cache_dir.exists() else []
trt_note = (
    f"\nTRT cache after run: {len(cache_files_after)} file(s)"
    + ((" — engine was BUILT this run (first time)" if len(cache_files_after) > len(cache_files) else " — engine was LOADED from cache (fast path)"))
    if cache_dir.exists() else ""
)
print(trt_note)

# ── write log file ────────────────────────────────────────────
run_num = 1
if LOG_FILE.exists():
    existing = LOG_FILE.read_text(encoding="utf-8")
    run_num = existing.count("=== RUN") + 1

with open(LOG_FILE, "a", encoding="utf-8") as f:
    f.write(f"\n\n=== RUN {run_num}  [{datetime.now().isoformat()}] ===\n")
    f.write(f"Platform: {sys.platform}  |  ORT providers: {providers}\n")
    f.write(f"TRT cache files before: {len(cache_files)}  after: {len(cache_files_after)}\n")
    f.write("\n".join(rows))
    f.write(f"\n  {'─'*50}\n  {total:.3f}s  TOTAL\n")

print(f"\n📄 Results appended to: {LOG_FILE.resolve()}")
print("   Run this script again (2nd and 3rd time) to compare startup times.")
