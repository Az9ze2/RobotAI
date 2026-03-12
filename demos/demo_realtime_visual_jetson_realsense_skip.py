"""
Real-Time Vision Pipeline  *** JETSON PRODUCTION VERSION ***
============================================================
Headless-ready, production-grade script designed to run continuously
on a Jetson Orin Nano (JetPack 6) connected to an Intel RealSense camera.

Key improvements over the demo version
---------------------------------------
* CLI argument parsing (--no-display, --stream-enabled, --stream-host, etc.)
* Graceful shutdown on SIGINT / SIGTERM (Ctrl-C or systemd stop)
* No hard 100-frame benchmark stop – runs until killed
* Structured JSON logging via loguru instead of bare print() calls
* Optional ZMQ LAN streaming to a downstream device (e.g. Raspberry Pi 5)
  connected over a network switch

Usage
-----
  # Headless with LAN streaming:
  python demos/demo_realtime_visual_jetson_realsense_skip.py \\
      --no-display --stream-enabled --stream-host 192.168.1.100 --stream-port 5555

  # Interactive with display (debug):
  python demos/demo_realtime_visual_jetson_realsense_skip.py

Controls (when display is active)
----------------------------------
  q  – Quit          s  – Save screenshot          r  – Reset tracker

Pipeline steps
--------------
  1. Face Detection   – SCRFD (ONNX / TensorRT)
  2. Face Tracking    – ByteTracker, frame-skip with CSRT/MIL CV trackers
  3. Head Pose        – Yaw / Pitch / Roll estimation from landmarks
  4. Recognition Trigger – cooldown-based gate
  5. Face Recognition – ArcFace embedding → database lookup
  6. LAN Stream       – ZMQ PUB (optional)
"""

from __future__ import annotations

import argparse
import signal
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import psutil
from loguru import logger

# ── Add src to path ──────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from vision.detector_factory import create_scrfd_detector
from vision.tracker import ByteTracker
from vision.head_pose import HeadPoseEstimator
from vision.recognition_trigger import RecognitionTrigger
from vision.recognizer import FaceRecognizer
from vision.database import EnrollmentDatabase
from vision.stream_sender import StreamSender, build_result

# ── Jetson GPU stats via sysfs ────────────────────────────────────────────────
_GPU_DEVFREQ = "/sys/class/devfreq/17000000.gpu"


def _read_gpu_stats() -> dict:
    """Read GPU freq & utilisation from Jetson sysfs. Returns defaults on failure."""
    stats: dict = {"cur_mhz": 0, "max_mhz": 0, "util_pct": 0.0}
    try:
        with open(f"{_GPU_DEVFREQ}/cur_freq") as f:
            stats["cur_mhz"] = int(f.read().strip()) // 1_000_000
        with open(f"{_GPU_DEVFREQ}/max_freq") as f:
            stats["max_mhz"] = int(f.read().strip()) // 1_000_000
        cur, mx = int(stats["cur_mhz"]), int(stats["max_mhz"])
        if mx > 0:
            stats["util_pct"] = cur / mx * 100.0
    except Exception:
        pass
    return stats


# ─────────────────────────────────────────────────────────────────────────────
# Argument parser
# ─────────────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="RobotAI Jetson Vision Pipeline – production runner",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--no-display", action="store_true",
                   help="Headless mode – skip cv2.imshow (recommended for systemd deployment)")
    p.add_argument("--skip-frames", type=int, default=10, metavar="N",
                   help="Run SCRFD detector every N frames; use CSRT tracker in between")
    p.add_argument("--stream-enabled", action="store_true",
                   help="Enable ZMQ LAN streaming to a downstream device")
    p.add_argument("--stream-host", default="0.0.0.0", metavar="IP",
                   help="Interface to bind the ZMQ PUB socket on (Jetson side)")
    p.add_argument("--stream-port", type=int, default=5555, metavar="PORT",
                   help="TCP port for the ZMQ PUB socket")
    p.add_argument("--jpeg-quality", type=int, default=60,
                   help="JPEG quality used when streaming (0-100)")
    p.add_argument("--stream-width", type=int, default=320,
                   help="Resize width of streamed frame (smaller = less bandwidth)")
    p.add_argument("--stream-height", type=int, default=240,
                   help="Resize height of streamed frame")
    p.add_argument("--det-model", default="models/buffalo_l/det_10g.onnx",
                   help="Path to SCRFD ONNX detection model")
    p.add_argument("--rec-model", default="models/arcface_r100_v1_fp16.onnx",
                   help="Path to ArcFace ONNX recognition model")
    p.add_argument("--db-path", default="data/enrollments.json",
                   help="Path to enrollment JSON database")
    p.add_argument("--log-level", default="INFO",
                   choices=["TRACE", "DEBUG", "INFO", "WARNING", "ERROR"],
                   help="Loguru log level")
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Main pipeline class
# ─────────────────────────────────────────────────────────────────────────────

class JetsonVisionPipeline:
    """Production-grade real-time face recognition pipeline for Jetson."""

    def __init__(self, args: argparse.Namespace):
        self._args = args
        self._running = False

        logger.info("Initialising Jetson Vision Pipeline…")

        # ── Detector ─────────────────────────────────────────────────────────
        logger.info("  Loading face detector (SCRFD)…")
        self.detector = create_scrfd_detector(
            model_path=args.det_model,
            confidence_threshold=0.5,
            nms_threshold=0.4,
            input_size=(640, 640),
            device="cuda",
            use_tensorrt=True,
        )

        # ── Tracker ──────────────────────────────────────────────────────────
        logger.info("  Loading face tracker (ByteTracker)…")
        self.tracker = ByteTracker(
            track_thresh=0.5,
            track_buffer=30,
            match_thresh=0.4,
            min_box_area=100,
        )

        # ── Head Pose ────────────────────────────────────────────────────────
        logger.info("  Loading head pose estimator…")
        self.head_pose = HeadPoseEstimator(
            yaw_threshold=25,
            pitch_threshold=15,
            roll_threshold=30,
        )

        # ── Recognition Trigger ──────────────────────────────────────────────
        logger.info("  Loading recognition trigger…")
        self.trigger = RecognitionTrigger(
            cooldown_seconds=5.0,
            require_attention=True,
        )

        # ── Face Recognizer ──────────────────────────────────────────────────
        logger.info("  Loading face recognizer (ArcFace)…")
        try:
            self.recognizer = FaceRecognizer(
                model_path=args.rec_model,
                device="cuda",
                use_tensorrt=True,
            )
            self.recognizer_available = True
        except Exception as exc:
            logger.warning(f"  Face recognizer unavailable: {exc}")
            self.recognizer_available = False

        # ── Enrollment Database ──────────────────────────────────────────────
        logger.info("  Loading enrollment database…")
        self.db = EnrollmentDatabase(args.db_path)
        logger.info(f"  {len(self.db)} enrolled students loaded.")

        # ── LAN Streamer (optional) ──────────────────────────────────────────
        self.sender: StreamSender | None = None
        if args.stream_enabled:
            self.sender = StreamSender(
                host=args.stream_host,
                port=args.stream_port,
                jpeg_quality=args.jpeg_quality,
                resize=(args.stream_width, args.stream_height),
            )

        # ── Runtime state ────────────────────────────────────────────────────
        self.frame_count: int = 0
        self.fps: float = 0.0
        self._last_time = time.monotonic()
        self._last_resource_log = time.monotonic()
        self._process = psutil.Process()
        self.skip_frames: int = args.skip_frames
        self.cv_trackers: list = []
        self.confirmed_tracks: set = set()
        self.track_names: dict[int, str] = {}

        # Cache ONNX provider string for display
        try:
            prov = self.detector.session.get_providers()[0]
            self._gpu_provider = prov.replace("ExecutionProvider", "")
        except Exception:
            self._gpu_provider = "unknown"

        logger.success("Pipeline initialised successfully.")

    # ── Signal handling ───────────────────────────────────────────────────────

    def _install_signal_handlers(self):
        """Register SIGINT and SIGTERM for clean shutdown."""
        for sig in (signal.SIGINT, signal.SIGTERM):
            signal.signal(sig, self._handle_signal)

    def _handle_signal(self, signum, _frame):
        logger.info(f"Signal {signum} received – shutting down…")
        self._running = False

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _calc_iou(b1, b2) -> float:
        x1, y1 = max(b1[0], b2[0]), max(b1[1], b2[1])
        x2, y2 = min(b1[2], b2[2]), min(b1[3], b2[3])
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        a1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
        a2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
        union = a1 + a2 - inter
        return inter / union if union > 0 else 0.0

    # ── Draw helpers ──────────────────────────────────────────────────────────

    def _draw_info_panel(self, frame, detections, tracks):
        h, w = frame.shape[:2]
        panel_w = 300
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (panel_w, h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

        y = 30
        cv2.putText(frame, "VISION PIPELINE", (10, y),
                    cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 255, 255), 2)
        y += 40
        cv2.putText(frame, f"FPS: {self.fps:.1f}", (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        y += 25
        cv2.putText(frame, f"Frame: {self.frame_count}", (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        has_confirmed = any(t.track_id in self.confirmed_tracks for t in tracks)
        has_triggered = any(getattr(t, "should_recognize", False) for t in tracks)
        has_head_pose = any(bool(getattr(t, "head_pose", None)) for t in tracks)
        is_looking = any(
            bool(getattr(t, "head_pose", None)) and
            getattr(t, "head_pose", {}).get("is_looking", False)
            for t in tracks
        )

        y += 40
        cv2.putText(frame, "PIPELINE STEPS:", (10, y),
                    cv2.FONT_HERSHEY_DUPLEX, 0.6, (0, 255, 255), 2)
        steps = [
            ("1. Detection",    len(detections) > 0,               (0, 255, 0) if len(detections) > 0 else (100, 100, 100)),
            ("2. Tracking",     len(tracks) > 0,                   (0, 255, 0) if len(tracks) > 0 else (100, 100, 100)),
            ("3. Head Pose",    has_head_pose,                      (0, 255, 0) if is_looking else (255, 100, 0) if has_head_pose else (100, 100, 100)),
            ("4. Trigger",      has_triggered or has_confirmed,     (0, 255, 0) if (has_triggered or has_confirmed) else (100, 100, 100)),
            ("5. Recognition",  self.recognizer_available,          (0, 255, 0) if self.recognizer_available else (0, 0, 255)),
            ("6. LAN Stream",   self.sender is not None,            (0, 255, 255) if self.sender else (100, 100, 100)),
        ]
        for step, active, color in steps:
            y += 30
            cv2.putText(frame, f"{'●' if active else '○'} {step}", (10, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        y += 40
        cv2.putText(frame, "STATS:", (10, y),
                    cv2.FONT_HERSHEY_DUPLEX, 0.6, (0, 255, 255), 2)
        y += 30
        cv2.putText(frame, f"Detections: {len(detections)}", (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        y += 25
        cv2.putText(frame, f"Active Tracks: {len(tracks)}", (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Resource stats
        n_cores = psutil.cpu_count() or 1
        sys_cpu = psutil.cpu_percent(interval=None)
        proc_cpu = min(self._process.cpu_percent(interval=None) / n_cores, 100.0)
        mem = psutil.virtual_memory()
        swap = psutil.swap_memory()
        ram_gb = mem.used / (1024 ** 3)
        ram_total_gb = mem.total / (1024 ** 3)
        swap_gb = swap.used / (1024 ** 3)
        gpu = _read_gpu_stats()

        now = time.monotonic()
        if now - self._last_resource_log >= 1.0:
            logger.info(
                f"SYS_CPU={sys_cpu:>4.1f}% | PROC_CPU={proc_cpu:>4.1f}% | "
                f"RAM={ram_gb:.1f}/{ram_total_gb:.1f}GB ({mem.percent}%) | "
                f"SWAP={swap_gb:.1f}GB ({swap.percent}%) | "
                f"GPU={gpu['cur_mhz']}MHz/{gpu['max_mhz']}MHz ({gpu['util_pct']:.0f}%) [{self._gpu_provider}] | "
                f"FPS={self.fps:.1f}"
            )
            self._last_resource_log = now

        y += 40
        cv2.putText(frame, "RESOURCES:", (10, y),
                    cv2.FONT_HERSHEY_DUPLEX, 0.6, (0, 255, 255), 2)
        cpu_color = (0, 255, 0) if sys_cpu < 70 else (0, 165, 255) if sys_cpu < 90 else (0, 0, 255)
        gpu_color = (0, 255, 0) if gpu["util_pct"] < 70 else (0, 165, 255) if gpu["util_pct"] < 90 else (0, 0, 255)
        ram_color = (0, 255, 0) if mem.percent < 70 else (0, 165, 255) if mem.percent < 90 else (0, 0, 255)
        for text, color in [
            (f"CPU sys : {sys_cpu:.0f}%  ({n_cores} cores)", cpu_color),
            (f"CPU proc: {proc_cpu:.0f}% rel", cpu_color),
            (f"GPU freq: {gpu['cur_mhz']}/{gpu['max_mhz']}MHz ({gpu['util_pct']:.0f}%)", gpu_color),
            (f"GPU mode: {self._gpu_provider}", gpu_color),
            (f"RAM : {ram_gb:.1f}/{ram_total_gb:.1f}GB ({mem.percent:.0f}%)", ram_color),
            (f"Swap: {swap_gb:.1f}GB ({swap.percent:.0f}%)", (0, 255, 0) if swap.percent < 50 else (0, 165, 255)),
        ]:
            y += 22
            cv2.putText(frame, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

        if self.sender:
            y += 28
            cv2.putText(frame, f"STREAM → {self._args.stream_host}:{self._args.stream_port}", (10, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)

        if not self._args.no_display:
            y = frame.shape[0] - 80
            cv2.putText(frame, "CONTROLS:", (10, y),
                        cv2.FONT_HERSHEY_DUPLEX, 0.6, (0, 255, 255), 2)
            for label in ["Q - Quit", "S - Screenshot", "R - Reset"]:
                y += 20
                cv2.putText(frame, label, (10, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    def _draw_detection(self, frame, detection, track=None):
        bbox = detection["bbox"]
        conf = detection["confidence"]
        landmarks = detection.get("landmarks", [])
        x1, y1, x2, y2 = map(int, bbox)

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        if landmarks:
            for lx, ly in landmarks:
                cv2.circle(frame, (int(lx), int(ly)), 3, (0, 0, 255), -1)

        if track:
            student_name = self.track_names.get(track.track_id)
            if student_name:
                cv2.putText(frame, student_name, (x1, y1 - 70),
                            cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 255, 255), 2)
                cv2.putText(frame, f"ID:{track.track_id}", (x1, y1 - 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
            else:
                cv2.putText(frame, f"ID:{track.track_id}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            head_pose = getattr(track, "head_pose", None)
            if head_pose:
                pose_text = f"Y:{head_pose['yaw']:.0f} P:{head_pose['pitch']:.0f} R:{head_pose['roll']:.0f}"
                cv2.putText(frame, pose_text, (x1, y1 - 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
                looking_color = (0, 255, 0) if head_pose["is_looking"] else (0, 0, 255)
                cv2.putText(frame, "LOOKING" if head_pose["is_looking"] else "NOT LOOKING",
                            (x1, y1 - 45), cv2.FONT_HERSHEY_SIMPLEX, 0.4, looking_color, 1)

            if track.track_id in self.confirmed_tracks:
                cv2.putText(frame, "CONFIRMED", (x1, y2 + 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
            elif getattr(track, "should_recognize", False):
                cv2.putText(frame, "RECOGNIZE!", (x1, y2 + 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

            if getattr(track, "has_embedding", False):
                cv2.putText(frame, "Embedding OK", (x1, y2 + 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
        else:
            cv2.putText(frame, f"{conf:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # ── Frame processing ──────────────────────────────────────────────────────

    def _process_frame(self, frame: np.ndarray) -> tuple[np.ndarray, list]:
        """Run the full pipeline on one frame. Returns (annotated_frame, tracks)."""

        # Step 1 – Detection / tracking (frame skip strategy)
        if self.frame_count % self.skip_frames == 0:
            detections = self.detector.detect(frame)
            self.cv_trackers = []
            for det in detections:
                try:
                    if hasattr(cv2, "legacy") and hasattr(cv2.legacy, "TrackerCSRT_create"):
                        tracker = cv2.legacy.TrackerCSRT_create()
                    elif hasattr(cv2, "TrackerCSRT_create"):
                        tracker = cv2.TrackerCSRT_create()
                    else:
                        tracker = cv2.TrackerMIL_create()
                except Exception:
                    tracker = cv2.TrackerMIL_create()
                x1, y1, x2, y2 = det["bbox"]
                w, h = max(1, int(x2 - x1)), max(1, int(y2 - y1))
                tracker.init(frame, (int(x1), int(y1), w, h))
                self.cv_trackers.append({
                    "tracker": tracker,
                    "confidence": det["confidence"],
                    "landmarks": det.get("landmarks", []),
                })
        else:
            detections = []
            valid = []
            for t_info in self.cv_trackers:
                ok, bbox = t_info["tracker"].update(frame)
                if ok:
                    x, y, w, h = [int(v) for v in bbox]
                    detections.append({
                        "bbox": [x, y, x + w, y + h],
                        "confidence": t_info["confidence"],
                        "landmarks": t_info["landmarks"],
                    })
                    valid.append(t_info)
            self.cv_trackers = valid

        # Step 2 – ByteTracker
        tracks = self.tracker.update(detections)

        # Steps 3-5 – per-track processing
        for track in tracks:
            best_iou, best_det = 0.0, None
            for d in detections:
                iou = self._calc_iou(getattr(track, "bbox", [0, 0, 0, 0]), d["bbox"])
                if iou > best_iou:
                    best_iou, best_det = iou, d
            if not best_det or best_iou < 0.3:
                continue
            det = best_det

            # Step 3 – Head pose
            lm = det.get("landmarks")
            if lm is not None:
                setattr(track, "head_pose", self.head_pose.estimate_simple(lm))

            # Step 4 – Recognition trigger
            hp = getattr(track, "head_pose", {}) or {}
            decision = self.trigger.should_trigger(
                track_id=getattr(track, "track_id", 0),
                confidence=getattr(track, "confidence", 0.0),
                track_age=getattr(track, "age", 0),
                is_looking=hp.get("is_looking", False),
            )
            setattr(track, "should_recognize", decision.should_trigger)
            setattr(track, "trigger_reason", decision.reason)

            # Step 5 – Embedding + recognition
            tid = getattr(track, "track_id", -1)
            if (getattr(track, "should_recognize", False)
                    and self.recognizer_available
                    and tid not in self.track_names):
                try:
                    embedding = self.recognizer.extract_embedding(
                        frame, getattr(track, "bbox", [0, 0, 0, 0]),
                        det.get("landmarks"),
                    )
                    setattr(track, "has_embedding", True)
                    result = self.db.recognize(embedding, threshold=0.4)
                    if result:
                        s_id, similarity, name = result
                        self.track_names[tid] = name
                        logger.info(f"Recognised: {name} (ID={s_id}, sim={similarity:.3f})")
                    else:
                        self.track_names[tid] = "Unknown"
                        logger.info("Recognition: no match above threshold")
                    self.confirmed_tracks.add(tid)
                except Exception as exc:
                    logger.error(f"Embedding extraction failed: {exc}")
                    setattr(track, "has_embedding", False)

            self._draw_detection(frame, det, track)

        self._draw_info_panel(frame, detections, tracks)
        return frame, tracks

    # ── Main loop ─────────────────────────────────────────────────────────────

    def run(self):
        """Initialise camera and enter the main processing loop."""
        self._install_signal_handlers()
        self._running = True

        import pyrealsense2 as rs

        logger.info("Starting Intel RealSense camera…")
        pipeline = rs.pipeline()
        config = rs.config()
        wrapper = rs.pipeline_wrapper(pipeline)
        try:
            config.resolve(wrapper)
        except Exception as exc:
            logger.error(f"No RealSense camera found: {exc}")
            return

        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        try:
            pipeline.start(config)
        except Exception as exc:
            logger.error(f"Failed to start RealSense pipeline: {exc}")
            return

        logger.info("Camera ready (640×480).")

        # Start LAN stream
        if self.sender is not None:
            self.sender.start()
            logger.info(
                f"ZMQ stream active → bind {self._args.stream_host}:{self._args.stream_port}"
            )

        if not self._args.no_display:
            cv2.namedWindow("Vision Pipeline – Jetson", cv2.WINDOW_NORMAL)

        try:
            while self._running:
                # --- Capture ---
                try:
                    frames = pipeline.wait_for_frames()
                    color_frame = frames.get_color_frame()
                    if not color_frame:
                        logger.warning("Empty color frame – skipping.")
                        continue
                    frame = np.asanyarray(color_frame.get_data())
                except Exception as exc:
                    logger.error(f"Frame capture error: {exc}")
                    break

                # --- FPS ---
                now = time.monotonic()
                elapsed = now - self._last_time
                if elapsed > 0:
                    self.fps = 1.0 / elapsed
                self._last_time = now
                self.frame_count += 1

                # --- Pipeline ---
                frame, tracks = self._process_frame(frame)

                # --- LAN stream ---
                if self.sender is not None:
                    payload = build_result(
                        tracks=tracks,
                        fps=self.fps,
                        frame_id=self.frame_count,
                        confirmed_tracks=self.confirmed_tracks,
                        track_names=self.track_names,
                    )
                    self.sender.send(frame, payload)

                # --- Display (optional) ---
                if not self._args.no_display:
                    cv2.imshow("Vision Pipeline – Jetson", frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord("q"):
                        logger.info("'q' pressed – quitting.")
                        break
                    elif key == ord("s"):
                        fname = f"screenshot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
                        cv2.imwrite(fname, frame)
                        logger.info(f"Screenshot saved: {fname}")
                    elif key == ord("r"):
                        logger.info("Resetting tracker.")
                        self.tracker.reset()

        finally:
            self._running = False
            pipeline.stop()
            if self.sender is not None:
                self.sender.stop()
            if not self._args.no_display:
                cv2.destroyAllWindows()
            logger.info("Pipeline shut down cleanly.")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main() -> int:
    args = _parse_args()

    # Configure loguru
    logger.remove()
    logger.add(
        sys.stderr,
        level=args.log_level,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | {message}",
        colorize=True,
    )

    logger.info("=" * 60)
    logger.info("RobotAI  –  Jetson Vision Pipeline  (production)")
    logger.info("=" * 60)
    logger.info(f"Display: {'OFF (headless)' if args.no_display else 'ON'}")
    logger.info(f"Frame skip: every {args.skip_frames} frames")
    logger.info(f"LAN stream: {'ENABLED → ' + args.stream_host + ':' + str(args.stream_port) if args.stream_enabled else 'DISABLED'}")

    try:
        pipeline = JetsonVisionPipeline(args)
        pipeline.run()
    except KeyboardInterrupt:
        logger.info("Interrupted by user.")
    except Exception as exc:
        logger.exception(f"Fatal error: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
