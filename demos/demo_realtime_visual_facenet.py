"""
Real-Time Vision Pipeline Benchmark: RetinaFace (MTCNN) + FaceNet

This script is highly modified to benchmark the difference in FPS when using a
PyTorch-based RetinaFace + FaceNet stack compared to the original ONNX 
SCRFD + ArcFace stack.

It uses the Intel RealSense Camera input to eliminate webcam initialization delay.
"""

import cv2
import numpy as np
import sys
import threading
from pathlib import Path
from datetime import datetime
import psutil
from loguru import logger
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from vision.tracker import ByteTracker
from vision.head_pose import HeadPoseEstimator
from vision.recognition_trigger import RecognitionTrigger
from vision.database import EnrollmentDatabase

# ---------- Jetson GPU stats via sysfs ----------
_GPU_DEVFREQ = "/sys/class/devfreq/17000000.gpu"

def _read_gpu_stats() -> dict:
    """Read GPU freq & utilisation from Jetson sysfs. Returns defaults on failure."""
    stats: dict[str, float | int | str] = {"cur_mhz": 0, "max_mhz": 0, "util_pct": 0.0, "provider": "CPU"}
    try:
        with open(f"{_GPU_DEVFREQ}/cur_freq") as f:
            stats["cur_mhz"] = int(f.read().strip()) // 1_000_000
        with open(f"{_GPU_DEVFREQ}/max_freq") as f:
            stats["max_mhz"] = int(f.read().strip()) // 1_000_000
        
        cur_mhz = int(stats["cur_mhz"])
        max_mhz = int(stats["max_mhz"])
        if max_mhz > 0:
            stats["util_pct"] = cur_mhz / max_mhz * 100.0
    except Exception:
        pass
    return stats

class VisualPipelineDemo:
    """Real-time pipeline demo with visual feedback."""
    
    def __init__(self):
        """Initialize all pipeline components."""
        print("Initializing Vision Pipeline Demo (PyTorch MTCNN+FaceNet)...")
        print("  Camera: RealSense will initialize in run()...")

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"  PyTorch Target Device: {self.device}")

        # Initialize detector
        print("  Loading MTCNN (Face Detector)...")
        self.detector = MTCNN(keep_all=True, device=self.device)
        
        # Initialize tracker
        print("  Loading face tracker...")
        self.tracker = ByteTracker(
            track_thresh=0.5,
            track_buffer=30,
            match_thresh=0.4,
            min_box_area=100
        )
        
        # Initialize head pose estimator
        print("  Loading head pose estimator...")
        self.head_pose = HeadPoseEstimator(
            yaw_threshold=25,
            pitch_threshold=15,
            roll_threshold=30
        )
        
        # Initialize recognition trigger
        print("  Loading recognition trigger...")
        self.trigger = RecognitionTrigger(
            cooldown_seconds=5.0,
            require_attention=True
        )
        
        # Initialize face recognizer
        print("  Loading InceptionResnetV1 (FaceNet Recognizer)...")
        try:
            self.recognizer = InceptionResnetV1(pretrained='vggface2').eval().to(self.device)
            self.recognizer_available = True
        except Exception as e:
            print(f"    Warning: Face recognizer not available: {e}")
            self.recognizer_available = False
        
        # Stats
        self.frame_count: int = 0
        self.fps: float = 0.0
        self.last_time = datetime.now()
        self.last_resource_log_time = datetime.now()
        self._process = psutil.Process()
        self._gpu_provider = "CUDA" if torch.cuda.is_available() else "CPU"
        
        # Track confirmed recognitions
        self.confirmed_tracks: set = set()
        self.track_names: dict[int, str] = {}
        
        # Load enrollment database
        print("  Loading enrollment database...")
        self.db = EnrollmentDatabase("data/enrollments.json")
        print(f"    Loaded {len(self.db)} enrolled students")
        
        print("✓ Pipeline initialized successfully!\n")

    def draw_info_panel(self, frame, detections, tracks):
        """Draw information panel on the left side."""
        h, w = frame.shape[:2]
        panel_width = 300
        
        # Create semi-transparent panel
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (panel_width, h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # Title
        y = 30
        cv2.putText(frame, "VISION PIPELINE", (10, y), 
                   cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 255, 255), 2)
        
        # Stats
        y += 40
        cv2.putText(frame, f"FPS: {self.fps:.1f}", (10, y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        y += 25
        cv2.putText(frame, f"Frame: {self.frame_count}", (10, y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Pipeline steps
        y += 40
        cv2.putText(frame, "PIPELINE STEPS:", (10, y), 
                   cv2.FONT_HERSHEY_DUPLEX, 0.6, (0, 255, 255), 2)
        
        # Check if any track has triggered recognition or is confirmed
        has_confirmed = any(t.track_id in self.confirmed_tracks for t in tracks)
        has_triggered = any(getattr(t, 'should_recognize', False) for t in tracks)
        has_head_pose = any(bool(getattr(t, 'head_pose', None)) for t in tracks)
        is_looking = any(bool(getattr(t, 'head_pose', None)) and getattr(t, 'head_pose', {}).get('is_looking', False) for t in tracks)
        
        steps = [
            ("1. Detection", len(detections) > 0, (0, 255, 0) if len(detections) > 0 else (100, 100, 100)),
            ("2. Tracking", len(tracks) > 0, (0, 255, 0) if len(tracks) > 0 else (100, 100, 100)),
            ("3. Head Pose", has_head_pose, (0, 255, 0) if is_looking else (255, 100, 0) if has_head_pose else (100, 100, 100)),
            ("4. Trigger", has_triggered or has_confirmed, (0, 255, 0) if (has_triggered or has_confirmed) else (100, 100, 100)),
            ("5. Recognition", self.recognizer_available, (0, 255, 0) if self.recognizer_available else (0, 0, 255))
        ]
        
        for step, active, color in steps:
            y += 30
            status = "●" if active else "○"
            cv2.putText(frame, f"{status} {step}", (10, y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        # Stats
        y += 40
        cv2.putText(frame, "STATS:", (10, y), 
                   cv2.FONT_HERSHEY_DUPLEX, 0.6, (0, 255, 255), 2)
        y += 30
        cv2.putText(frame, f"Detections: {len(detections)}", (10, y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        y += 25
        cv2.putText(frame, f"Active Tracks: {len(tracks)}", (10, y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # CPU / RAM / GPU stats
        n_cores  = psutil.cpu_count() or 1
        sys_cpu  = psutil.cpu_percent(interval=None)
        proc_cpu_norm = min(self._process.cpu_percent(interval=None) / n_cores, 100.0)
        mem       = psutil.virtual_memory()
        swap      = psutil.swap_memory()
        ram_used  = mem.used  / (1024 ** 3)
        ram_total = mem.total / (1024 ** 3)
        swap_used = swap.used / (1024 ** 3)
        gpu       = _read_gpu_stats()

        # Log to terminal once per second
        now = datetime.now()
        if (now - self.last_resource_log_time).total_seconds() >= 1.0:
            logger.info(
                f"SYS_CPU={sys_cpu:>4.1f}% | PROC_CPU={proc_cpu_norm:>4.1f}% | "
                f"RAM={ram_used:>4.1f}/{ram_total:.1f}GB ({mem.percent}%) | "
                f"SWAP={swap_used:>4.1f}GB ({swap.percent}%) | "
                f"GPU={gpu['cur_mhz']}MHz/{gpu['max_mhz']}MHz ({gpu['util_pct']:.0f}%) [{self._gpu_provider}]"
            )
            self.last_resource_log_time = now

        y += 40
        cv2.putText(frame, "RESOURCES:", (10, y),
                   cv2.FONT_HERSHEY_DUPLEX, 0.6, (0, 255, 255), 2)
        y += 28
        cpu_color = (0, 255, 0) if sys_cpu < 70 else (0, 165, 255) if sys_cpu < 90 else (0, 0, 255)
        cv2.putText(frame, f"CPU sys : {sys_cpu:.0f}%  ({n_cores} cores)", (10, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.45, cpu_color, 1)
        y += 22
        cv2.putText(frame, f"CPU proc: {proc_cpu_norm:.0f}% rel", (10, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.45, cpu_color, 1)
        y += 22
        gpu_color = (0, 255, 0) if gpu['util_pct'] < 70 else (0, 165, 255) if gpu['util_pct'] < 90 else (0, 0, 255)
        cv2.putText(frame, f"GPU freq: {gpu['cur_mhz']}/{gpu['max_mhz']}MHz ({gpu['util_pct']:.0f}%)", (10, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.45, gpu_color, 1)
        y += 22
        cv2.putText(frame, f"GPU mode: {self._gpu_provider}", (10, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.45, gpu_color, 1)
        y += 22
        ram_color = (0, 255, 0) if mem.percent < 70 else (0, 165, 255) if mem.percent < 90 else (0, 0, 255)
        # Note: Jetson unified mem — GPU carve-out not visible here
        cv2.putText(frame, f"RAM : {ram_used:.1f}/{ram_total:.1f}GB ({mem.percent:.0f}%)", (10, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.45, ram_color, 1)
        y += 22
        swap_color = (0, 255, 0) if swap.percent < 50 else (0, 165, 255) if swap.percent < 80 else (0, 0, 255)
        cv2.putText(frame, f"Swap: {swap_used:.1f}GB ({swap.percent:.0f}%)", (10, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.45, swap_color, 1)

        # Controls
        y = h - 80
        cv2.putText(frame, "CONTROLS:", (10, y),
                   cv2.FONT_HERSHEY_DUPLEX, 0.6, (0, 255, 255), 2)
        y += 25
        cv2.putText(frame, "Q - Quit", (10, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        y += 20
        cv2.putText(frame, "S - Screenshot", (10, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        y += 20
        cv2.putText(frame, "R - Reset", (10, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    
    def draw_detection(self, frame, detection, track=None):
        """Draw detection with all information."""
        bbox = detection["bbox"]
        conf = detection["confidence"]
        landmarks = detection.get("landmarks", [])
        
        x1, y1, x2, y2 = map(int, bbox)
        
        # Draw bounding box (green for detection)
        color = (0, 255, 0)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        
        # Draw landmarks
        if landmarks:
            for i, (lx, ly) in enumerate(landmarks):
                cv2.circle(frame, (int(lx), int(ly)), 3, (0, 0, 255), -1)
        
        # Track info
        if track:
            # Check if we have a persistent name for this track
            student_name = self.track_names.get(track.track_id)
            
            # Debug: Print track_names dictionary periodically
            if self.frame_count % 30 == 0 and self.track_names:
                print(f"DEBUG: track_names = {self.track_names}, current track_id = {track.track_id}")
            
            # Display student name if recognized (from persistent storage)
            if student_name:
                name_label = f"{student_name}"
                # Move name higher up to avoid overlapping
                cv2.putText(frame, name_label, (x1, y1 - 70), 
                           cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 255, 255), 2)
                # Track ID below name
                id_label = f"ID:{track.track_id}"
                cv2.putText(frame, id_label, (x1, y1 - 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
            else:
                # Track ID only (no name yet)
                label = f"ID:{track.track_id}"
                cv2.putText(frame, label, (x1, y1 - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # Head pose
            head_pose = getattr(track, 'head_pose', None)
            if head_pose:
                pose = head_pose
                y_offset = y1 - 30
                
                # Pose angles
                pose_text = f"Y:{pose['yaw']:.0f} P:{pose['pitch']:.0f} R:{pose['roll']:.0f}"
                cv2.putText(frame, pose_text, (x1, y_offset), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
                
                # Looking indicator
                looking_color = (0, 255, 0) if pose['is_looking'] else (0, 0, 255)
                looking_text = "LOOKING" if pose['is_looking'] else "NOT LOOKING"
                cv2.putText(frame, looking_text, (x1, y_offset - 15), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, looking_color, 1)
            
            # Recognition trigger status
            should_recognize = getattr(track, 'should_recognize', None)
            if should_recognize is not None:
                # Check if track is confirmed
                if track.track_id in self.confirmed_tracks:
                    trigger_color = (0, 255, 0)
                    trigger_text = "CONFIRMED"
                elif should_recognize:
                    trigger_color = (0, 255, 0)
                    trigger_text = "RECOGNIZE!"
                else:
                    trigger_color = (100, 100, 100)
                    # Show reason if available
                    trigger_text = getattr(track, 'trigger_reason', 'Waiting...')
                cv2.putText(frame, trigger_text, (x1, y2 + 20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, trigger_color, 1)
            
            # Embedding status
            has_embedding = getattr(track, 'has_embedding', False)
            if has_embedding:
                cv2.putText(frame, "Embedding OK", (x1, y2 + 40), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
        else:
            # Just confidence
            cv2.putText(frame, f"{conf:.2f}", (x1, y1 - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    def process_frame(self, frame):
        """Process a single frame through the pipeline."""
        
        # Step 1: Detect faces with MTCNN
        # Convert BGR (OpenCV) to RGB (PyTorch MTCNN expects RGB)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        boxes, probs, points = self.detector.detect(rgb_frame, landmarks=True)
        
        detections = []
        if boxes is not None:
            for box, prob, land in zip(boxes, probs, points):
                if prob < 0.5:
                    continue
                detections.append({
                    "bbox": box.tolist(),
                    "confidence": float(prob),
                    "landmarks": land.tolist() if getattr(land, 'tolist', None) else []
                })

        # Step 2: Track faces
        tracks = self.tracker.update(detections)
        
        # Helper function to calculate IOU
        def calc_iou(bbox1, bbox2):
            x1 = max(bbox1[0], bbox2[0])
            y1 = max(bbox1[1], bbox2[1])
            x2 = min(bbox1[2], bbox2[2])
            y2 = min(bbox1[3], bbox2[3])
            
            inter_area = max(0, x2 - x1) * max(0, y2 - y1)
            
            bbox1_area = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
            bbox2_area = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
            
            union_area = bbox1_area + bbox2_area - inter_area
            
            if union_area == 0:
                return 0.0
            
            return inter_area / union_area
        
        # Step 3-5: Process each track
        for track in tracks:
            # Find matching detection using IOU
            best_iou = 0
            best_det = None
            for d in detections:
                iou = calc_iou(getattr(track, 'bbox', [0,0,0,0]), d["bbox"])
                if iou > best_iou:
                    best_iou = iou
                    best_det = d
            
            if not best_det or best_iou < 0.3:  # Require at least 30% IOU
                continue
            
            det = best_det if isinstance(best_det, dict) else {}
            
            # Step 3: Estimate head pose
            landmarks = det.get("landmarks")
            if landmarks is not None:
                pose = self.head_pose.estimate_simple(landmarks)
                setattr(track, 'head_pose', pose)
            
            # Step 4: Check recognition trigger
            track_head_pose = getattr(track, 'head_pose', {}) or {}
            is_looking = track_head_pose.get('is_looking', False)
            decision = self.trigger.should_trigger(
                track_id=getattr(track, 'track_id', 0),
                confidence=getattr(track, 'confidence', 0.0),
                track_age=getattr(track, 'age', 0),
                is_looking=is_looking
            )
            setattr(track, 'should_recognize', decision.should_trigger)
            setattr(track, 'trigger_reason', decision.reason)  # Store reason for display
            
            # Step 5: Extract embedding if triggered
            if getattr(track, 'should_recognize', False) and self.recognizer_available:
                # Skip if already recognized
                track_id = getattr(track, 'track_id', -1)
                if track_id in self.track_names:
                    continue
                    
                try:
                    # FaceNet (InceptionResnetV1) processing 
                    # 1. Expand bbox by a margin (FaceNet standard is a 160x160 crop with margin)
                    b = getattr(track, 'bbox', [0,0,0,0])
                    x1, y1, x2, y2 = map(int, b)
                    
                    margin = 20
                    h_m = max(0, y1 - margin)
                    h_M = min(frame.shape[0], y2 + margin)
                    w_m = max(0, x1 - margin)
                    w_M = min(frame.shape[1], x2 + margin)
                    
                    face_crop = frame[h_m:h_M, w_m:w_M]
                    
                    # 2. Resize to 160x160
                    face_resized = cv2.resize(face_crop, (160, 160))
                    
                    # 3. Convert to tensor and normalize exactly as InceptionResnetV1 expects
                    face_tensor = torch.tensor(face_resized).permute(2, 0, 1).float()
                    face_tensor = (face_tensor - 127.5) / 128.0
                    face_tensor = face_tensor.unsqueeze(0).to(self.device)
                    
                    # 4. Extract embedding
                    recognizer_fn = getattr(self, 'recognizer', None)
                    if recognizer_fn is not None and callable(recognizer_fn):
                        with torch.no_grad():
                            tensor_out = recognizer_fn(face_tensor)
                            embedding = tensor_out.cpu().numpy()[0] if hasattr(tensor_out, 'cpu') else None
                    else:
                        embedding = None
                    
                    setattr(track, 'has_embedding', True)
                    
                    # Recognize student from database
                    result = self.db.recognize(embedding, threshold=0.4)
                    if result:
                        # Result is a tuple: (student_id, similarity, name)
                        student_id, similarity, student_name = result
                        
                        # Store in persistent dictionary
                        self.track_names[track.track_id] = student_name
                        print(f"  → Recognized: {student_name} (ID: {student_id}, similarity: {similarity:.3f})")
                        print(f"  → Stored in track_names[{track.track_id}] = '{student_name}'")
                    else:
                        # Store "Unknown" in persistent dictionary
                        self.track_names[track.track_id] = "Unknown"
                        print(f"  → Unknown person (no match above threshold)")
                    
                    # Mark this track as confirmed (persists for LLM interaction)
                    self.confirmed_tracks.add(track_id)
                except Exception as e:
                    print(f"  → Failed to extract embedding: {e}")
                    setattr(track, 'has_embedding', False)
            
            # Draw detection with all info
            self.draw_detection(frame, det, track)
        
        # Draw info panel
        self.draw_info_panel(frame, detections, tracks)
        
        return frame
    
    def run(self):
        """Run the demo."""
        print("Starting Intel RealSense camera...")
        print("Press 'q' to quit, 's' to save screenshot, 'r' to reset\n")
        
        import pyrealsense2 as rs
        
        pipeline = rs.pipeline()
        config = rs.config()

        pipeline_wrapper = rs.pipeline_wrapper(pipeline)
        try:
             pipeline_profile = config.resolve(pipeline_wrapper)
             device = pipeline_profile.get_device()
        except Exception as e:
             print(f"❌ Failed to find RealSense camera: {e}")
             return

        # Enable color stream
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        
        try:
             # Start streaming
             pipeline.start(config)
        except Exception as e:
             print(f"❌ Failed to start RealSense pipeline: {e}")
             return

        print("  Camera ready (640×480).")
        
        cv2.namedWindow("Vision Pipeline Demo", cv2.WINDOW_NORMAL)
        
        while True:
             try:
                 # Wait for a coherent pair of frames: depth and color
                 frames = pipeline.wait_for_frames()
                 color_frame = frames.get_color_frame()
                 
                 if not color_frame:
                     print("Failed to grab color frame")
                     continue
                 
                 # Convert images to numpy arrays
                 frame = np.asanyarray(color_frame.get_data())
             except Exception as e:
                 print(f"Failed to read frame: {e}")
                 break
             
             # Update FPS
             current_time = datetime.now()
             elapsed = (current_time - self.last_time).total_seconds()
             if elapsed > 0:
                 self.fps = 1.0 / elapsed
             self.last_time = current_time
             self.frame_count += 1
             
             # Process frame
             frame = self.process_frame(frame)
             
             # Display
             cv2.imshow("Vision Pipeline Demo", frame)
             
             if self.frame_count >= 100:
                 print(f"BENCHMARK COMPLETE:")
                 print(f"  Average FPS (last 100 frames): {self.fps:.1f} fps")
                 break
                 
             # Handle keys
             key = cv2.waitKey(1) & 0xFF
             
             if key == ord('q'):
                 print("\nQuitting...")
                 break
             elif key == ord('s'):
                 filename = f"screenshot_facenet_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
                 cv2.imwrite(filename, frame)
                 print(f"  Screenshot saved: {filename}")
             elif key == ord('r'):
                 print("  Resetting tracker...")
                 self.tracker.reset()
        
        pipeline.stop()
        cv2.destroyAllWindows()
        print("Demo ended.")


def main():
    """Main entry point."""
    print("=" * 60)
    print("REAL-TIME VISION PIPELINE DEMO: PYTORCH FACENET")
    print("=" * 60)
    print()
    
    try:
        demo = VisualPipelineDemo()
        demo.run()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    except Exception as e:
        print(f"\nFAILED Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
