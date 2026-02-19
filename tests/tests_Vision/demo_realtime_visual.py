"""
Real-Time Vision Pipeline Demo with Visual Feedback

This script shows a live webcam feed with visual overlays for each pipeline step:
1. Face Detection - Green bounding boxes
2. Face Tracking - Track IDs and age
3. Head Pose - Yaw/Pitch/Roll angles
4. Recognition Trigger - Status indicators
5. Face Recognition - Embedding extraction

Press 'q' to quit
Press 's' to save screenshot
Press 'r' to reset tracker
"""

import cv2
import numpy as np
import sys
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from vision.detector_factory import create_scrfd_detector
from vision.tracker import ByteTracker
from vision.head_pose import HeadPoseEstimator
from vision.recognition_trigger import RecognitionTrigger
from vision.recognizer import FaceRecognizer

class VisualPipelineDemo:
    """Real-time pipeline demo with visual feedback."""
    
    def __init__(self):
        """Initialize all pipeline components."""
        print("Initializing Vision Pipeline Demo...")
        
        # Initialize detector
        print("  Loading face detector...")
        self.detector = create_scrfd_detector(
            model_path="models/scrfd_2.5g_kps.pth",
            confidence_threshold=0.5,
            nms_threshold=0.4,
            input_size=(640, 640),
            device="cpu"
        )
        
        # Initialize tracker
        print("  Loading face tracker...")
        self.tracker = ByteTracker(
            track_thresh=0.5,
            track_buffer=30,
            match_thresh=0.7,
            min_box_area=100
        )
        
        # Initialize head pose estimator
        print("  Loading head pose estimator...")
        self.head_pose = HeadPoseEstimator(
            yaw_threshold=30,
            pitch_threshold=30,
            roll_threshold=30
        )
        
        # Initialize recognition trigger
        print("  Loading recognition trigger...")
        self.trigger = RecognitionTrigger(
            cooldown_seconds=5.0,
            require_attention=True,
            min_track_age=3
        )
        
        # Initialize face recognizer
        print("  Loading face recognizer...")
        try:
            self.recognizer = FaceRecognizer(
                model_path="models/arcface_r100_v1_fp16.onnx",
                device="cpu"
            )
            self.recognizer_available = True
        except Exception as e:
            print(f"    Warning: Face recognizer not available: {e}")
            self.recognizer_available = False
        
        # Stats
        self.frame_count = 0
        self.fps = 0
        self.last_time = datetime.now()
        
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
                   cv2.FONT_HERSHEY_BOLD, 0.7, (0, 255, 255), 2)
        
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
                   cv2.FONT_HERSHEY_BOLD, 0.6, (0, 255, 255), 2)
        
        steps = [
            ("1. Detection", len(detections) > 0, (0, 255, 0) if len(detections) > 0 else (100, 100, 100)),
            ("2. Tracking", len(tracks) > 0, (0, 255, 0) if len(tracks) > 0 else (100, 100, 100)),
            ("3. Head Pose", any(hasattr(t, 'head_pose') for t in tracks), (0, 255, 0)),
            ("4. Trigger", False, (100, 100, 100)),  # Will update per track
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
                   cv2.FONT_HERSHEY_BOLD, 0.6, (0, 255, 255), 2)
        y += 30
        cv2.putText(frame, f"Detections: {len(detections)}", (10, y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        y += 25
        cv2.putText(frame, f"Active Tracks: {len(tracks)}", (10, y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Controls
        y = h - 100
        cv2.putText(frame, "CONTROLS:", (10, y), 
                   cv2.FONT_HERSHEY_BOLD, 0.6, (0, 255, 255), 2)
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
            # Track ID and age
            label = f"ID:{track.track_id} Age:{track.age}"
            cv2.putText(frame, label, (x1, y1 - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # Head pose
            if hasattr(track, 'head_pose') and track.head_pose:
                pose = track.head_pose
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
            if hasattr(track, 'should_recognize'):
                trigger_color = (0, 255, 0) if track.should_recognize else (100, 100, 100)
                trigger_text = "RECOGNIZE!" if track.should_recognize else "Waiting..."
                cv2.putText(frame, trigger_text, (x1, y2 + 20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, trigger_color, 2)
            
            # Embedding status
            if hasattr(track, 'has_embedding') and track.has_embedding:
                cv2.putText(frame, "Embedding OK", (x1, y2 + 40), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
        else:
            # Just confidence
            cv2.putText(frame, f"{conf:.2f}", (x1, y1 - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    def process_frame(self, frame):
        """Process a single frame through the pipeline."""
        # Step 1: Detect faces
        detections = self.detector.detect(frame)
        
        # Step 2: Track faces
        tracks = self.tracker.update(detections)
        
        # Step 3-5: Process each track
        for track in tracks:
            # Find matching detection
            det = next((d for d in detections if d["bbox"] == track.bbox), None)
            if not det:
                continue
            
            # Step 3: Estimate head pose
            if det.get("landmarks"):
                pose = self.head_pose.estimate_simple(det["landmarks"])
                track.head_pose = pose
            
            # Step 4: Check recognition trigger
            should_recognize = self.trigger.should_recognize(track)
            track.should_recognize = should_recognize
            
            # Step 5: Extract embedding if triggered
            if should_recognize and self.recognizer_available:
                try:
                    embedding = self.recognizer.extract_embedding(
                        frame, track.bbox, det.get("landmarks")
                    )
                    track.has_embedding = True
                    print(f"  → Extracted embedding for Track {track.track_id}")
                except Exception as e:
                    print(f"  → Failed to extract embedding: {e}")
                    track.has_embedding = False
            
            # Draw detection with all info
            self.draw_detection(frame, det, track)
        
        # Draw info panel
        self.draw_info_panel(frame, detections, tracks)
        
        return frame
    
    def run(self):
        """Run the demo."""
        print("Starting webcam...")
        print("Press 'q' to quit, 's' to save screenshot, 'r' to reset\n")
        
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("❌ Failed to open webcam!")
            return
        
        # Set resolution
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        cv2.namedWindow("Vision Pipeline Demo", cv2.WINDOW_NORMAL)
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to read frame")
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
            
            # Handle keys
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                print("\nQuitting...")
                break
            elif key == ord('s'):
                filename = f"screenshot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
                cv2.imwrite(filename, frame)
                print(f"  Screenshot saved: {filename}")
            elif key == ord('r'):
                print("  Resetting tracker...")
                self.tracker.reset()
        
        cap.release()
        cv2.destroyAllWindows()
        print("Demo ended.")


def main():
    """Main entry point."""
    print("=" * 60)
    print("REAL-TIME VISION PIPELINE DEMO")
    print("=" * 60)
    print()
    
    try:
        demo = VisualPipelineDemo()
        demo.run()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
