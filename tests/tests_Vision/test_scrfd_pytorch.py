"""
Test the SCRFD PyTorch detector with model.pth

This script tests the PyTorch-based SCRFD detector using your model.pth file.
"""

import cv2
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from vision import create_scrfd_detector

def main():
    print("=" * 60)
    print("SCRFD PyTorch Detector Test")
    print("=" * 60)
    
    # Create detector (automatically uses PyTorch for .pth files)
    print("\nInitializing detector with model.pth...")
    detector = create_scrfd_detector(
        model_path="model.pth",
        confidence_threshold=0.7,
        nms_threshold=0.4,
        device="cpu"  # Use CPU for now
    )
    
    print(f"Detector type: {type(detector).__name__}")
    print("\nOpening webcam...")
    
    # Open webcam
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return
    
    print("\nControls:")
    print("  'q' - Quit")
    print("  'space' - Pause/Resume")
    print("\nStarting detection...")
    
    paused = False
    frame_count = 0
    
    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame")
                break
            
            frame_count += 1
            
            # Detect faces
            detections = detector.detect(frame)
            
            # Draw results
            for det in detections:
                bbox = det["bbox"]
                conf = det["confidence"]
                landmarks = det.get("landmarks")
                
                # Draw bounding box
                x1, y1, x2, y2 = map(int, bbox)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Draw confidence
                label = f"{conf:.2f}"
                cv2.putText(
                    frame,
                    label,
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    2
                )
                
                # Draw landmarks if available
                if landmarks:
                    for lm in landmarks:
                        cv2.circle(frame, (int(lm[0]), int(lm[1])), 2, (0, 0, 255), -1)
            
            # Show info
            info_text = f"Faces: {len(detections)} | Frame: {frame_count}"
            cv2.putText(
                frame,
                info_text,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )
        
        # Show frame
        cv2.imshow('SCRFD PyTorch Detector Test', frame)
        
        # Handle keys
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord(' '):
            paused = not paused
            print(f"{'Paused' if paused else 'Resumed'}")
    
    cap.release()
    cv2.destroyAllWindows()
    print("\nTest completed!")


if __name__ == "__main__":
    main()
