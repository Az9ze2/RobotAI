"""
Step 1: Test Face Detection (SCRFD)

This script tests the face detection component using the PyTorch SCRFD model.
"""

import cv2
import numpy as np
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from vision.detector_factory import create_scrfd_detector

def create_test_image():
    """Create a simple test image with a face-like pattern."""
    img = np.ones((480, 640, 3), dtype=np.uint8) * 128
    
    # Draw a simple face
    cv2.rectangle(img, (200, 150), (440, 380), (150, 150, 150), -1)  # Face
    cv2.circle(img, (270, 220), 20, (50, 50, 50), -1)  # Left eye
    cv2.circle(img, (370, 220), 20, (50, 50, 50), -1)  # Right eye
    cv2.circle(img, (320, 280), 15, (100, 100, 100), -1)  # Nose
    cv2.ellipse(img, (320, 320), (50, 25), 0, 0, 180, (80, 80, 80), -1)  # Mouth
    
    return img

def main():
    print("=" * 60)
    print("STEP 1: Testing Face Detection")
    print("=" * 60)
    
    # Check if model exists
    model_path = Path("models/scrfd_2.5g_kps.pth")
    if not model_path.exists():
        print(f"\n❌ Model not found: {model_path}")
        print("\nPlease download the model first:")
        print("  See models/SCRFD_DOWNLOAD_SOLUTION.md for instructions")
        return 1
    
    print(f"\n✓ Model found: {model_path}")
    print(f"  Size: {model_path.stat().st_size / (1024*1024):.2f} MB")
    
    # Initialize detector
    print("\n1. Initializing SCRFD detector...")
    try:
        detector = create_scrfd_detector(
            model_path=str(model_path),
            confidence_threshold=0.5,
            nms_threshold=0.4,
            input_size=(640, 640),
            device="cpu"
        )
        print("   ✓ Detector initialized successfully")
        print(f"   - Using: PyTorch SCRFD with OpenCV fallback")
        print(f"   - Device: CPU")
        print(f"   - Confidence threshold: 0.5")
    except Exception as e:
        print(f"   ❌ Failed to initialize detector: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    # Create test image
    print("\n2. Creating test image...")
    test_image = create_test_image()
    print(f"   ✓ Test image created: {test_image.shape}")
    
    # Detect faces
    print("\n3. Detecting faces...")
    try:
        detections = detector.detect(test_image)
        print(f"   ✓ Detection complete")
        print(f"   - Detected {len(detections)} face(s)")
        
        for i, det in enumerate(detections):
            bbox = det["bbox"]
            conf = det["confidence"]
            landmarks = det.get("landmarks", [])
            
            print(f"\n   Face {i+1}:")
            print(f"     - Bbox: [{bbox[0]:.0f}, {bbox[1]:.0f}, {bbox[2]:.0f}, {bbox[3]:.0f}]")
            print(f"     - Confidence: {conf:.3f}")
            print(f"     - Landmarks: {len(landmarks)} points")
            
            # Draw on image
            x1, y1, x2, y2 = map(int, bbox)
            cv2.rectangle(test_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(test_image, f"{conf:.2f}", (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # Draw landmarks
            if landmarks:
                for lm in landmarks:
                    cv2.circle(test_image, (int(lm[0]), int(lm[1])), 3, (0, 0, 255), -1)
        
    except Exception as e:
        print(f"   ❌ Detection failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    # Save result
    output_path = "test_detection_result.jpg"
    cv2.imwrite(output_path, test_image)
    print(f"\n4. Result saved to: {output_path}")
    
    print("\n" + "=" * 60)
    print("✅ STEP 1 COMPLETE: Face Detection Working!")
    print("=" * 60)
    print("\nNext step: test_step2_tracking.py")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
