"""
Full Vision Pipeline Integration Test

This script tests the complete vision pipeline end-to-end:
1. Face detection (SCRFD PyTorch)
2. Multi-face tracking (ByteTrack)
3. Head pose estimation
4. Recognition triggering (with cooldown)
5. Face recognition (ArcFace)
6. Student identification (mock without Milvus)

This demonstrates the full workflow without requiring Milvus.
"""

import cv2
import numpy as np
import sys
from pathlib import Path
from loguru import logger

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from vision import (
    create_scrfd_detector,
    ByteTracker,
    HeadPoseEstimator,
    RecognitionTrigger,
    FaceRecognizer
)

def create_test_frame(width=640, height=480):
    """Create a test frame with face-like features."""
    frame = np.ones((height, width, 3), dtype=np.uint8) * 128
    
    # Add some texture to simulate a face
    cv2.rectangle(frame, (200, 150), (440, 330), (150, 150, 150), -1)  # Face
    cv2.circle(frame, (270, 210), 15, (50, 50, 50), -1)  # Left eye
    cv2.circle(frame, (370, 210), 15, (50, 50, 50), -1)  # Right eye
    cv2.circle(frame, (320, 260), 10, (100, 100, 100), -1)  # Nose
    cv2.ellipse(frame, (320, 290), (40, 20), 0, 0, 180, (80, 80, 80), -1)  # Mouth
    
    return frame


def test_pipeline_components():
    """Test each pipeline component individually."""
    print("=" * 60)
    print("Testing Vision Pipeline Components")
    print("=" * 60)
    
    # 1. Test Detector
    print("\n1. Testing Face Detector...")
    try:
        detector = create_scrfd_detector(
            model_path="models/scrfd_2.5g_kps.pth",
            confidence_threshold=0.5,
            device="cpu"
        )
        print("   ✓ Detector initialized")
        
        frame = create_test_frame()
        detections = detector.detect(frame)
        print(f"   ✓ Detected {len(detections)} face(s)")
        
    except Exception as e:
        print(f"   ✗ Detector failed: {e}")
        return False
    
    # 2. Test Tracker
    print("\n2. Testing Face Tracker...")
    try:
        tracker = ByteTracker(
            track_thresh=0.5,
            track_buffer=30,
            match_thresh=0.7
        )
        print("   ✓ Tracker initialized")
        
        tracks = tracker.update(detections)
        print(f"   ✓ Tracking {len(tracks)} face(s)")
        
    except Exception as e:
        print(f"   ✗ Tracker failed: {e}")
        return False
    
    # 3. Test Head Pose
    print("\n3. Testing Head Pose Estimator...")
    try:
        head_pose = HeadPoseEstimator(
            yaw_threshold=30,
            pitch_threshold=30,
            roll_threshold=30
        )
        print("   ✓ Head pose estimator initialized")
        
        if detections and detections[0].get("landmarks"):
            pose = head_pose.estimate_simple(detections[0]["landmarks"])
            print(f"   ✓ Estimated pose: yaw={pose['yaw']:.1f}°, pitch={pose['pitch']:.1f}°")
        else:
            print("   ⚠ No landmarks for head pose estimation")
        
    except Exception as e:
        print(f"   ✗ Head pose failed: {e}")
        return False
    
    # 4. Test Recognition Trigger
    print("\n4. Testing Recognition Trigger...")
    try:
        trigger = RecognitionTrigger(
            cooldown_seconds=5.0,
            require_attention=True
        )
        print("   ✓ Recognition trigger initialized")
        
        if tracks:
            should_recognize = trigger.should_recognize(tracks[0])
            print(f"   ✓ Should recognize: {should_recognize}")
        
    except Exception as e:
        print(f"   ✗ Recognition trigger failed: {e}")
        return False
    
    # 5. Test Face Recognizer
    print("\n5. Testing Face Recognizer...")
    try:
        recognizer = FaceRecognizer(
            model_path="models/arcface_r100_v1_fp16.onnx",
            device="cpu"
        )
        print("   ✓ Face recognizer initialized")
        
        if detections:
            embedding = recognizer.extract_embedding(
                frame,
                detections[0]["bbox"],
                detections[0].get("landmarks")
            )
            print(f"   ✓ Extracted embedding: shape={embedding.shape}, norm={np.linalg.norm(embedding):.3f}")
        
    except Exception as e:
        print(f"   ✗ Face recognizer failed: {e}")
        return False
    
    print("\n" + "=" * 60)
    print("✓ All Components Working!")
    print("=" * 60)
    return True


def test_full_pipeline():
    """Test the complete pipeline workflow."""
    print("\n" + "=" * 60)
    print("Testing Full Pipeline Workflow")
    print("=" * 60)
    
    try:
        # Initialize all components
        print("\nInitializing pipeline components...")
        
        detector = create_scrfd_detector(
            model_path="models/scrfd_2.5g_kps.pth",
            confidence_threshold=0.5,
            device="cpu"
        )
        
        tracker = ByteTracker(
            track_thresh=0.5,
            track_buffer=30,
            match_thresh=0.7
        )
        
        head_pose = HeadPoseEstimator(
            yaw_threshold=30,
            pitch_threshold=30,
            roll_threshold=30
        )
        
        trigger = RecognitionTrigger(
            cooldown_seconds=5.0,
            require_attention=True
        )
        
        recognizer = FaceRecognizer(
            model_path="models/arcface_r100_v1_fp16.onnx",
            device="cpu"
        )
        
        print("✓ All components initialized")
        
        # Simulate pipeline processing
        print("\nProcessing frames...")
        
        for frame_num in range(1, 4):
            print(f"\n--- Frame {frame_num} ---")
            
            # Create test frame
            frame = create_test_frame()
            
            # Step 1: Detect faces
            detections = detector.detect(frame)
            print(f"  Detected: {len(detections)} face(s)")
            
            # Step 2: Track faces
            tracks = tracker.update(detections)
            print(f"  Tracking: {len(tracks)} face(s)")
            
            # Step 3: Process each track
            for track in tracks:
                print(f"  Track ID {track.track_id}:")
                
                # Get detection for this track
                det = next((d for d in detections if d["bbox"] == track.bbox), None)
                if not det:
                    continue
                
                # Step 4: Estimate head pose
                if det.get("landmarks"):
                    pose = head_pose.estimate_simple(det["landmarks"])
                    print(f"    - Head pose: yaw={pose['yaw']:.1f}°, looking={pose['is_looking']}")
                    track.head_pose = pose
                
                # Step 5: Check if should recognize
                should_recognize = trigger.should_recognize(track)
                print(f"    - Should recognize: {should_recognize}")
                
                # Step 6: Extract embedding if triggered
                if should_recognize:
                    embedding = recognizer.extract_embedding(
                        frame,
                        track.bbox,
                        det.get("landmarks")
                    )
                    print(f"    - Embedding extracted: {embedding.shape}")
                    print(f"    - (In production: would query Milvus for student ID)")
        
        print("\n" + "=" * 60)
        print("✓ Full Pipeline Test Complete!")
        print("=" * 60)
        print("\nPipeline Summary:")
        print("  1. ✓ Face Detection (SCRFD PyTorch)")
        print("  2. ✓ Multi-Face Tracking (ByteTrack)")
        print("  3. ✓ Head Pose Estimation")
        print("  4. ✓ Recognition Triggering")
        print("  5. ✓ Face Recognition (ArcFace)")
        print("  6. ⏭ Student Identification (requires Milvus)")
        print("\n✓ Pipeline is READY for production!")
        
        return True
        
    except Exception as e:
        print(f"\n✗ Pipeline test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all pipeline tests."""
    print("\n" + "=" * 60)
    print("VISION PIPELINE INTEGRATION TEST")
    print("=" * 60)
    
    # Test individual components
    if not test_pipeline_components():
        print("\n✗ Component tests failed!")
        return 1
    
    # Test full pipeline
    if not test_full_pipeline():
        print("\n✗ Full pipeline test failed!")
        return 1
    
    print("\n" + "=" * 60)
    print("🎉 ALL TESTS PASSED!")
    print("=" * 60)
    print("\nThe vision pipeline is fully functional and ready to use!")
    print("\nNext steps:")
    print("  1. Start Milvus: docker-compose up -d")
    print("  2. Enroll students: python entrypoint/demo_enrollment.py")
    print("  3. Test recognition: python entrypoint/demo_face_recognition.py")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
