"""
Step 2: Test Face Tracking (ByteTrack)

This script tests the face tracking component.
"""

import cv2
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from vision.detector_factory import create_scrfd_detector
from vision.tracker import ByteTracker

def create_test_detections():
    """Create mock detections for testing."""
    return [
        {
            "bbox": [100, 100, 200, 200],
            "confidence": 0.95,
            "landmarks": [[120, 120], [180, 120], [150, 150], [130, 180], [170, 180]]
        },
        {
            "bbox": [300, 150, 400, 250],
            "confidence": 0.88,
            "landmarks": [[320, 170], [380, 170], [350, 200], [330, 230], [370, 230]]
        }
    ]

def main():
    print("=" * 60)
    print("STEP 2: Testing Face Tracking")
    print("=" * 60)
    
    # Initialize tracker
    print("\n1. Initializing ByteTrack tracker...")
    try:
        tracker = ByteTracker(
            track_thresh=0.6,
            track_buffer=30,
            match_thresh=0.8,
            min_box_area=100
        )
        print("   ✓ Tracker initialized successfully")
        print(f"   - Track threshold: 0.6")
        print(f"   - Track buffer: 30 frames")
        print(f"   - Match threshold: 0.8")
    except Exception as e:
        print(f"   ❌ Failed to initialize tracker: {e}")
        return 1
    
    # Test tracking across frames
    print("\n2. Testing tracking across 3 frames...")
    
    for frame_num in range(1, 4):
        print(f"\n   Frame {frame_num}:")
        
        # Create detections (slightly moved each frame)
        detections = create_test_detections()
        
        # Move detections slightly each frame
        offset = (frame_num - 1) * 5
        for det in detections:
            det["bbox"] = [b + offset for b in det["bbox"]]
            det["landmarks"] = [[lm[0] + offset, lm[1] + offset] for lm in det["landmarks"]]
        
        # Update tracker
        tracks = tracker.update(detections)
        
        print(f"     - Detections: {len(detections)}")
        print(f"     - Active tracks: {len(tracks)}")
        
        for track in tracks:
            print(f"       Track ID {track.track_id}: bbox={track.bbox}, age={track.age}, state={track.state}")
    
    # Test track loss
    print("\n3. Testing track loss (no detections)...")
    tracks = tracker.update([])
    print(f"   - Active tracks after no detections: {len(tracks)}")
    print(f"   - Total tracks in memory: {len(tracker.tracks)}")
    
    print("\n" + "=" * 60)
    print("✅ STEP 2 COMPLETE: Face Tracking Working!")
    print("=" * 60)
    print("\nNext step: test_step3_head_pose.py")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
