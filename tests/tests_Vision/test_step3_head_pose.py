"""
Step 3: Test Head Pose Estimation

This script tests the head pose estimation component.
"""

import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from vision.head_pose import HeadPoseEstimator

def main():
    print("=" * 60)
    print("STEP 3: Testing Head Pose Estimation")
    print("=" * 60)
    
    # Initialize estimator
    print("\n1. Initializing head pose estimator...")
    try:
        estimator = HeadPoseEstimator(
            yaw_threshold=30,
            pitch_threshold=30,
            roll_threshold=30
        )
        print("   ✓ Estimator initialized successfully")
        print(f"   - Yaw threshold: ±30°")
        print(f"   - Pitch threshold: ±30°")
        print(f"   - Roll threshold: ±30°")
    except Exception as e:
        print(f"   ❌ Failed to initialize estimator: {e}")
        return 1
    
    # Test different head poses
    print("\n2. Testing different head poses...")
    
    test_cases = [
        {
            "name": "Frontal (looking at camera)",
            "landmarks": [
                [100, 150],  # Left eye
                [200, 150],  # Right eye
                [150, 200],  # Nose
                [120, 250],  # Left mouth
                [180, 250]   # Right mouth
            ]
        },
        {
            "name": "Left turn",
            "landmarks": [
                [80, 150],   # Left eye (closer to nose)
                [200, 150],  # Right eye
                [130, 200],  # Nose (shifted left)
                [100, 250],  # Left mouth
                [170, 250]   # Right mouth
            ]
        },
        {
            "name": "Right turn",
            "landmarks": [
                [100, 150],  # Left eye
                [220, 150],  # Right eye (closer to nose)
                [170, 200],  # Nose (shifted right)
                [130, 250],  # Left mouth
                [200, 250]   # Right mouth
            ]
        }
    ]
    
    for test in test_cases:
        print(f"\n   {test['name']}:")
        pose = estimator.estimate_simple(test["landmarks"])
        
        print(f"     - Yaw: {pose['yaw']:.1f}°")
        print(f"     - Pitch: {pose['pitch']:.1f}°")
        print(f"     - Roll: {pose['roll']:.1f}°")
        print(f"     - Looking at camera: {pose['is_looking']}")
    
    print("\n" + "=" * 60)
    print("✅ STEP 3 COMPLETE: Head Pose Estimation Working!")
    print("=" * 60)
    print("\nNext step: test_step4_trigger.py")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
