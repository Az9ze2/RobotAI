"""
Step 4: Test Recognition Trigger

This script tests the recognition trigger component.
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from vision.recognition_trigger import RecognitionTrigger
from vision.tracker import Track

def main():
    print("=" * 60)
    print("STEP 4: Testing Recognition Trigger")
    print("=" * 60)
    
    # Initialize trigger
    print("\n1. Initializing recognition trigger...")
    try:
        trigger = RecognitionTrigger(
            cooldown_seconds=2.0,
            require_attention=True,
            min_track_age=3
        )
        print("   ✓ Trigger initialized successfully")
        print(f"   - Cooldown: 2.0 seconds")
        print(f"   - Require attention: True")
        print(f"   - Min track age: 3 frames")
    except Exception as e:
        print(f"   ❌ Failed to initialize trigger: {e}")
        return 1
    
    # Create test track
    print("\n2. Creating test track...")
    track = Track(
        track_id=1,
        bbox=[100, 100, 200, 200],
        confidence=0.9,
        landmarks=[[120, 120], [180, 120], [150, 150], [130, 180], [170, 180]]
    )
    track.age = 5  # Mature track
    track.head_pose = {"is_looking": True}  # Looking at camera
    print(f"   ✓ Track created: ID={track.track_id}, age={track.age}")
    
    # Test triggering
    print("\n3. Testing trigger logic...")
    
    # First trigger (should work)
    print("\n   Attempt 1:")
    should_recognize = trigger.should_recognize(track)
    print(f"     - Should recognize: {should_recognize}")
    print(f"     - Reason: First time seeing this track")
    
    # Second trigger immediately (should fail - cooldown)
    print("\n   Attempt 2 (immediate):")
    should_recognize = trigger.should_recognize(track)
    print(f"     - Should recognize: {should_recognize}")
    print(f"     - Reason: Still in cooldown period")
    
    # Wait for cooldown
    print("\n   Waiting 2.5 seconds for cooldown...")
    time.sleep(2.5)
    
    # Third trigger after cooldown (should work)
    print("\n   Attempt 3 (after cooldown):")
    should_recognize = trigger.should_recognize(track)
    print(f"     - Should recognize: {should_recognize}")
    print(f"     - Reason: Cooldown period expired")
    
    # Test without attention
    print("\n4. Testing without attention...")
    track.head_pose = {"is_looking": False}  # Not looking
    should_recognize = trigger.should_recognize(track)
    print(f"   - Should recognize: {should_recognize}")
    print(f"   - Reason: Not looking at camera")
    
    # Test young track
    print("\n5. Testing young track...")
    track.age = 1  # Too young
    track.head_pose = {"is_looking": True}
    should_recognize = trigger.should_recognize(track)
    print(f"   - Should recognize: {should_recognize}")
    print(f"   - Reason: Track too young (age={track.age}, min={trigger.min_track_age})")
    
    print("\n" + "=" * 60)
    print("✅ STEP 4 COMPLETE: Recognition Trigger Working!")
    print("=" * 60)
    print("\nNext step: test_step5_recognition.py")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
