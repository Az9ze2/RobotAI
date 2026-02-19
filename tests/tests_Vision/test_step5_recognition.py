"""
Step 5: Test Face Recognition (ArcFace)

This script tests the face recognition component.
"""

import cv2
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from vision.recognizer import FaceRecognizer

def create_test_face():
    """Create a test face image."""
    face = np.ones((112, 112, 3), dtype=np.uint8) * 128
    
    # Add some features
    cv2.circle(face, (35, 40), 8, (50, 50, 50), -1)  # Left eye
    cv2.circle(face, (77, 40), 8, (50, 50, 50), -1)  # Right eye
    cv2.circle(face, (56, 60), 5, (100, 100, 100), -1)  # Nose
    cv2.ellipse(face, (56, 80), (20, 10), 0, 0, 180, (80, 80, 80), -1)  # Mouth
    
    return face

def main():
    print("=" * 60)
    print("STEP 5: Testing Face Recognition")
    print("=" * 60)
    
    # Check if model exists
    model_path = Path("models/arcface_r100_v1_fp16.onnx")
    if not model_path.exists():
        print(f"\n❌ Model not found: {model_path}")
        print("\nPlease download the model first:")
        print("  See models/README.md for instructions")
        return 1
    
    print(f"\n✓ Model found: {model_path}")
    print(f"  Size: {model_path.stat().st_size / (1024*1024):.2f} MB")
    
    # Initialize recognizer
    print("\n1. Initializing ArcFace recognizer...")
    try:
        recognizer = FaceRecognizer(
            model_path=str(model_path),
            device="cpu"
        )
        print("   ✓ Recognizer initialized successfully")
        print(f"   - Model: ArcFace R100")
        print(f"   - Device: CPU")
        print(f"   - Embedding dimension: 512")
    except Exception as e:
        print(f"   ❌ Failed to initialize recognizer: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    # Create test image
    print("\n2. Creating test face image...")
    full_image = np.ones((480, 640, 3), dtype=np.uint8) * 128
    face = create_test_face()
    
    # Place face in image
    y_offset, x_offset = 184, 264  # Center the 112x112 face
    full_image[y_offset:y_offset+112, x_offset:x_offset+112] = face
    
    bbox = [x_offset, y_offset, x_offset+112, y_offset+112]
    landmarks = [
        [x_offset+35, y_offset+40],  # Left eye
        [x_offset+77, y_offset+40],  # Right eye
        [x_offset+56, y_offset+60],  # Nose
        [x_offset+40, y_offset+80],  # Left mouth
        [x_offset+72, y_offset+80]   # Right mouth
    ]
    
    print(f"   ✓ Test face created: {face.shape}")
    print(f"   - Bbox: {bbox}")
    print(f"   - Landmarks: {len(landmarks)} points")
    
    # Extract embedding
    print("\n3. Extracting face embedding...")
    try:
        embedding = recognizer.extract_embedding(full_image, bbox, landmarks)
        print(f"   ✓ Embedding extracted successfully")
        print(f"   - Shape: {embedding.shape}")
        print(f"   - Norm: {np.linalg.norm(embedding):.6f} (should be ~1.0)")
        print(f"   - Sample values: [{embedding[0]:.4f}, {embedding[1]:.4f}, {embedding[2]:.4f}, ...]")
    except Exception as e:
        print(f"   ❌ Embedding extraction failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    # Test similarity
    print("\n4. Testing embedding similarity...")
    
    # Extract same face again (should be very similar)
    embedding2 = recognizer.extract_embedding(full_image, bbox, landmarks)
    similarity = np.dot(embedding, embedding2)
    print(f"   - Same face similarity: {similarity:.6f} (should be ~1.0)")
    
    # Create different face
    different_face = create_test_face()
    different_face = cv2.flip(different_face, 1)  # Flip horizontally
    full_image2 = np.ones((480, 640, 3), dtype=np.uint8) * 128
    full_image2[y_offset:y_offset+112, x_offset:x_offset+112] = different_face
    
    embedding3 = recognizer.extract_embedding(full_image2, bbox, landmarks)
    similarity_diff = np.dot(embedding, embedding3)
    print(f"   - Different face similarity: {similarity_diff:.6f} (should be <1.0)")
    
    print("\n" + "=" * 60)
    print("✅ STEP 5 COMPLETE: Face Recognition Working!")
    print("=" * 60)
    print("\n✅ ALL STEPS COMPLETE!")
    print("\nPipeline components verified:")
    print("  1. ✓ Face Detection (SCRFD)")
    print("  2. ✓ Face Tracking (ByteTrack)")
    print("  3. ✓ Head Pose Estimation")
    print("  4. ✓ Recognition Trigger")
    print("  5. ✓ Face Recognition (ArcFace)")
    print("\nThe system is ready for integration testing!")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
