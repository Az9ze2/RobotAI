import sys
import os
import onnxruntime
import numpy as np
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from vision.detector_factory import create_scrfd_detector

def check_model():
    model_path = "models/buffalo_l/det_10g.onnx"
    print(f"Checking model: {model_path}")
    
    # 1. Inspect ONNX model directly
    try:
        session = onnxruntime.InferenceSession(model_path, providers=['CPUExecutionProvider'])
        print("\nModel Inputs:")
        for i in session.get_inputs():
            print(f"  {i.name}: {i.shape}")
            
        print("\nModel Outputs:")
        for i, out in enumerate(session.get_outputs()):
            print(f"  {out.name}: {out.shape}")
            
    except Exception as e:
        print(f"Failed to inspect model: {e}")
        return

    # 2. Run inference to see actual output shapes
    print("\nRunning test inference...")
    try:
        detector = create_scrfd_detector(
            model_path=model_path,
            confidence_threshold=0.5,
            device="cpu"
        )
        
        # Create dummy image
        img = np.zeros((640, 640, 3), dtype=np.uint8)
        
        # Run detection (this will trigger our debug prints)
        detector.detect(img)
        
    except Exception as e:
        print(f"\nInference failed: {e}")

if __name__ == "__main__":
    check_model()
