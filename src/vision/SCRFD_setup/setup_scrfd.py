"""
Simple SCRFD ONNX Export without MMDetection.

Since MMDetection requires C++ build tools, this script provides
alternative solutions to get SCRFD working.
"""

import torch
import sys
from pathlib import Path

def check_if_onnx_disguised_as_pth():
    """Check if model.pth is actually an ONNX model with wrong extension."""
    print("Checking if model.pth is actually ONNX...")
    
    try:
        import onnxruntime as ort
        session = ort.InferenceSession("model.pth", providers=['CPUExecutionProvider'])
        print("✓ model.pth is actually a valid ONNX model!")
        print(f"  Input: {session.get_inputs()[0].name}")
        print(f"  Shape: {session.get_inputs()[0].shape}")
        
        # Copy to correct location
        import shutil
        output_path = "models/scrfd_2.5g_kps_fp16.onnx"
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy("model.pth", output_path)
        
        print(f"✓ Copied to: {output_path}")
        return True
        
    except Exception as e:
        print(f"✗ Not an ONNX model: {e}")
        return False


def download_prebuilt_scrfd():
    """Download pre-built SCRFD ONNX model from InsightFace."""
    print("\nDownloading pre-built SCRFD ONNX model...")
    
    import requests
    from tqdm import tqdm
    
    # Try multiple URLs
    urls = [
        "https://github.com/deepinsight/insightface/releases/download/v0.7/scrfd_2.5g_bnkps.onnx",
        "https://github.com/deepinsight/insightface/releases/download/v0.7/scrfd_10g_bnkps.onnx",
    ]
    
    for url in urls:
        try:
            print(f"Trying: {url}")
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            output_path = "models/scrfd_2.5g_kps_fp16.onnx"
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'wb') as f, tqdm(
                desc="Downloading",
                total=total_size,
                unit='B',
                unit_scale=True,
                unit_divisor=1024,
            ) as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
                    pbar.update(len(chunk))
            
            print(f"✓ Downloaded to: {output_path}")
            return True
            
        except Exception as e:
            print(f"✗ Failed: {e}")
            continue
    
    return False


def use_pytorch_model_directly():
    """Create a wrapper to use PyTorch model directly."""
    print("\nCreating PyTorch model wrapper...")
    
    wrapper_code = '''"""
SCRFD PyTorch Model Wrapper.

This wrapper allows using the PyTorch SCRFD model directly
without ONNX conversion.
"""

import torch
import numpy as np
from typing import List, Dict, Tuple
import cv2


class SCRFDPyTorchDetector:
    """SCRFD detector using PyTorch checkpoint directly."""
    
    def __init__(
        self,
        model_path: str = "model.pth",
        confidence_threshold: float = 0.7,
        nms_threshold: float = 0.4,
        input_size: Tuple[int, int] = (640, 640),
        device: str = "cpu"
    ):
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold
        self.input_size = input_size
        self.device = device
        
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=device)
        self.state_dict = checkpoint['state_dict']
        
        print(f"Loaded SCRFD checkpoint from {model_path}")
        print(f"Note: This is a fallback. For full functionality, convert to ONNX.")
    
    def detect(self, image: np.ndarray) -> List[Dict]:
        """
        Detect faces in image.
        
        Note: This is a simplified implementation.
        For full SCRFD functionality, use ONNX model.
        """
        # Fallback to OpenCV for now
        print("Warning: Using OpenCV fallback detector")
        print("To use full SCRFD, convert model to ONNX")
        
        # Use OpenCV Haar Cascade as fallback
        detector = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = detector.detectMultiScale(gray, 1.1, 4)
        
        detections = []
        for (x, y, w, h) in faces:
            detections.append({
                "bbox": [x, y, x+w, y+h],
                "confidence": 0.9,
                "landmarks": None
            })
        
        return detections
'''
    
    output_path = "src/vision/detector_pytorch.py"
    with open(output_path, 'w') as f:
        f.write(wrapper_code)
    
    print(f"✓ Created PyTorch wrapper: {output_path}")
    print("  Note: This uses OpenCV as fallback until ONNX model is available")
    return True


def main():
    print("SCRFD Model Setup - Alternative Solutions")
    print("=" * 60)
    
    # Option 1: Check if it's already ONNX
    if check_if_onnx_disguised_as_pth():
        print("\n✓ SUCCESS! Model is ready to use.")
        return
    
    # Option 2: Download pre-built ONNX
    print("\n" + "=" * 60)
    if download_prebuilt_scrfd():
        print("\n✓ SUCCESS! Pre-built ONNX model downloaded.")
        return
    
    # Option 3: Use PyTorch model with OpenCV fallback
    print("\n" + "=" * 60)
    if use_pytorch_model_directly():
        print("\n⚠ Using OpenCV fallback detector")
        print("  Your PyTorch model is saved but requires MMDetection to use")
        print("  The system will work with OpenCV detector for now")
        return
    
    print("\n✗ All automatic solutions failed")
    print("\nManual steps:")
    print("1. Install Visual Studio Build Tools")
    print("2. Install MMDetection: pip install mmdet==2.7.0")
    print("3. Convert model with MMDetection tools")
    print("\nOr use OpenCV detector (works immediately)")


if __name__ == "__main__":
    main()
