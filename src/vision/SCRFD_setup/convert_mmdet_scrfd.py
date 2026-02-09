"""
Convert SCRFD MMDetection model to ONNX.

The model.pth is an MMDetection checkpoint (v2.7.0) for SCRFD_2.5G_KPS.
This script converts it to ONNX format using MMDetection tools.
"""

import torch
import numpy as np
from pathlib import Path

def convert_mmdet_to_onnx():
    """
    Convert MMDetection SCRFD model to ONNX.
    
    The model.pth contains:
    - MMDetection version: 2.7.0
    - Model: SCRFD (Sample and Computation Redistributed Face Detection)
    - Variant: 2.5G with keypoints (KPS)
    """
    
    print("SCRFD MMDetection to ONNX Converter")
    print("=" * 60)
    
    # Check if MMDetection is installed
    try:
        import mmdet
        print(f"✓ MMDetection installed: {mmdet.__version__}")
    except ImportError:
        print("✗ MMDetection not installed")
        print("\nTo install MMDetection:")
        print("  pip install mmdet==2.7.0")
        print("  pip install mmcv-full")
        print("\nAlternatively, use the pre-converted ONNX model or OpenCV detector.")
        return False
    
    # Load checkpoint
    checkpoint_path = "model.pth"
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    print(f"\nCheckpoint info:")
    print(f"  MMDet version: {checkpoint['meta']['mmdet_version']}")
    print(f"  Classes: {checkpoint['meta']['CLASSES']}")
    
    # Check if config is in meta
    if 'cfg' in checkpoint['meta']:
        print("  Config: Available in checkpoint")
        cfg = checkpoint['meta']['cfg']
        
        # Try to build model from config
        try:
            from mmdet.models import build_detector
            from mmcv import Config
            
            # Create config object
            if isinstance(cfg, dict):
                cfg = Config(cfg)
            
            # Build model
            model = build_detector(cfg.model, test_cfg=cfg.get('test_cfg'))
            model.load_state_dict(checkpoint['state_dict'])
            model.eval()
            
            print("✓ Model built successfully")
            
            # Export to ONNX
            output_path = "models/scrfd_2.5g_kps_fp16.onnx"
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            
            # Create dummy input
            dummy_input = torch.randn(1, 3, 640, 640)
            
            print(f"\nExporting to ONNX: {output_path}")
            torch.onnx.export(
                model,
                dummy_input,
                output_path,
                export_params=True,
                opset_version=11,
                do_constant_folding=True,
                input_names=['input'],
                output_names=['output'],
                dynamic_axes={
                    'input': {0: 'batch_size'},
                    'output': {0: 'batch_size'}
                }
            )
            
            print("✓ ONNX export successful")
            
            # Verify
            import onnx
            onnx_model = onnx.load(output_path)
            onnx.checker.check_model(onnx_model)
            print("✓ ONNX model verified")
            
            return True
            
        except Exception as e:
            print(f"✗ Error building model: {e}")
            return False
    else:
        print("  Config: Not in checkpoint")
        print("\nNeed SCRFD config file to build model.")
        print("Download from: https://github.com/deepinsight/insightface/tree/master/detection/scrfd")
        return False


def alternative_solution():
    """Provide alternative solutions."""
    print("\n" + "=" * 60)
    print("Alternative Solutions:")
    print("=" * 60)
    
    print("\n1. Use Pre-converted ONNX Model:")
    print("   The SCRFD authors provide pre-converted ONNX models.")
    print("   Download from InsightFace model zoo.")
    
    print("\n2. Use OpenCV Face Detector (Immediate Solution):")
    print("   OpenCV includes built-in face detectors that work well.")
    print("   No additional downloads needed.")
    print("   See: FINAL_SETUP.md for implementation")
    
    print("\n3. Install MMDetection and Convert:")
    print("   pip install mmdet==2.7.0 mmcv-full")
    print("   python convert_mmdet_scrfd.py")
    
    print("\n4. Use the model with MMDetection directly:")
    print("   Keep using .pth format with MMDetection inference")


if __name__ == "__main__":
    success = convert_mmdet_to_onnx()
    
    if not success:
        alternative_solution()
