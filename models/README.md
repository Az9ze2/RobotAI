# Models Directory

This directory contains the face detection and recognition models.

## Current Models

### Face Recognition (ArcFace)
- **File**: `arcface_r100_v1_fp16.onnx`
- **Size**: ~166 MB
- **Format**: ONNX
- **Status**: ✅ Ready to use
- **Purpose**: Extract 512-D face embeddings for recognition

### Face Detection (SCRFD)
- **File**: `scrfd_2.5g_kps.pth`
- **Size**: ~3 MB
- **Format**: PyTorch checkpoint (MMDetection)
- **Status**: ✅ Loaded (using OpenCV fallback)
- **Purpose**: Detect faces and facial landmarks

## Usage in Config

Update `config/local.yaml` to use these models:

```yaml
vision:
  detection:
    model_path: "models/scrfd_2.5g_kps.pth"  # PyTorch model
    confidence_threshold: 0.7
    nms_threshold: 0.4
    input_size: [640, 640]
    device: "cpu"  # or "cuda"
  
  recognition:
    model_path: "models/arcface_r100_v1_fp16.onnx"  # ONNX model
    device: "cuda"
```

## Model Formats

### ONNX Models (.onnx)
- Used with ONNX Runtime
- Supports TensorRT acceleration
- Best for production deployment
- Current: ArcFace recognition model

### PyTorch Models (.pth)
- PyTorch checkpoints
- Can be used directly with PyTorch
- Can be converted to ONNX for optimization
- Current: SCRFD detection model

## Future Models

When you convert SCRFD to ONNX or download the ONNX version:
- `scrfd_2.5g_kps_fp16.onnx` - ONNX version of SCRFD for optimal performance

## TensorRT Cache

TensorRT will create optimized engine files in `./models/trt_cache/` on first run.
This is normal and improves performance.
