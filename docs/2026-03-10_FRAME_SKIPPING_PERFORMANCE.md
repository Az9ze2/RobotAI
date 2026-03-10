# Hardware Performance Report: Frame Skipping Tracker

**Date**: March 10, 2026

## Objective
To implement and benchmark a "Frame-Skipping Object Tracker" to determine if the camera's full potential (~30 FPS) can be realized on the NVIDIA Jetson platform by offloading the face detector workload. 

## Hardware & System Environment 
The benchmarks were run natively on the Jetson target hardware:
- **Architecture**: ARM64 (aarch64)
- **CPU**: 6-core Cortex-A78AE (max 1.51 GHz)
- **RAM**: ~3.5 GB Physical / 11 GB Swap (Unified Memory architecture)
- **OS**: Ubuntu 22.04.5 LTS
- **Camera**: Intel RealSense D415 (via `pyrealsense2` at 640x480)
- **Deep Learning Stack**: PyTorch, `facenet-pytorch` (MTCNN Face Detector + InceptionResnetV1 Recognizer)
- **Execution Target**: CPU (Due to ONNX/CUDA unrecoverable out-of-memory errors on this specific Nano board).

## Methodology
The vision pipeline has iterated through several versions. To clarify the benchmarks, here is the version mapping:
- **v1.0 (Jetson Native)**: `demo_realtime_visual_jetson.py` (ONNX/SCRFD)
- **v1.1 (RealSense Fix)**: `demo_realtime_visual_jetson_realsense.py` (ONNX/SCRFD + Intel RealSense SDK)
- **v2.0 (PyTorch CPU)**: `demo_realtime_visual_facenet.py` (MTCNN/FaceNet)
- **v2.1 (Frame Skipping)**: `demo_realtime_visual_facenet_skip.py` (MTCNN/FaceNet + OpenCV CSRT)
- **Camera Benchmark**: `demo_realtime_visual_realsense.py` (No AI, pure camera feed)

The v2.0 pipeline (`demo_realtime_visual_facenet.py`) processed the heavy MTCNN face detector on *every single frame*. 

To optimize this, a new v2.1 pipeline (`demo_realtime_visual_facenet_skip.py`) was created with the following frame-skipping mechanism:
1. **Frame 0 (Detection)**: The heavy MTCNN PyTorch model detects faces. For each face found, an OpenCV `CSRT` (Channel and Spatial Reliability Tracking) tracking algorithm is initialized around the bounding box.
2. **Frames 1 through 9 (Tracking)**: The MTCNN detector is completely bypassed. The OpenCV `CSRT` trackers update the faces' position mathematically without invoking deep learning. This uses significantly less CPU.
3. **Frame 10 (Detection)**: The process repeats, refreshing the true bounding boxes using MTCNN.

## Benchmark Results (100-Frame Test)
Both scripts were executed on the exact same hardware reading the exact same RealSense input stream.

| Pipeline Variation | FPS | Improvement | CPU Usage (Process) |
| :--- | :--- | :--- | :--- |
| **Camera Benchmark** (Pure Capture, no AI) | ~29.8 | Baseline | ~10% |
| **v2.0** (MTCNN on Every Frame) | **5.1 FPS** | 1.0x | 85% - 94% |
| **v2.1** (MTCNN Every 10th Frame + CSRT) | **14.2 FPS** | **2.78x** | 35% - 40% |

## Conclusion
Implementing the frame-skipping logic with OpenCV's `CSRT` mathematical tracker successfully boosted the face-tracking pipeline from **5.1 FPS** to **14.2 FPS**, a massive **278% performance increase**, while drastically reducing CPU load.

However, the pipeline still falls short of the RealSense camera's 30 FPS physical limit. The limiting factors are:
1. **OpenCV CSRT Overhead**: While the CSRT algorithm is much faster than PyTorch MTCNN, it still carries an internal processing cost per tracked object per frame, creating a soft ceiling on performance. Replacing this with an even lighter tracker (like `MOSSE` or `KCF`) could yield 20+ FPS.
2. **CPU-only Execution**: The lack of functioning CUDA/TensorRT acceleration on this specific Jetson configuration prevents the heavy AI models from running efficiently. If GPU functionality is restored, the baseline without frame-skipping would likely exceed 20 FPS natively, pushing the skipped pipeline past 60 FPS.
