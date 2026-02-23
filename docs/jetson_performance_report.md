# Jetson Orin CPU Performance Report: Vision Pipeline

This report summarizes the performance of the RobotAI Vision Pipeline (`demo_realtime_visual_jetson.py`) when running strictly on the CPU of the NVIDIA Jetson Orin (JetPack R36 / Linux 5.15 aarch64).

## 1. System Context

*   **Platform**: NVIDIA Jetson Orin (aarch64)
*   **OS**: Linux 5.15.148-tegra (JetPack R36.4.7)
*   **CPU**: 12 Cores
*   **RAM**: ~3.5 GB available to the OS (Unified Memory architecture; the remainder of the 8GB is reserved as a GPU carve-out by the manufacturer).
*   **Inference Engine**: ONNX Runtime 1.23.2 (`CPUExecutionProvider`)

## 2. Resource Utilization (Live Vision Pipeline)

During real-time execution of the face detection, tracking, and recognition pipeline, the following resource metrics were observed via `psutil`:

*   **CPU (System-Wide)**: ~95% - 99%. Almost all 12 cores are highly saturated. ONNX Runtime parallelizes operations effectively across available CPU threads.
*   **CPU (Process)**: ~85% - 92% (normalized). The vision pipeline process itself is the primary consumer of compute resources.
*   **RAM**: ~3.0 GB / 3.5 GB (~85% utilization). Memory usage is high but stable.
*   **Swap**: ~0.5 GB / 11.0 GB (~4% utilization). The OS places some inactive pages in Swap space to keep the model weights readily available in fast memory.

## 3. Inference Latency Benchmarks 

A standalone test benchmarking the raw ONNX models over 20 runs yielded the following processing times:

| Model | Resolution | Latency (ms/frame) | Max FPS |
| :--- | :--- | :--- | :--- |
| **det_10g** (SCRFace Detection) | 640x640 | 298.0 ms | 3.4 FPS |
| **arcface** (Face Recognition) | 112x112 | 185.3 ms | 5.4 FPS |

*Note: Since the vision pipeline runs both models sequentially for every tracked frame containing a face, the cumulative latency for a frame where recognition triggers will be exceptionally high (>480 ms). Actual visual FPS hovers around 2–3 FPS during live tracking.*

## 4. Conclusion & Recommendations

The application operates fundamentally fine but visibly suffers from severe compute bottlenecks. CPU mode on an ARM architecture like the Jetson Orin is incapable of delivering smooth real-time performance (30+ FPS) for these models.

**Recommendations**:
1.  **Hardware Acceleration**: The most critical improvement is migrating from `CPUExecutionProvider` to `CUDAExecutionProvider` or ideally `TensorrtExecutionProvider`. This requires installing an ONNX Runtime wheel explicitly built for JetPack 6 (`onnxruntime-gpu`).
2.  **Model Quantization**: Using FP16 or INT8 versions of the SCRFD and ArcFace models would lighten CPU load if GPU acceleration remains unavailable.
3.  **Frame Skipping**: Until GPU acceleration is enabled, introducing a hard frame-skip in the tracker (e.g., only running detection every 3rd or 4th frame) will keep the pipeline more responsive.
