# Pi 5 vs Jetson Orin Performance Report: Vision Pipeline

This report compares the performance of the RobotAI Vision Pipeline
(`demo_realtime_visual_jetson.py`) when running on two separate ARM-based
hardware platforms: the **Raspberry Pi 5** (tested 2026-02-28) and the
**NVIDIA Jetson Orin** (previously benchmarked, as documented in
`jetson_cpu_vs_gpu_report.md`).

---

## 1. System Context

| Property | Raspberry Pi 5 | Jetson Orin (CPU) | Jetson Orin (GPU/CUDA) |
| :--- | :--- | :--- | :--- |
| **Platform** | Raspberry Pi 5 (aarch64) | NVIDIA Jetson Orin (aarch64) | NVIDIA Jetson Orin (aarch64) |
| **OS** | Linux (aarch64) | Linux 5.15.148-tegra (JetPack R36.4.7) | Linux 5.15.148-tegra (JetPack R36.4.7) |
| **CPU** | 4-core ARM Cortex-A76 | 12-core ARM Cortex-A78AE | 12-core ARM Cortex-A78AE |
| **GPU** | VideoCore VII (no CUDA) | NVIDIA Ampere iGPU, 624 MHz max | NVIDIA Ampere iGPU, 624 MHz max |
| **RAM** | ~3.9 GB visible to OS | ~3.5 GB visible to OS | ~3.5 GB visible to OS |
| **ONNX Provider** | `CPUExecutionProvider` | `CPUExecutionProvider` | `CUDAExecutionProvider` |
| **Models Used** | `det_10g.onnx`, `arcface_r100_v1_fp16.onnx` | Same | Same |

> **Note:** The Raspberry Pi 5 has no CUDA-capable GPU. The VideoCore VII GPU
> is not supported by ONNX Runtime, so all inference runs on CPU only.

---

## 2. Live Pipeline Resource Utilisation

Metrics were collected via `demo_realtime_visual_jetson.py` (which includes
live `psutil` logging) during active face-tracking in a real webcam session on
the Pi 5, and from previous Jetson benchmarks.

| Metric | Pi 5 (CPU) | Jetson Orin (CPU) | Jetson Orin (GPU/CUDA) |
| :--- | :--- | :--- | :--- |
| **CPU — System-Wide** | **84 – 95%** (4 cores) | 95 – 99% (12 cores) | 16 – 27% (12 cores) |
| **CPU — Process (normalised)** | **67 – 87%** | 85 – 92% | 9 – 12% |
| **RAM Used** | ~3.7 – 3.8 / 3.9 GB (**96%**) | ~3.0 / 3.5 GB (85%) | ~3.3 / 3.5 GB (95%) |
| **Swap Used** | ~0.0 GB (0%) | ~0.5 GB (4%) | ~2.9 GB (25%) |
| **GPU Frequency** | 0 MHz (no CUDA) | 0 MHz (idle) | 624 / 624 MHz (100%) |
| **ONNX Provider** | CPUExecutionProvider | CPUExecutionProvider | CUDAExecutionProvider |

---

## 3. Inference Latency & Frame Rate

### 3.1 Pi 5 — Live Pipeline Measurement (2026-02-28)

Observed from `demo_realtime_visual_jetson.py` frame timestamps during a live
webcam session with 1 face tracked:

| Metric | Observed Value |
| :--- | :--- |
| **Frame interval** | ~475 – 680 ms |
| **Live FPS (detection only)** | **~1.5 – 2.0 FPS** |
| **Live FPS (detection + recognition on trigger)** | **< 1 FPS** during recognition events |
| **Recognition result** | ✓ Correct (Krittin Sakharin matched, similarity 0.49 – 0.53) |

### 3.2 Jetson Orin — CPU Mode (standalone benchmark, 20 runs)

| Model | Input Size | Latency (ms/frame) | Max FPS |
| :--- | :--- | :--- | :--- |
| **det_10g** (SCRFD Detection) | 640 × 640 | 298.0 ms | 3.4 FPS |
| **arcface** (Face Recognition) | 112 × 112 | 185.3 ms | 5.4 FPS |
| **Combined (sequential)** | — | > 480 ms | **~2 FPS live** |

### 3.3 Jetson Orin — GPU (CUDA) Mode (live pipeline)

| Metric | Observed Value |
| :--- | :--- |
| **Frame interval** | 125 – 145 ms |
| **Live FPS** | **~7 – 8 FPS** |

---

## 4. Head-to-Head Summary

| Metric | Pi 5 (CPU only) | Jetson Orin (CPU) | Jetson Orin (GPU/CUDA) |
| :--- | :--- | :--- | :--- |
| **Live FPS (tracking)** | **~1.5 – 2 FPS** | ~2 – 3 FPS | **~7 – 8 FPS** |
| **CPU Load (system)** | 84 – 95% (4 cores) | 95 – 99% (12 cores) | 16 – 27% (12 cores) |
| **RAM Pressure** | Very high (96%) | High (85%) | High (95%) |
| **GPU Acceleration** | ❌ Not available | ❌ Not used | ✅ CUDA |
| **Recognition Accuracy** | ✓ Correct matches | ✓ Correct matches | ✓ Correct matches |
| **Thermal Headroom** | Minimal (4 cores saturated) | Minimal (12 cores saturated) | Good (CPU stays cool) |

---

## 5. Conclusions & Hardware Fit for Vision

### Raspberry Pi 5
- **Not suitable** for real-time face recognition at 30 FPS.
- The 4-core Cortex-A76 CPU is **slower per-model than the Jetson Orin's 12 cores**, resulting in ~1.5–2 FPS live — barely usable for tracking.
- RAM is nearly fully consumed (~96%), leaving almost no headroom for other system processes (LLM, TTS, ASR).
- No CUDA GPU available — cannot accelerate ONNX models beyond CPU.
- **Viable only** for very lightweight models (e.g., SCRFD 500M, MobileNet) or heavily quantised INT8 models with frame skipping.

### Jetson Orin — CPU Mode
- Similar pipeline FPS (~2–3 FPS) to the Pi 5, but **with 12 cores to distribute load**, CPU saturation is better managed.
- **Not recommended** for real-time vision in production for the same reasons as Pi 5 CPU mode.

### Jetson Orin — GPU (CUDA) Mode ✅ Recommended
- **~3–4× faster** than Pi 5 and Jetson CPU (~7–8 FPS vs 1.5–2 FPS).
- **~75% lower CPU usage**, freeing cores for LLM, ASR, and TTS processes.
- The only configuration of the tested hardware capable of approaching smooth real-time tracking.
- With TensorRT (FP16), an additional 2–4× gain is possible (est. **~15–25 FPS**).

---

## 6. Recommendations

| Hardware | Verdict | Notes |
| :--- | :--- | :--- |
| **Raspberry Pi 5** | ❌ Not Recommended for Vision | Too slow, insufficient RAM headroom, no CUDA. Use only for lightweight prototyping. |
| **Jetson Orin (CPU)** | ⚠️ Functional but Insufficient | Acceptable only with frame-skipping and lightweight models. |
| **Jetson Orin (CUDA)** | ✅ Best Fit | Recommended production target. Enable with `onnxruntime-gpu` (JetPack 6 wheel). |
| **Jetson Orin (TensorRT)** | ✅✅ Optimal | Further 2–4× gain possible; requires one-time engine build (~5–20 min). |
