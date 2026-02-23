# Jetson Orin Performance Report: CPU vs GPU Vision Pipeline

This report compares the performance of the RobotAI Vision Pipeline
(`demo_realtime_visual_jetson.py`) when running on CPU-only versus GPU-accelerated
(CUDA) execution on the NVIDIA Jetson Orin (JetPack R36 / Linux 5.15 aarch64).

---

## 1. System Context

| Property | Value |
| :--- | :--- |
| **Platform** | NVIDIA Jetson Orin (aarch64) |
| **OS** | Linux 5.15.148-tegra (JetPack R36.4.7) |
| **CPU** | 12-core ARM Cortex-A78AE |
| **GPU** | NVIDIA Ampere iGPU (Jetson Orin), 624 MHz max |
| **Unified Memory** | 8 GB total (≈3.5 GB visible to OS; remainder is GPU carve-out) |
| **ONNX Runtime (CPU run)** | v1.23.2 — `CPUExecutionProvider` |
| **ONNX Runtime (GPU run)** | v1.23.0 — `CUDAExecutionProvider` (onnxruntime-gpu, JetPack 6 / CUDA 12.6 wheel) |

> **Note:** TensorRT (`TensorrtExecutionProvider`) was attempted but automatically
> fell back to CUDA due to GPU memory pressure during engine compilation. CUDA
> inference results below are therefore achieved **without** TensorRT's additional
> graph optimisation pass.

---

## 2. Live Pipeline Resource Utilisation

Metrics collected via `psutil` + Jetson sysfs (`/sys/class/devfreq/17000000.gpu/`)
during active face detection, tracking, and recognition.

| Metric | CPU Mode | GPU (CUDA) Mode |
| :--- | :--- | :--- |
| **CPU — System-Wide** | 95 – 99% (all 12 cores saturated) | **16 – 27%** |
| **CPU — Process (normalised)** | 85 – 92% | **9 – 12%** |
| **RAM Used** | ~3.0 / 3.5 GB (85%) | ~3.3 / 3.5 GB (95%) |
| **Swap Used** | ~0.5 GB (4%) | ~2.9 GB (25%) |
| **GPU Frequency** | 0 MHz (idle) | **624 / 624 MHz (100%)** |
| **ONNX Provider** | CPUExecutionProvider | CUDAExecutionProvider |

> **RAM note:** GPU mode shows higher RAM and Swap usage because CUDA runtime
> libraries and GPU driver buffers load into the shared unified memory pool.

---

## 3. Inference Latency & Frame-Rate

### 3.1 CPU Mode (standalone model benchmark, 20 runs each)

| Model | Input Size | Latency (ms/frame) | Max Theoretical FPS |
| :--- | :--- | :--- | :--- |
| **det_10g** (SCRFD Detection) | 640 × 640 | 298.0 ms | 3.4 FPS |
| **arcface** (Face Recognition) | 112 × 112 | 185.3 ms | 5.4 FPS |
| **Combined (sequential)** | — | > 480 ms | **~2 FPS live** |

### 3.2 GPU (CUDA) Mode (live pipeline measurement)

Frame timestamps from the live pipeline run on 2026-02-24:

| Metric | Observed Value |
| :--- | :--- |
| **Frame interval** | 125 – 145 ms |
| **Live FPS** | **~7 – 8 FPS** |
| **Detection latency (estimated)** | ≈ 50 – 80 ms |
| **Recognition latency (estimated)** | ≈ 30 – 50 ms |

---

## 4. Head-to-Head Summary

| Metric | CPU Mode | GPU (CUDA) Mode | Improvement |
| :--- | :--- | :--- | :--- |
| **Live FPS** | 2 – 3 FPS | **7 – 8 FPS** | **~3× faster** |
| **CPU Load (system)** | 95 – 99% | **16 – 27%** | **~75% lower** |
| **CPU Load (process)** | 85 – 92% | **9 – 12%** | **~82% lower** |
| **Thermal headroom** | Minimal (fully saturated) | Good (CPU stays cool) |  |
| **Startup time** | ~2 s | ~5 min (first run, TRT build) / ~5 s (cached) |  |

---

## 5. Installation Summary

The GPU acceleration was enabled by replacing the standard `onnxruntime` package
with the NVIDIA-provided JetPack 6 / CUDA 12.6 wheel:

```bash
# Inside the project virtualenv
pip uninstall -y onnxruntime
pip install onnxruntime-gpu==1.23.0 \
    --extra-index-url https://pypi.jetson-ai-lab.io/jp6/cu126
pip install "numpy<2"   # Required: onnxruntime-gpu was compiled against NumPy 1.x
```

Key code changes made to the vision models (`detector.py`, `recognizer.py`):
- All demos updated from `device="cpu"` → `device="cuda"`
- Tiered provider fallback: **TensorRT → CUDA → CPU** (auto-degrades gracefully)
- `trt_max_workspace_size: 512 MB` cap added to prevent TRT OOM on unified memory

---

## 6. Conclusions & Recommendations

The move from CPU-only to GPU-accelerated (`CUDAExecutionProvider`) inference
delivers meaningful real-world gains on the Jetson Orin:

- **~3× FPS improvement** (2–3 → 7–8 FPS) with no model changes
- **~75% reduction in CPU load**, freeing cores for voice, LLM, and OS tasks
- The system is now able to sustain smooth real-time tracking

### Further Improvement Options

| Approach | Expected Gain | Notes |
| :--- | :--- | :--- |
| **TensorRT (FP16)** | +2–4× over CUDA | Requires one-time engine build (~5–20 min); increase GPU carve-out in `/boot/extlinux/extlinux.conf` if OOM occurs |
| **Model Quantisation (INT8)** | +1.5–2× | Quantise SCRFD and ArcFace offline; slight accuracy trade-off |
| **Increase GPU Carve-out** | Enables TensorRT | Edit `mem=3500M` → `mem=2500M` in boot config to give GPU more unified memory |
| **Frame skipping** | Lower latency feel | Run detection every 2nd frame; tracker holds state in between |
