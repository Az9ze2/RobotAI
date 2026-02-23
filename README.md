# 🤖 RobotAI - End-to-End Vision + Voice Interaction System

**An intelligent Thai-language conversational AI system combining computer vision, speech recognition, and natural language processing for personalized student interactions.**

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [System Architecture](#system-architecture)
- [Quick Start](#quick-start)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Models](#models)
- [Configuration](#configuration)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [License](#license)

---

## 🎯 Overview

RobotAI is a complete end-to-end interaction pipeline designed for KMITL (King Mongkut's Institute of Technology Ladkrabang) that enables natural, personalized conversations with students in Thai language. The system:

1. **Detects and recognizes** students using computer vision
2. **Listens** to their questions using speech-to-text
3. **Understands context** and generates personalized responses
4. **Responds** with natural Thai voice synthesis

### Key Highlights

- ✅ **Real-time face detection and tracking** with SCRFD and ByteTrack
- ✅ **Gaze detection** to trigger interactions only when looking at camera
- ✅ **Face recognition** with ArcFace for student identification
- ✅ **Automatic silence detection** for natural voice input
- ✅ **Thai language STT** using Typhoon ASR
- ✅ **Student-aware LLM responses** with Qwen 2.5
- ✅ **Natural Thai TTS** using VachanaTTS (MMS)
- ✅ **Year-based personalization** (addressing students by their academic year)

---

## ✨ Features

### Vision Pipeline
- **Face Detection**: SCRFD 10G model for accurate real-time detection
- **Face Tracking**: ByteTrack for multi-face tracking with unique IDs
- **Head Pose Estimation**: Detects if person is looking at camera
- **Face Recognition**: ArcFace R100 for student identification
- **Recognition Trigger**: Smart triggering with cooldown and gaze validation

### Voice Pipeline
- **Speech-to-Text**: Typhoon ASR for Thai language transcription
- **Silence Detection**: Automatic cutoff after 3 seconds of silence
- **Text-to-Speech**: VachanaTTS with MMS Thai voice
- **Natural Audio**: High-quality Thai voice synthesis

### Intelligence
- **LLM Integration**: Qwen 2.5 (7B) for Thai language understanding
- **Context Management**: Student-aware responses with name and year
- **Personalization**: Year-based greetings (น้องปี 1, พี่ปี 4, etc.)
- **Conversation Memory**: Context builder for multi-turn conversations

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     CAMERA FEED INPUT                           │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                   VISION PIPELINE                               │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐       │
│  │  SCRFD   │─▶│ByteTrack │─▶│Head Pose │─▶│ ArcFace  │       │
│  │Detection │  │ Tracking │  │Estimation│  │Recognition│       │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘       │
└────────────────────────┬────────────────────────────────────────┘
                         │ Student Identified
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                   VOICE INPUT (STT)                             │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐                     │
│  │Microphone│─▶│ Silence  │─▶│ Typhoon  │                     │
│  │  Input   │  │Detection │  │   ASR    │                     │
│  └──────────┘  └──────────┘  └──────────┘                     │
└────────────────────────┬────────────────────────────────────────┘
                         │ Transcribed Text
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                   LLM PROCESSING                                │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐                     │
│  │ Context  │─▶│  Qwen    │─▶│ Response │                     │
│  │ Builder  │  │   2.5    │  │Generation│                     │
│  └──────────┘  └──────────┘  └──────────┘                     │
└────────────────────────┬────────────────────────────────────────┘
                         │ Thai Response
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                   VOICE OUTPUT (TTS)                            │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐                     │
│  │VachanaTTS│─▶│   Audio  │─▶│ Speaker  │                     │
│  │   MMS    │  │Generation│  │  Output  │                     │
│  └──────────┘  └──────────┘  └──────────┘                     │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🚀 Quick Start

### Prerequisites

- Python 3.12+
- Webcam
- Microphone and speakers
- NVIDIA GPU (recommended) or CPU
- Ollama installed and running

### Installation (5 minutes)

```bash
# 1. Clone the repository
git clone https://github.com/yourusername/RobotAI.git
cd RobotAI

# 2. Install dependencies (have requirements.txt inside already)
chmod +x setup.sh
./setup.sh

# 3. Start Milvus
cd ~/milvus
sudo docker compose up -d

# 4. Enroll a student
python demos/demo_enrollment.py

# 5. Run the end-to-end demo
python demos/demo_end_to_end.py
```

### First Interaction

1. **Look at the camera** for 2-3 seconds
2. **Wait for recognition** - your name will appear
3. **Speak in Thai** - microphone opens automatically
4. **Wait for response** - system stops after 3s of silence
5. **Listen to answer** - Thai voice response plays

---

## 📦 Installation

### Step 1: System Requirements

- **OS**: Windows 10/11, Linux, or macOS
- **Python**: 3.12 or higher
- **RAM**: 8GB minimum, 16GB recommended
- **GPU**: NVIDIA GPU with 6GB+ VRAM (optional, for faster inference)
- **Storage**: 10GB for models and dependencies

### Step 2: Python Environment

```bash
# Create virtual environment (recommended)
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate
```


### Step 3: Install Dependencies

```bash
# edit setup.sh at line 27 "VACHANA_DIR"to your folder path
chmod +x setup.sh
./setup.sh
```
### Step 4: Download Models

#### Face Detection (SCRFD 10G)
```bash
# Download from InsightFace model zoo
# this git repo have model already
# Place in: models/buffalo_l/det_10g.onnx

```

#### Face Recognition (ArcFace R100)
```bash
# Download from InsightFace model zoo
# this git repo have model already
# Place in: models/arcface_r100_v1_fp16.onnx
pip install -U insightface
```

#### LLM (Qwen 2.5)
```bash
# Install Ollama from https://ollama.ai
#Already installed by setup.sh
```

#### STT (Typhoon ASR)
```bash
# Automatically downloaded on first run
# Model: scb10x/typhoon-asr-realtime
```

#### TTS (VachanaTTS MMS)
```bash
# download zip model from ..., then unzip and move that folder to /RobotAI/vachanatts/models
# https://huggingface.co/VIZINTZOR/MMS-TTS-THAI-MALEV1
# Model: MMS-TTS-THAI-MALEV1
```

### Step 5: Configuration

Edit `config/settings.yaml`:

```yaml
llm:
  api_url: "http://localhost:11434"  # Ollama API
  model: "qwen2.5:7b-instruct"

vision:
  detector_model: "models/buffalo_l/det_10g.onnx"
  recognizer_model: "models/arcface_r100_v1_fp16.onnx"
  confidence_threshold: 0.5

stt:
  model: "scb10x/typhoon-asr-realtime"
  language: "th"

tts:
  model: "MMS-TTS-THAI-MALEV1"
  voice: "male"
```

---

## 🎮 Usage

### 1. Enroll Students

```bash
python demos/demo_enrollment.py
```

- Enter student ID (format: YYXXXXXX, e.g., 65011356)
- Enter student name (Thai and English)
- Look at camera from multiple angles
- System captures 5 face images
- Enrollment saved to `database/dataset/registration.json`

### 2. Run End-to-End Demo

```bash
python demos/demo_end_to_end.py
```

**Controls:**
- `q` - Quit application
- `r` - Reset recognition

**On-Screen Display:**
- FPS counter (top-left)
- Student info (top-left, cyan)
- Head pose angles (above face, yellow)
- Student name or "Unknown" (below face)
- Status message (bottom-left, color-coded)

### 3. Run Vision-Only Demo

```bash
python demos/demo_realtime_visual.py
```

Tests vision pipeline without voice interaction.

---

## 📁 Project Structure

```
RobotAI/
├── demos/                      # Demo scripts
│   ├── demo_end_to_end.py     # Main end-to-end pipeline
│   ├── demo_enrollment.py     # Student enrollment tool
│   └── demo_realtime_visual.py # Vision pipeline demo
│
├── src/                        # Source code
│   ├── vision/                # Vision components
│   │   ├── detector.py        # SCRFD face detector
│   │   ├── tracker.py         # ByteTrack tracker
│   │   ├── head_pose.py       # Head pose estimator
│   │   ├── recognizer.py      # ArcFace recognizer
│   │   ├── database.py        # Enrollment database
│   │   └── recognition_trigger.py  # Recognition trigger logic
│   │
│   ├── stt/                   # Speech-to-text
│   │   └── typhoon_asr_client.py  # Typhoon ASR client
│   │
│   ├── tts/                   # Text-to-speech
│   │   └── vachana_client.py  # VachanaTTS client
│   │
│   ├── llm/                   # LLM clients
│   │   └── ollama_client.py   # Ollama API client
│   │
│   └── mcp/                   # Context management
│       └── context_builder.py # Context builder
│
├── models/                     # Model files
│   ├── buffalo_l/             # SCRFD detector
│   │   └── det_10g.onnx
│   └── arcface_r100_v1_fp16.onnx  # Face recognition
│
├── data/                       # Data files
│   └── enrollments.json       # Student enrollments
│
├── config/                     # Configuration
│   ├── settings.yaml          # Main settings
│   ├── local.yaml             # Local overrides
│   └── prod.yaml              # Production settings
│
├── scripts/                    # Utility scripts
│   ├── start_server.bat       # Start API server
│   └── start_milvus.bat       # Start Milvus (optional)
│
├── tests/                      # Tests
│   ├── test_vision.py         # Vision tests
│   ├── test_stt.py            # STT tests
│   └── test_integration.py    # Integration tests
│
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

---

## 🤖 Models

### Face Detection: SCRFD 10G
- **Purpose**: Detect faces in real-time
- **Input**: 640x640 RGB image
- **Output**: Bounding boxes, landmarks, confidence scores
- **Performance**: ~30ms per frame (GPU)
- **Download**: [InsightFace Model Zoo](https://github.com/deepinsight/insightface/tree/master/model_zoo)

### Face Recognition: ArcFace R100
- **Purpose**: Extract face embeddings for recognition
- **Input**: 112x112 aligned face
- **Output**: 512-D normalized embedding
- **Performance**: ~100ms per face (GPU)
- **Download**: [InsightFace Model Zoo](https://github.com/deepinsight/insightface/tree/master/model_zoo)

### Speech-to-Text: Typhoon ASR
- **Purpose**: Transcribe Thai speech to text
- **Model**: scb10x/typhoon-asr-realtime
- **Language**: Thai
- **Performance**: ~0.2s for 5s audio
- **Auto-download**: Yes (on first run)

### Text-to-Speech: VachanaTTS (MMS)
- **Purpose**: Generate Thai speech from text
- **Model**: MMS-TTS-THAI-MALEV1
- **Voice**: Male Thai voice
- **Performance**: ~2.5s for typical response
- **Auto-download**: Yes (on first run)

### LLM: Qwen 2.5 (7B Instruct)
- **Purpose**: Generate Thai language responses
- **Model**: qwen2.5:7b-instruct
- **Context**: 32K tokens
- **Performance**: ~4.3s per response (GPU)
- **Installation**: `ollama pull qwen2.5:7b-instruct`

---

## ⚙️ Configuration

### Adjust Silence Detection

Edit `demos/demo_end_to_end.py`:

```python
audio_data, duration = self.record_with_silence_detection(
    silence_threshold=0.02,  # Lower = more sensitive
    silence_duration=3.0,    # Seconds before stopping
    max_duration=30.0        # Maximum recording time
)
```

### Adjust Gaze Thresholds

```python
self.head_pose = HeadPoseEstimator(
    yaw_threshold=25,      # ±25° left/right
    pitch_threshold=15,    # ±15° up/down
    roll_threshold=30      # ±30° head tilt
)
```

### Adjust Recognition Threshold

```python
student_id, similarity, student_name = self.db.recognize(
    embedding, threshold=0.4  # 0.3-0.5 recommended
)
```

---

## 🔧 Troubleshooting

### Camera Not Opening

```bash
# Test camera
python -c "import cv2; cap = cv2.VideoCapture(0); print(cap.isOpened())"
```

### No Face Detection

- Check lighting (need good illumination)
- Ensure `models/buffalo_l/det_10g.onnx` exists
- Face should be clearly visible and not too small

### Recognition Not Triggering

- Look directly at camera for 2-3 seconds
- Check if face is enrolled in `data/enrollments.json`
- Verify gaze thresholds aren't too strict

### STT Not Working

```bash
# Test STT
python -c "from stt.typhoon_asr_client import TyphoonASR; print('OK')"
```

### LLM Not Responding

```bash
# Ensure Ollama running
ollama serve

# Check model
ollama list | findstr qwen2.5
```

### Slow Startup

- VachanaTTS imports take 30-60 seconds (one-time cost)
- After initialization, pipeline runs smoothly

---

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

---

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## 🙏 Acknowledgments

- **KMITL** - King Mongkut's Institute of Technology Ladkrabang
- **InsightFace** - Face detection and recognition models
- **SCB 10X** - Typhoon ASR model
- **VISTEC** - VachanaTTS
- **Alibaba Cloud** - Qwen 2.5 LLM

---

## 📞 Contact

For questions or support, please open an issue on GitHub.

---

**Made with ❤️ for KMITL students**
