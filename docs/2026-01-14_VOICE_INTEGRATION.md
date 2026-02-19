# 🎙️ Voice Integration Guide

Complete Thai language voice interaction system for ROS2 Robot AI Brain.

## Overview

The voice system enables full spoken Thai language interaction with the robot:

```
User speaks (Thai) → Whisper STT → FastAPI → Typhoon LLM → VachanaTTS → Robot speaks (Thai)
```

---

## Components

### 1. **Whisper STT** (`stt/whisper_client.py`)
- **Model:** OpenAI Whisper (small model)
- **Language:** Thai (th)
- **Features:**
  - Real-time audio transcription
  - Confidence scoring
  - Multiple audio format support (wav, mp3, m4a, etc.)
  - GPU/CPU auto-detection

### 2. **VachanaTTS** (`tts/vachana_client.py`)
- **Model:** VITS-based Thai TTS
- **Source:** VachanaTTS (MMS-TTS-THA)
- **Features:**
  - Natural Thai speech synthesis
  - Adjustable speaking rate
  - Multiple voice models support
  - High-quality audio output

### 3. **FastAPI Endpoints** (`api/main.py`)
- `POST /audio/transcribe` - STT endpoint
- `POST /audio/synthesize` - TTS endpoint
- `POST /voice/interact` - Complete voice loop

### 4. **Audio Utilities** (`utils/audio_utils.py`)
- Microphone recording
- Audio playback
- Format conversion
- Silence detection

---

## Installation

### 1. Install Required Packages

```bash
# Core dependencies
pip install openai-whisper sounddevice soundfile scipy

# Optional for better resampling
pip install librosa
```

### 2. Download VachanaTTS Models

1. Visit [VIZINTZOR on HuggingFace](https://huggingface.co/VIZINTZOR)
2. Download a Thai TTS model
3. Place in: `C:\Users\Win 10 Pro\Desktop\VachanaTTS\models\`

Example models:
- `MMS-TTS-THAI-MALEV1` - Male voice
- `MMS-TTS-THAI-FEMALEV1` - Female voice

---

## Quick Start

### Test Individual Components

#### Test STT Only
```bash
python demo_voice.py stt
```
Records 5 seconds and shows transcription.

#### Test TTS Only
```bash
python demo_voice.py tts
```
Synthesizes and plays Thai text.

### Full Voice Interaction

```bash
# 1. Start API server (in terminal 1)
python api/main.py

# 2. Run voice demo (in terminal 2)
python demo_voice.py full
```

---

## API Usage

### 1. Transcribe Audio (STT)

**Endpoint:** `POST /audio/transcribe`

```python
import requests

with open("audio.wav", "rb") as f:
    files = {"audio": f}
    response = requests.post(
        "http://localhost:8000/audio/transcribe",
        files=files
    )

result = response.json()
print(result["text"])        # Transcribed Thai text
print(result["confidence"])  # 0.0-1.0
```

**Response:**
```json
{
  "status": "success",
  "text": "สวัสดีครับ ห้องสมุดอยู่ที่ไหน",
  "confidence": 0.92,
  "language": "th",
  "duration": 2.5
}
```

---

### 2. Synthesize Speech (TTS)

**Endpoint:** `POST /audio/synthesize?text=<thai_text>`

```python
import requests

response = requests.post(
    "http://localhost:8000/audio/synthesize",
    params={"text": "สวัสดีครับ ยินดีต้อนรับ"}
)

# Save audio file
with open("response.wav", "wb") as f:
    f.write(response.content)
```

**Response:** Audio file (wav format)

**Headers:**
- `X-Text`: Original text
- `X-Duration`: Audio duration

---

### 3. Complete Voice Interaction

**Endpoint:** `POST /voice/interact?session_id=<id>`

```python
import requests

with open("user_question.wav", "rb") as f:
    files = {"audio": f}
    response = requests.post(
        "http://localhost:8000/voice/interact",
        files=files,
        params={"session_id": "session_001"}
    )

# Get AI response as audio
with open("ai_response.wav", "wb") as f:
    f.write(response.content)

# Check headers for metadata
print(response.headers["X-User-Text"])      # What user said
print(response.headers["X-Response-Text"])  # AI's response
print(response.headers["X-Intent"])         # conversation/navigation
```

---

## Python API

### Using WhisperSTT

```python
from stt.whisper_client import WhisperSTT

# Initialize
stt = WhisperSTT(model_size="small", language="th")

# Transcribe file
result = stt.transcribe_audio("audio.wav")
print(result["text"])
print(result["confidence"])

# Transcribe numpy array
import numpy as np
audio_data = np.random.randn(16000)  # 1 second at 16kHz
result = stt.transcribe_numpy(audio_data, sample_rate=16000)
```

### Using VachanaTTS

```python
from tts.vachana_client import VachanaTTS

# Initialize
tts = VachanaTTS()

# Synthesize
audio_file, metadata = tts.synthesize("สวัสดีครับ")
print(f"Audio: {audio_file}")
print(f"Duration: {metadata['duration']}s")

# Batch synthesis
texts = ["สวัสดี", "ยินดีต้อนรับ", "ขอบคุณ"]
results = tts.synthesize_batch(texts)
```

### Using Audio Utilities

```python
from utils.audio_utils import (
    record_audio, 
    play_audio_file,
    record_until_silence
)

# Record fixed duration
audio, sr = record_audio(duration=5.0)

# Record until silence
audio, sr = record_until_silence(
    silence_threshold=0.01,
    silence_duration=2.0
)

# Play audio
play_audio_file("response.wav")
```

---

## Voice Bot Class

Complete voice interaction bot:

```python
from demo_voice import VoiceBot

# Initialize
bot = VoiceBot(
    api_url="http://localhost:8000",
    session_id="robot_001"
)

# Single interaction
bot.interact_once(duration=5.0)

# Continuous mode
bot.continuous_mode(recording_duration=5.0)
```

---

## Configuration

### STT Settings (`config/settings.yaml`)

```yaml
stt:
  model: "small"              # tiny, base, small, medium, large
  language: "th"              # Thai
  confidence_threshold: 0.7   # Minimum confidence
```

### TTS Settings

Configure in VachanaTTS:
- Model selection
- Speaking rate (0.1-2.0)
- Voice cloning (optional)

---

## Performance

### Typical Latencies

| Component | Time | Hardware |
|-----------|------|----------|
| Recording (5s) | 5.0s | - |
| STT (Whisper small) | 2-4s | GPU |
| STT (Whisper small) | 5-10s | CPU |
| API Processing | 3-8s | - |
| TTS Synthesis | 1-3s | GPU |
| TTS Synthesis | 2-5s | CPU |
| **Total (GPU)** | **~15-20s** | GPU |
| **Total (CPU)** | **~20-30s** | CPU |

### Optimization Tips

1. **Use GPU:** Significantly faster for both STT and TTS
2. **Reduce recording duration:** Use silence detection
3. **Cache common responses:** Pre-generate frequent replies
4. **Use smaller Whisper model:** `tiny` or `base` for speed
5. **Batch processing:** Process multiple requests together

---

## Troubleshooting

### STT Issues

**"Whisper model not found"**
```bash
# Models auto-download on first use
# If fails, manually download:
python -c "import whisper; whisper.load_model('small')"
```

**"No speech detected"**
- Check microphone connection
- Adjust recording volume
- Use `list_audio_devices()` to verify device

**Low confidence scores**
- Reduce background noise
- Speak clearer
- Use better microphone
- Try larger Whisper model

### TTS Issues

**"VachanaTTS not available"**
- Check VachanaTTS path in `tts/vachana_client.py`
- Verify models are downloaded
- Test VachanaTTS separately: `python C:\Users\Win 10 Pro\Desktop\VachanaTTS\app-th.py`

**"No TTS models found"**
- Download models from HuggingFace
- Place in `VachanaTTS/models/` directory
- Check model format (should have config.json and model files)

### Audio Issues

**"No default input device"**
```python
from utils.audio_utils import list_audio_devices
list_audio_devices()  # See available devices
```

**"Device not found"**
- Install audio drivers
- Check Windows sound settings
- Try different device index

---

## Integration with ROS2

### Publish Speech Events

```python
# In ROS2 node
def on_speech_detected(audio_data):
    # Transcribe
    result = stt.transcribe_numpy(audio_data)
    
    # Publish to /robot/speech/text topic
    msg = String()
    msg.data = result["text"]
    publisher.publish(msg)
```

### Subscribe to TTS Requests

```python
# In ROS2 node
def tts_callback(msg):
    # Synthesize
    audio_file, _ = tts.synthesize(msg.data)
    
    # Play through robot speakers
    play_audio_file(audio_file)
```

---

## Examples

### Example 1: Simple Voice Query

```
User speaks: "ห้องสมุดอยู่ที่ไหนครับ"
              ↓ (STT)
Text: "ห้องสมุดอยู่ที่ไหนครับ" (confidence: 0.92)
              ↓ (API)
Response: "ห้องสมุดอยู่ที่อาคาร 5 ชั้น 2 ครับ"
Intent: "conversation"
              ↓ (TTS)
Robot speaks: "ห้องสมุดอยู่ที่อาคาร 5 ชั้น 2 ครับ"
```

### Example 2: Navigation Request

```
User speaks: "พาฉันไปโรงอาหารหน่อย"
              ↓ (STT)
Text: "พาฉันไปโรงอาหารหน่อย" (confidence: 0.89)
              ↓ (API)
Response: "แน่นอนครับ เดี๋ยวพาไปครับ"
Intent: "navigation"
Navigation: {"target_location": "โรงอาหารกลาง"}
              ↓ (TTS + ROS2)
Robot speaks + moves to cafeteria
```

### Example 3: Low Confidence Handling

```
User speaks: [unclear/noisy audio]
              ↓ (STT)
Text: "..." (confidence: 0.45)
              ↓ (API)
Response: "ขอโทษค่ะ ฉันไม่ค่อยได้ยินชัดเจน ช่วยพูดอีกครั้งได้ไหมคะ"
Intent: "clarification"
              ↓ (TTS)
Robot asks for repeat
```

---

## Next Steps

1. **Integrate with ROS2:**
   - Create ROS2 bridge node
   - Subscribe to microphone topic
   - Publish to speaker topic

2. **Add Real-time STT:**
   - Stream audio chunks
   - Process in real-time
   - Lower latency

3. **Voice Activity Detection:**
   - Auto-detect when user speaks
   - No manual recording trigger
   - Continuous listening mode

4. **Multi-speaker Support:**
   - Speaker diarization
   - Different voices for different contexts
   - Emotion in speech

5. **Improve Latency:**
   - Use streaming APIs
   - Implement audio buffering
   - Optimize model loading

---

## Resources

### Models & Downloads
- **Whisper:** https://github.com/openai/whisper
- **VachanaTTS:** https://github.com/VYNCX/VachanaTTS
- **VITS Models:** https://huggingface.co/VIZINTZOR

### Documentation
- **Whisper Docs:** https://github.com/openai/whisper#available-models-and-languages
- **Thai TTS:** https://huggingface.co/facebook/mms-tts-tha
- **Audio Processing:** https://python-sounddevice.readthedocs.io/

---

## Credits

- **STT:** OpenAI Whisper
- **TTS:** VachanaTTS (VYNCX) + MMS-TTS-THA (Facebook)
- **Audio:** sounddevice, soundfile, scipy
- **Integration:** Custom FastAPI endpoints

---

**Version:** 1.0.0  
**Last Updated:** 2026-01-14  
**Status:** ✅ Fully Implemented
