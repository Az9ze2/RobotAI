# Typhoon ASR Testing Guide

## Overview
This test suite integrates the Typhoon ASR model into the RobotAI project and provides comprehensive testing with and without grammar correction.

## Files Created

### 1. STT Client
- **`src/stt/typhoon_asr_client.py`** - Typhoon ASR client with similar interface to Whisper

### 2. Utilities
- **`src/utils/audio_recorder.py`** - Audio recording utility for microphone input

### 3. Tests
- **`tests/test_typhoon_asr.py`** - Unit tests for both modes:
  - Test 1: Transcription WITHOUT grammar checker
  - Test 2: Transcription WITH grammar checker

### 4. Configuration
- **`config/local.yaml`** - Updated with Typhoon ASR settings
- **`requirements-typhoon.txt`** - Typhoon ASR dependencies

## Installation

### Step 1: Install Dependencies

```bash
cd "c:\Users\Win 10 Pro\Desktop\RobotAI"
pip install -r requirements-typhoon.txt
```

**Note:** The Typhoon ASR model requires NeMo toolkit which may have Windows compatibility issues. If installation fails, you may need to use WSL or Linux.

### Step 2: Verify Installation

```bash
python -c "import nemo.collections.asr as nemo_asr; print('NeMo ASR installed successfully!')"
```

## Running Tests

### Option 1: Run Both Tests
```bash
python -m pytest tests/test_typhoon_asr.py -v -s
```

### Option 2: Run Individual Tests

**Test 1: Without Grammar Checker**
```bash
python -m pytest tests/test_typhoon_asr.py::TestTyphoonASR::test_transcribe_no_grammar -v -s
```

**Test 2: With Grammar Checker**
```bash
python -m pytest tests/test_typhoon_asr.py::TestTyphoonASR::test_transcribe_with_grammar -v -s
```

### Option 3: Run Directly (Interactive)
```bash
python tests/test_typhoon_asr.py
```

## Test Workflow

Each test follows this workflow:

1. **Audio Recording** (5 seconds)
   - 3-second countdown
   - Records from default microphone
   - Saves to temporary WAV file

2. **Transcription**
   - Loads Typhoon ASR model
   - Transcribes audio to Thai text
   - Calculates confidence and timing metrics

3. **Grammar Correction** (Test 2 only)
   - Applies dictionary-based corrections
   - Applies LLM-based corrections (if available)
   - Shows before/after comparison

4. **Results Display**
   - Transcribed text
   - Confidence score
   - Processing time
   - RTF (Real-Time Factor)
   - Corrections applied (Test 2)

## Expected Output

### Test 1: Without Grammar Checker
```
🧪 TEST 1: TRANSCRIPTION WITHOUT GRAMMAR CHECKER
================================================================================

📹 Step 1: Recording audio from microphone
--------------------------------------------------------------------------------
🎤 Recording will start in...
   3...
   2...
   1...
   🔴 RECORDING NOW! Speak into your microphone...
✅ Recording complete! (5.0s)

🎙️  Step 2: Transcribing with Typhoon ASR
--------------------------------------------------------------------------------

================================================================================
📊 RESULTS (WITHOUT GRAMMAR CORRECTION)
================================================================================
✅ Transcription: 'สวัสดีครับ ผมต้องการไปห้องแลป'
📈 Confidence: 85.00%
🌐 Language: th
⏱️  Processing Time: 1.23s
🚀 RTF (Real-Time Factor): 0.246x
   ✨ Real-time capable!
================================================================================
```

### Test 2: With Grammar Checker
```
🧪 TEST 2: TRANSCRIPTION WITH GRAMMAR CHECKER
================================================================================

📹 Step 1: Recording audio from microphone
--------------------------------------------------------------------------------
[Recording process...]

🎙️  Step 2: Transcribing with Typhoon ASR
--------------------------------------------------------------------------------
✅ Raw transcription: 'สวัสดีครับ ผมต้องการไปห้องแหลก'

📝 Step 3: Applying grammar correction
--------------------------------------------------------------------------------

================================================================================
📊 RESULTS (WITH GRAMMAR CORRECTION)
================================================================================
🎤 Original Text: 'สวัสดีครับ ผมต้องการไปห้องแหลก'
✨ Corrected Text: 'สวัสดีครับ ผมต้องการไปห้องแลป'
📈 Confidence: 85.00%
🌐 Language: th
⏱️  Transcription Time: 1.23s
⏱️  Correction Time: 0.15s
⏱️  Total Time: 1.38s
🚀 RTF (Real-Time Factor): 0.246x

🔧 Corrections Applied: 1
--------------------------------------------------------------------------------
   1. Dictionary: 'แหลก' → 'แลป'
================================================================================
```

## Troubleshooting

### Issue: NeMo toolkit installation fails on Windows
**Solution:** The Typhoon ASR README states Windows is not officially supported. Try:
1. Use WSL (Windows Subsystem for Linux)
2. Use Docker
3. Use a Linux VM

### Issue: No microphone detected
**Solution:** 
```bash
python -c "from src.utils.audio_recorder import AudioRecorder; AudioRecorder().list_devices()"
```

### Issue: CUDA out of memory
**Solution:** Change device to CPU in `config/local.yaml`:
```yaml
stt:
  typhoon:
    device: "cpu"
```

### Issue: LLM not available for grammar correction
**Solution:** Make sure Ollama is running:
```bash
# Check if Ollama is running
curl http://localhost:11434/api/tags
```

## Configuration

Edit `config/local.yaml` to switch between Whisper and Typhoon ASR:

```yaml
stt:
  provider: "typhoon"  # Change to "whisper" to use Whisper instead
  
  typhoon:
    model: "scb10x/typhoon-asr-realtime"
    device: "auto"  # "auto", "cpu", or "cuda"
    language: "th"
```

## Performance Metrics

- **RTF (Real-Time Factor)**: Processing time / Audio duration
  - RTF < 1.0 = Real-time capable (can process faster than real-time)
  - RTF > 1.0 = Batch processing (slower than real-time)

## Notes

- Both tests record **5 seconds** of audio from your microphone
- Make sure your microphone is working before running tests
- The grammar checker uses both dictionary and LLM-based corrections
- Temporary audio files are automatically cleaned up after tests
