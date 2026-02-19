"""
Test Typhoon ASR transcription directly
"""
import sys
sys.path.insert(0, "src")

from stt.typhoon_asr_client import TyphoonASR
import soundfile as sf
import numpy as np

print("=" * 80)
print("🌪️  TYPHOON ASR DIRECT TEST")
print("=" * 80)

# Check audio file
audio_file = "microphone_test.wav"
print(f"\n📁 Analyzing: {audio_file}")

data, sr = sf.read(audio_file)
rms = np.sqrt(np.mean(data**2))

print(f"   Sample rate: {sr} Hz")
print(f"   Duration: {len(data)/sr:.2f}s")
print(f"   RMS (loudness): {rms:.6f}")
print(f"   Max amplitude: {data.max():.6f}")
print(f"   Min amplitude: {data.min():.6f}")

# Initialize Typhoon ASR
print("\n" + "=" * 80)
print("Loading Typhoon ASR model...")
print("=" * 80)

asr = TyphoonASR(
    model_name="scb10x/typhoon-asr-realtime",
    device="cpu",
    language="th"
)

# Transcribe
print("\n" + "=" * 80)
print("Transcribing...")
print("=" * 80)

result = asr.transcribe_audio(audio_file)

print("\n" + "=" * 80)
print("TRANSCRIPTION RESULT")
print("=" * 80)
print(f"Text: '{result['text']}'")
print(f"Confidence: {result['confidence']:.2%}")
print(f"Language: {result['language']}")
print(f"Duration: {result['duration']:.2f}s")

if result['text']:
    print(f"\n✅ SUCCESS: Transcribed {len(result['text'])} characters")
else:
    print("\n⚠️  WARNING: Empty transcription!")
    print("\nPossible reasons:")
    print("1. You spoke in English (model only understands Thai)")
    print("2. Audio is too quiet (try speaking louder)")
    print("3. Background noise interference")
    print("4. Microphone quality issue")
    print("\n💡 TIP: Try recording again and speak in THAI language clearly and loudly")

print("=" * 80)
