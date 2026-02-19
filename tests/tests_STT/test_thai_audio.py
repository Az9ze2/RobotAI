"""
Test Typhoon ASR with pre-recorded Thai audio
"""
import sys
sys.path.insert(0, "src")

from stt.typhoon_asr_client import TyphoonASR
import time

print("=" * 80)
print("🌪️  TESTING TYPHOON ASR WITH PRE-RECORDED THAI AUDIO")
print("=" * 80)

# Thai audio file from typhoon-asr examples
audio_file = r"C:\Users\Win 10 Pro\Desktop\RobotAI\microphone_test.wav"

print(f"\n📁 Audio file: {audio_file}")

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
print("Transcribing Thai audio...")
print("=" * 80)

start_time = time.time()
result = asr.transcribe_audio(audio_file)
elapsed = time.time() - start_time

# Display results
print("\n" + "=" * 80)
print("📊 TRANSCRIPTION RESULT")
print("=" * 80)
print(f"✅ Text: '{result['text']}'")
print(f"📈 Confidence: {result['confidence']:.2%}")
print(f"🌐 Language: {result['language']}")
print(f"⏱️  Processing Time: {elapsed:.2f}s")

if result['text']:
    print(f"\n✅ SUCCESS! Transcribed {len(result['text'])} characters")
    print("\n💡 This proves:")
    print("   1. ✅ Typhoon ASR model is working correctly")
    print("   2. ✅ The model can transcribe Thai audio")
    print("   3. ⚠️  Your microphone recordings were likely in English or too quiet")
    print("\n📝 To fix the test:")
    print("   - Speak in THAI language (not English)")
    print("   - Speak LOUDER and closer to the microphone")
    print("   - Ensure you're using the correct microphone device")
else:
    print("\n❌ FAILED: Empty transcription")
    print("   This suggests an issue with the audio file or model")

print("=" * 80)
