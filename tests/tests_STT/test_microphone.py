"""
Test microphone recording and playback
"""
import sys
import os
sys.path.insert(0, "src")

from utils.audio_recorder import AudioRecorder
import soundfile as sf
import numpy as np

print("=" * 80)
print("🎤 MICROPHONE TEST")
print("=" * 80)

# List available devices
recorder = AudioRecorder()
recorder.list_devices()

# Record 5 seconds
print("\n" + "=" * 80)
print("Recording 5 seconds of audio...")
print("=" * 80)

audio_data, file_path = recorder.record_and_save(
    output_path="tests/Sound_Samples/microphone_test.wav",
    duration=5.0,
    show_countdown=True
)

# Check audio statistics
print("\n" + "=" * 80)
print("AUDIO ANALYSIS")
print("=" * 80)
print(f"✅ File saved: {file_path}")
print(f"📊 Audio shape: {audio_data.shape}")
print(f"📊 Data type: {audio_data.dtype}")
print(f"📊 Sample rate: 16000 Hz")
print(f"📊 Duration: {len(audio_data) / 16000:.2f} seconds")
print(f"📊 Min value: {audio_data.min():.6f}")
print(f"📊 Max value: {audio_data.max():.6f}")
print(f"📊 Mean value: {audio_data.mean():.6f}")
print(f"📊 RMS (loudness): {np.sqrt(np.mean(audio_data**2)):.6f}")

# Check if audio is silent
rms = np.sqrt(np.mean(audio_data**2))
if rms < 0.001:
    print("\n⚠️  WARNING: Audio appears to be SILENT!")
    print("   RMS value is very low, microphone might not be working")
    print("   or wrong device is selected.")
else:
    print(f"\n✅ Audio detected! RMS: {rms:.6f}")

# Check file size
file_size = os.path.getsize(file_path)
print(f"\n📁 File size: {file_size:,} bytes ({file_size / 1024:.1f} KB)")

print("\n" + "=" * 80)
print("Test complete! Check 'microphone_test.wav' to verify recording.")
print("=" * 80)
