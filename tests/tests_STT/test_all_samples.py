"""
Test Typhoon ASR with all pre-recorded Thai audio samples
"""
import sys
import os
from pathlib import Path
import time

# Add src to path
sys.path.insert(0, "src")

from stt.typhoon_asr_client import TyphoonASR

print("=" * 80)
print("🌪️  TESTING TYPHOON ASR WITH ALL AUDIO SAMPLES")
print("=" * 80)

# Find all audio files in Sound_Samples directory
samples_dir = Path("tests/Sound_Samples")
audio_files = list(samples_dir.glob("*.wav"))

if not audio_files:
    print("\n❌ ERROR: No audio files found in tests/Sound_Samples/")
    sys.exit(1)

print(f"\n📁 Found {len(audio_files)} audio file(s):")
for i, file in enumerate(audio_files, 1):
    file_size = file.stat().st_size / 1024  # KB
    print(f"   {i}. {file.name} ({file_size:.1f} KB)")

# Initialize Typhoon ASR
print("\n" + "=" * 80)
print("🔧 Loading Typhoon ASR model...")
print("=" * 80)

try:
    asr = TyphoonASR(
        model_name="scb10x/typhoon-asr-realtime",
        device="auto",  # Will use GPU if available, otherwise CPU
        language="th"
    )
    print("✅ Model loaded successfully!")
except Exception as e:
    print(f"\n❌ ERROR: Failed to load Typhoon ASR model")
    print(f"   {e}")
    sys.exit(1)

# Test each audio file
print("\n" + "=" * 80)
print("🎙️  TRANSCRIBING ALL AUDIO SAMPLES")
print("=" * 80)

results = []

for i, audio_file in enumerate(audio_files, 1):
    print(f"\n{'─' * 80}")
    print(f"📝 Test {i}/{len(audio_files)}: {audio_file.name}")
    print(f"{'─' * 80}")
    
    try:
        # Transcribe
        start_time = time.time()
        result = asr.transcribe_audio(str(audio_file))
        elapsed = time.time() - start_time
        
        # Store results
        results.append({
            'file': audio_file.name,
            'text': result['text'],
            'confidence': result['confidence'],
            'language': result['language'],
            'time': elapsed
        })
        
        # Display result
        print(f"✅ Transcription: '{result['text']}'")
        print(f"📈 Confidence: {result['confidence']:.2%}")
        print(f"🌐 Language: {result['language']}")
        print(f"⏱️  Processing Time: {elapsed:.2f}s")
        
        # Check if transcription is empty
        if not result['text'] or result['text'].strip() == "":
            print("⚠️  WARNING: Empty transcription!")
        
    except Exception as e:
        print(f"❌ ERROR: Transcription failed")
        print(f"   {e}")
        results.append({
            'file': audio_file.name,
            'text': None,
            'confidence': 0.0,
            'language': 'unknown',
            'time': 0.0,
            'error': str(e)
        })

# Summary
print("\n" + "=" * 80)
print("📊 SUMMARY OF ALL TESTS")
print("=" * 80)

total_tests = len(results)
successful_tests = sum(1 for r in results if r['text'] and r['text'].strip())
failed_tests = total_tests - successful_tests

print(f"\n📈 Overall Statistics:")
print(f"   Total tests: {total_tests}")
print(f"   ✅ Successful: {successful_tests}")
print(f"   ❌ Failed/Empty: {failed_tests}")
print(f"   Success rate: {(successful_tests/total_tests)*100:.1f}%")

print(f"\n📝 Detailed Results:")
print("─" * 80)

for i, result in enumerate(results, 1):
    print(f"\n{i}. {result['file']}")
    if 'error' in result:
        print(f"   ❌ Error: {result['error']}")
    elif not result['text'] or result['text'].strip() == "":
        print(f"   ⚠️  Empty transcription")
    else:
        print(f"   ✅ Text: '{result['text']}'")
        print(f"   📈 Confidence: {result['confidence']:.2%}")
        print(f"   ⏱️  Time: {result['time']:.2f}s")

print("\n" + "=" * 80)
print("✅ Testing complete!")
print("=" * 80)
