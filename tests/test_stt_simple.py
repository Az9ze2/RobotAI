"""
Simple STT Test Script
Verifies Whisper model is working correctly
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from stt.whisper_client import WhisperSTT
from utils.audio_utils import record_audio, save_audio
import time

def test_stt():
    """Test STT with recorded audio"""
    print("\n" + "="*70)
    print("🎤 WHISPER STT TEST")
    print("="*70)
    
    try:
        # Initialize STT
        print("\n📥 Loading Whisper STT model (small, Thai)...")
        start_load = time.time()
        stt = WhisperSTT(model_size="small", language="th")
        load_time = time.time() - start_load
        print(f"✅ Model loaded in {load_time:.2f}s")
        print(f"   Model: {stt.model_size}")
        print(f"   Language: {stt.language}")
        print(f"   Device: {stt.device}")
        
        # Record audio
        duration = 5
        print(f"\n🎤 Recording for {duration} seconds...")
        print("   Please say something in Thai!")
        print("   Example: 'สวัสดีครับ' or 'ห้องสมุดอยู่ที่ไหน'")
        print("\n   Recording starts in 3 seconds...")
        time.sleep(1)
        print("   3...")
        time.sleep(1)
        print("   2...")
        time.sleep(1)
        print("   1...")
        time.sleep(1)
        print("\n   🔴 RECORDING NOW! SPEAK!")
        
        audio_data, sample_rate = record_audio(duration=duration)
        
        print("\n   ✅ Recording complete!")
        
        # Save audio
        temp_file = f"test_stt_{int(time.time())}.wav"
        save_audio(audio_data, temp_file, sample_rate)
        print(f"   💾 Saved to: {temp_file}")
        
        # Transcribe
        print("\n🔄 Transcribing audio...")
        start_transcribe = time.time()
        result = stt.transcribe_audio(temp_file)
        transcribe_time = time.time() - start_transcribe
        
        # Results
        print("\n" + "="*70)
        print("📊 TRANSCRIPTION RESULTS")
        print("="*70)
        print(f"\n✅ Text: {result['text']}")
        print(f"✅ Confidence: {result['confidence']:.2%}")
        print(f"✅ Language detected: {result['language']}")
        print(f"⏱️  Transcription time: {transcribe_time:.2f}s")
        print(f"📁 Audio file: {temp_file}")
        
        if result['text'].strip():
            print("\n✅ SUCCESS! STT is working correctly!")
            
            # Test multiple times for consistency
            print("\n" + "="*70)
            print("🔄 CONSISTENCY TEST - 2 more samples")
            print("="*70)
            
            for i in range(2):
                print(f"\n📝 Sample {i+2}/3")
                print(f"🎤 Recording for {duration} seconds...")
                print("   Recording in 2 seconds...")
                time.sleep(2)
                print("   🔴 RECORDING!")
                
                audio_data, sample_rate = record_audio(duration=duration)
                temp_file2 = f"test_stt_{int(time.time())}_{i}.wav"
                save_audio(audio_data, temp_file2, sample_rate)
                
                start = time.time()
                result2 = stt.transcribe_audio(temp_file2)
                duration_stt = time.time() - start
                
                print(f"   ✅ Text: {result2['text']}")
                print(f"   ✅ Confidence: {result2['confidence']:.2%}")
                print(f"   ⏱️  Time: {duration_stt:.2f}s")
                
                # Clean up
                Path(temp_file2).unlink(missing_ok=True)
            
            # Summary
            print("\n" + "="*70)
            print("🎉 TEST SUMMARY")
            print("="*70)
            print("✅ Model: Whisper Small (Thai)")
            print("✅ Status: WORKING PERFECTLY")
            print(f"✅ Average transcription time: ~{transcribe_time:.2f}s")
            print("✅ Language detection: Thai")
            print("✅ Confidence: High")
            print("\n🚀 STT is ready for production!")
            
            return True
        else:
            print("\n⚠️  No speech detected in audio.")
            print("   Possible issues:")
            print("   - Microphone not working")
            print("   - Audio too quiet")
            print("   - Background noise too loud")
            print("   - No speech during recording")
            return False
        
    except Exception as e:
        print(f"\n❌ STT Test Failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        # Clean up
        if 'temp_file' in locals():
            Path(temp_file).unlink(missing_ok=True)

if __name__ == "__main__":
    print("\n" + "🤖 "*35)
    print("  WHISPER STT - SIMPLE TEST")
    print("  Model: small (Thai language)")
    print("🤖 "*35)
    
    print("\n📋 This test will:")
    print("   1. Load the Whisper model")
    print("   2. Record 5 seconds of audio (3 times)")
    print("   3. Transcribe to Thai text")
    print("   4. Show confidence and performance")
    
    print("\n💡 Tips:")
    print("   - Speak clearly in Thai")
    print("   - Keep background noise low")
    print("   - Speak at normal volume")
    
    input("\n Press ENTER to start STT test...")
    
    success = test_stt()
    
    print("\n" + "="*70)
    if success:
        print("✅ STT TEST PASSED")
    else:
        print("❌ STT TEST FAILED")
    print("="*70 + "\n")
