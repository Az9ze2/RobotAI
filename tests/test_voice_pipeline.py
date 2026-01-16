"""
Standalone End-to-End Voice Pipeline Test
Tests STT → TTS pipeline without requiring API server or Milvus

This is a simplified test to verify the complete voice interaction works:
1. Record audio from microphone
2. Transcribe with Whisper STT
3. Generate response with TTS
4. Play audio back
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from stt.whisper_client import WhisperSTT
from tts.vachana_client import VachanaTTS
from utils.audio_utils import record_audio, play_audio_file, save_audio
import time
from loguru import logger

def test_stt_only():
    """Test STT only"""
    print("\n" + "="*70)
    print("🎤 TEST 1: Speech-to-Text (STT) Only")
    print("="*70)
    
    try:
        # Initialize STT
        print("\n📥 Loading Whisper STT model...")
        stt = WhisperSTT(model_size="small", language="th")
        print("✅ Whisper STT loaded successfully")
        
        # Record audio
        duration = 5
        print(f"\n🎤 Recording for {duration} seconds...")
        print("   Say something in Thai!")
        
        audio_data, sample_rate = record_audio(duration=duration)
        
        # Save audio
        temp_file = f"test_stt_{int(time.time())}.wav"
        save_audio(audio_data, temp_file, sample_rate)
        print(f"💾 Audio saved to: {temp_file}")
        
        # Transcribe
        print("\n🔄 Transcribing...")
        start_time = time.time()
        result = stt.transcribe_audio(temp_file)
        transcribe_time = time.time() - start_time
        
        print("\n" + "="*70)
        print("📊 STT RESULTS")
        print("="*70)
        print(f"✅ Transcribed text: {result['text']}")
        print(f"✅ Confidence: {result['confidence']:.2%}")
        print(f"✅ Language: {result['language']}")
        print(f"⏱️  Processing time: {transcribe_time:.2f}s")
        print(f"📁 Audio file: {temp_file}")
        
        # Clean up
        Path(temp_file).unlink(missing_ok=True)
        
        return True, result
        
    except Exception as e:
        print(f"\n❌ STT Test Failed: {e}")
        import traceback
        traceback.print_exc()
        return False, None

def test_tts_only():
    """Test TTS only"""
    print("\n" + "="*70)
    print("🔊 TEST 2: Text-to-Speech (TTS) Only")
    print("="*70)
    
    try:
        # Initialize TTS
        print("\n📥 Loading VachanaTTS model...")
        tts = VachanaTTS()
        print("✅ VachanaTTS loaded successfully")
        
        # Test texts
        test_texts = [
            "สวัสดีครับ ยินดีต้อนรับสู่มหาวิทยาลัย",
            "ห้องสมุดอยู่ที่ชั้นสามของอาคารหลัก",
            "ขอบคุณสำหรับคำถามของคุณ"
        ]
        
        print(f"\n🎙️  Testing {len(test_texts)} sample texts...\n")
        
        for i, text in enumerate(test_texts, 1):
            print(f"\n{'='*70}")
            print(f"Sample {i}/{len(test_texts)}")
            print(f"{'='*70}")
            print(f"📝 Text: {text}")
            
            # Synthesize
            print("🔄 Synthesizing...")
            start_time = time.time()
            audio_file, metadata = tts.synthesize(text)
            synthesis_time = time.time() - start_time
            
            print(f"\n✅ Generated: {audio_file}")
            print(f"⏱️  Synthesis time: {synthesis_time:.3f}s")
            print(f"🔊 Sample rate: {metadata['sampling_rate']:,} Hz")
            print(f"📏 Audio duration: {metadata['duration']:.2f}s")
            print(f"🎵 Model: {metadata['model']}")
            
            # Play
            print(f"\n▶️  Playing audio...")
            play_audio_file(audio_file)
            
            # Clean up
            Path(audio_file).unlink(missing_ok=True)
            
            if i < len(test_texts):
                print("\n⏳ Waiting 2 seconds before next sample...")
                time.sleep(2)
        
        return True
        
    except Exception as e:
        print(f"\n❌ TTS Test Failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_full_pipeline():
    """Test complete STT → TTS pipeline"""
    print("\n" + "="*70)
    print("🎭 TEST 3: Full Voice Pipeline (STT → TTS)")
    print("="*70)
    
    try:
        # Initialize both
        print("\n📥 Loading models...")
        print("   - Whisper STT (small, Thai)")
        stt = WhisperSTT(model_size="small", language="th")
        print("   - VachanaTTS (MALEV1)")
        tts = VachanaTTS()
        print("✅ All models loaded\n")
        
        # Record
        duration = 5
        print(f"🎤 Recording for {duration} seconds...")
        print("   Say something in Thai (e.g., ask about the library location)")
        
        audio_data, sample_rate = record_audio(duration=duration)
        
        # Save
        temp_file = f"test_pipeline_{int(time.time())}.wav"
        save_audio(audio_data, temp_file, sample_rate)
        
        # Transcribe
        print("\n🔄 Step 1: Transcribing your speech...")
        stt_start = time.time()
        result = stt.transcribe_audio(temp_file)
        stt_time = time.time() - stt_start
        
        user_text = result['text']
        confidence = result['confidence']
        
        print(f"\n✅ You said: {user_text}")
        print(f"   Confidence: {confidence:.2%}")
        print(f"   Time: {stt_time:.2f}s")
        
        if not user_text.strip():
            print("\n⚠️  No speech detected. Try again with louder/clearer speech.")
            Path(temp_file).unlink(missing_ok=True)
            return False
        
        # Generate response (simulated - without API)
        print("\n🔄 Step 2: Generating response...")
        
        # Simple response logic
        if any(word in user_text.lower() for word in ["ห้องสมุด", "library", "หนังสือ"]):
            response = "ห้องสมุดอยู่ที่ชั้นสามของอาคารหลัก เปิดทำการตั้งแต่เช้าจนถึงเย็นครับ"
        elif any(word in user_text.lower() for word in ["สวัสดี", "hello", "ดี"]):
            response = "สวัสดีครับ ยินดีต้อนรับสู่มหาวิทยาลัย มีอะไรให้ผมช่วยไหมครับ"
        elif any(word in user_text.lower() for word in ["อาหาร", "food", "กิน", "ร้าน"]):
            response = "โรงอาหารกลางอยู่ระหว่างคณะวิทยาศาสตร์กับคณะบริหารธุรกิจครับ"
        else:
            response = f"ขอบคุณสำหรับคำถามครับ คุณถามว่า {user_text}"
        
        print(f"💬 Robot response: {response}")
        
        # Synthesize
        print("\n🔄 Step 3: Synthesizing speech...")
        tts_start = time.time()
        audio_file, metadata = tts.synthesize(response)
        tts_time = time.time() - tts_start
        
        print(f"✅ Audio generated in {tts_time:.3f}s")
        print(f"   Duration: {metadata['duration']:.2f}s")
        print(f"   Model: {metadata['model']}")
        print(f"   Sample rate: {metadata['sampling_rate']:,} Hz")
        
        # Play
        print("\n▶️  Step 4: Playing robot response...")
        play_audio_file(audio_file)
        
        # Summary
        total_time = stt_time + tts_time
        print("\n" + "="*70)
        print("📊 PIPELINE SUMMARY")
        print("="*70)
        print(f"⏱️  STT Time: {stt_time:.2f}s")
        print(f"⏱️  TTS Time: {tts_time:.3f}s")
        print(f"⏱️  Total Processing: {total_time:.2f}s")
        print(f"📝 Input: {user_text}")
        print(f"💬 Output: {response}")
        print("\n✅ Full pipeline test completed successfully!")
        
        # Clean up
        Path(temp_file).unlink(missing_ok=True)
        Path(audio_file).unlink(missing_ok=True)
        
        return True
        
    except Exception as e:
        print(f"\n❌ Full Pipeline Test Failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main test runner"""
    print("\n" + "🤖 "*35)
    print("  ROBOT AI VOICE PIPELINE - END-TO-END TEST")
    print("  Testing STT (Whisper) + TTS (VachanaTTS)")
    print("🤖 "*35)
    
    print("\n📋 Test Plan:")
    print("   1. Test STT only (record & transcribe)")
    print("   2. Test TTS only (synthesize & play)")
    print("   3. Test full pipeline (STT → Response → TTS)")
    
    input("\n Press ENTER to start testing...")
    
    # Test 1: STT
    stt_success, stt_result = test_stt_only()
    
    if not stt_success:
        print("\n⚠️  STT test failed. Cannot proceed with other tests.")
        return
    
    input("\n✅ STT test complete. Press ENTER to test TTS...")
    
    # Test 2: TTS
    tts_success = test_tts_only()
    
    if not tts_success:
        print("\n⚠️  TTS test failed. Skipping full pipeline test.")
        return
    
    input("\n✅ TTS test complete. Press ENTER for full pipeline test...")
    
    # Test 3: Full Pipeline
    pipeline_success = test_full_pipeline()
    
    # Final summary
    print("\n" + "="*70)
    print("🎉 TEST SUITE SUMMARY")
    print("="*70)
    print(f"✅ STT Test: {'PASSED' if stt_success else 'FAILED'}")
    print(f"✅ TTS Test: {'PASSED' if tts_success else 'FAILED'}")
    print(f"✅ Full Pipeline: {'PASSED' if pipeline_success else 'FAILED'}")
    
    if stt_success and tts_success and pipeline_success:
        print("\n🎉 ALL TESTS PASSED! Voice pipeline is ready for production!")
    else:
        print("\n⚠️  Some tests failed. Check errors above.")
    
    print("="*70 + "\n")

if __name__ == "__main__":
    main()
