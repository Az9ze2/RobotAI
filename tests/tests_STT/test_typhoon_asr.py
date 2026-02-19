"""
Typhoon ASR Unit Tests
Test transcription with and without grammar correction
"""

import pytest
import sys
import os
import io
from pathlib import Path
import time

# Configure UTF-8 encoding for Windows console
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from stt.typhoon_asr_client import TyphoonASR
from utils.audio_recorder import record_audio
from utils.text_corrector import ThaiTextCorrector
from llm.typhoon_client import TyphoonClient


class TestTyphoonASR:
    """Test suite for Typhoon ASR transcription"""
    
    @pytest.fixture(scope="class")
    def asr_model(self):
        """Initialize Typhoon ASR model once for all tests"""
        print("\n" + "=" * 80)
        print("🌪️  INITIALIZING TYPHOON ASR MODEL")
        print("=" * 80)
        
        try:
            model = TyphoonASR(
                model_name="scb10x/typhoon-asr-realtime",
                device="auto",
                language="th"
            )
            print("✅ Model loaded successfully!\n")
            return model
        except Exception as e:
            pytest.skip(f"Failed to load Typhoon ASR model: {e}")
    
    @pytest.fixture(scope="class")
    def text_corrector(self):
        """Initialize text corrector with LLM"""
        print("\n" + "=" * 80)
        print("📝 INITIALIZING TEXT CORRECTOR")
        print("=" * 80)
        
        try:
            # Initialize LLM client
            llm = TyphoonClient(
                api_url="http://localhost:11434",
                model="scb10x/typhoon2.1-gemma3-12b:latest"
            )
            
            # Initialize corrector
            corrector = ThaiTextCorrector(
                llm_client=llm,
                use_llm=True,
                use_dictionary=True
            )
            print("✅ Text corrector initialized!\n")
            return corrector
        except Exception as e:
            print(f"⚠️  Warning: Could not initialize text corrector: {e}")
            return None
    
    def test_transcribe_no_grammar(self, asr_model):
        """
        Test 1: Transcribe audio WITHOUT grammar correction
        Records 5 seconds of user audio and transcribes it
        """
        print("\n" + "=" * 80)
        print("🧪 TEST 1: TRANSCRIPTION WITHOUT GRAMMAR CHECKER")
        print("=" * 80)
        
        # Record audio
        print("\n📹 Step 1: Recording audio from microphone")
        print("-" * 80)
        
        audio_file = "test_audio_no_grammar.wav"
        try:
            audio_data, file_path = record_audio(
                duration=5.0,
                sample_rate=16000,
                output_path=audio_file,
                show_countdown=True
            )
            
            print(f"\n✅ Audio recorded: {file_path}")
            print(f"   Duration: 5.0 seconds")
            print(f"   Sample rate: 16000 Hz")
            
        except Exception as e:
            pytest.fail(f"Audio recording failed: {e}")
        
        # Transcribe
        print("\n🎙️  Step 2: Transcribing with Typhoon ASR")
        print("-" * 80)
        
        start_time = time.time()
        result = asr_model.transcribe_audio(audio_file)
        transcription_time = time.time() - start_time
        
        # Display results
        print("\n" + "=" * 80)
        print("📊 RESULTS (WITHOUT GRAMMAR CORRECTION)")
        print("=" * 80)
        print(f"✅ Transcription: '{result['text']}'")
        print(f"📈 Confidence: {result['confidence']:.2%}")
        print(f"🌐 Language: {result['language']}")
        print(f"⏱️  Processing Time: {transcription_time:.2f}s")
        print(f"🚀 RTF (Real-Time Factor): {transcription_time / 5.0:.3f}x")
        
        if transcription_time / 5.0 < 1.0:
            print("   ✨ Real-time capable!")
        
        print("=" * 80 + "\n")
        
        # Cleanup
        if os.path.exists(audio_file):
            os.remove(audio_file)
        
        # Assertions
        assert result['text'] is not None, "Transcription should not be None"
        assert result['confidence'] >= 0.0, "Confidence should be >= 0"
        assert result['language'] == 'th', "Language should be Thai"
    
    def test_transcribe_with_grammar(self, asr_model, text_corrector):
        """
        Test 2: Transcribe audio WITH grammar correction
        Records 5 seconds of user audio, transcribes, and applies grammar correction
        """
        print("\n" + "=" * 80)
        print("🧪 TEST 2: TRANSCRIPTION WITH GRAMMAR CHECKER")
        print("=" * 80)
        
        if text_corrector is None:
            pytest.skip("Text corrector not available")
        
        # Record audio
        print("\n📹 Step 1: Recording audio from microphone")
        print("-" * 80)
        
        audio_file = "test_audio_with_grammar.wav"
        try:
            audio_data, file_path = record_audio(
                duration=5.0,
                sample_rate=16000,
                output_path=audio_file,
                show_countdown=True
            )
            
            print(f"\n✅ Audio recorded: {file_path}")
            print(f"   Duration: 5.0 seconds")
            print(f"   Sample rate: 16000 Hz")
            
        except Exception as e:
            pytest.fail(f"Audio recording failed: {e}")
        
        # Transcribe
        print("\n🎙️  Step 2: Transcribing with Typhoon ASR")
        print("-" * 80)
        
        start_time = time.time()
        result = asr_model.transcribe_audio(audio_file)
        transcription_time = time.time() - start_time
        
        original_text = result['text']
        confidence = result['confidence']
        
        print(f"✅ Raw transcription: '{original_text}'")
        print(f"📈 Confidence: {confidence:.2%}")
        
        # Apply grammar correction
        print("\n📝 Step 3: Applying grammar correction")
        print("-" * 80)
        
        correction_start = time.time()
        correction_result = text_corrector.correct(
            text=original_text,
            confidence=confidence,
            skip_if_confident=False  # Always apply correction for testing
        )
        correction_time = time.time() - correction_start
        
        corrected_text = correction_result['corrected_text']
        corrections = correction_result['corrections']
        
        # Display results
        print("\n" + "=" * 80)
        print("📊 RESULTS (WITH GRAMMAR CORRECTION)")
        print("=" * 80)
        print(f"🎤 Original Text: '{original_text}'")
        print(f"✨ Corrected Text: '{corrected_text}'")
        print(f"📈 Confidence: {confidence:.2%}")
        print(f"🌐 Language: {result['language']}")
        print(f"⏱️  Transcription Time: {transcription_time:.2f}s")
        print(f"⏱️  Correction Time: {correction_time:.2f}s")
        print(f"⏱️  Total Time: {transcription_time + correction_time:.2f}s")
        print(f"🚀 RTF (Real-Time Factor): {transcription_time / 5.0:.3f}x")
        
        # Show corrections made
        print(f"\n🔧 Corrections Applied: {len(corrections)}")
        if corrections:
            print("-" * 80)
            for i, correction in enumerate(corrections, 1):
                if correction['type'] == 'dictionary':
                    print(f"   {i}. Dictionary: '{correction['wrong']}' → '{correction['correct']}'")
                elif correction['type'] == 'llm':
                    print(f"   {i}. LLM: Applied advanced correction")
        else:
            print("   No corrections needed")
        
        print("=" * 80 + "\n")
        
        # Cleanup
        if os.path.exists(audio_file):
            os.remove(audio_file)
        
        # Assertions
        assert result['text'] is not None, "Transcription should not be None"
        assert corrected_text is not None, "Corrected text should not be None"
        assert result['confidence'] >= 0.0, "Confidence should be >= 0"
        assert result['language'] == 'th', "Language should be Thai"


if __name__ == "__main__":
    """
    Run tests directly without pytest
    """
    print("\n" + "=" * 80)
    print("🌪️  TYPHOON ASR TESTING SUITE")
    print("=" * 80)
    print("\nThis test suite will:")
    print("1. Record 5 seconds of audio from your microphone (Test 1)")
    print("2. Transcribe without grammar correction")
    print("3. Record another 5 seconds of audio (Test 2)")
    print("4. Transcribe with grammar correction")
    print("\nMake sure your microphone is ready!")
    print("=" * 80)
    
    input("\nPress ENTER to start testing...")
    
    # Run with pytest
    pytest.main([__file__, "-v", "-s"])
