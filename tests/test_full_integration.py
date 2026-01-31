"""
Full Integration Test: STT → MCP → LLM → TTS
Tests the complete conversation pipeline end-to-end
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import time
import argparse
import yaml
from loguru import logger

# Import components
from stt.whisper_client import WhisperSTT
from tts.vachana_client import VachanaTTS
from llm.typhoon_client import TyphoonClient
from utils.audio_utils import record_audio, save_audio, play_audio_file
import requests


class FullIntegrationTest:
    """Full pipeline integration test"""
    
    def __init__(self, config_path: str = None, mcp_url: str = None):
        """Initialize all components"""
        # Load configuration
        if config_path is None:
            config_path = Path(__file__).parent.parent / 'config' / 'local.yaml'
        
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        # Set MCP URL from config or parameter
        if mcp_url:
            self.mcp_url = mcp_url
        else:
            host = self.config['api']['host']
            port = self.config['api']['port']
            # Use localhost instead of 0.0.0.0 for client connections
            host = 'localhost' if host == '0.0.0.0' else host
            self.mcp_url = f"http://{host}:{port}"
        
        self.components_ready = {}
        
    def check_components(self) -> bool:
        """Check if all components are available"""
        print("\n" + "="*70)
        print("🔍 CHECKING COMPONENTS")
        print("="*70)
        
        # Check STT (Whisper)
        print("\n1️⃣ Checking STT (Whisper)...")
        try:
            stt_config = self.config['stt']
            self.stt = WhisperSTT(
                model_size=stt_config['model'],
                language=stt_config['language']
            )
            print(f"   ✅ Whisper STT loaded (model: {stt_config['model']})")
            self.components_ready['stt'] = True
        except Exception as e:
            print(f"   ❌ Whisper STT failed: {e}")
            self.components_ready['stt'] = False
        
        # Check MCP Server
        print("\n2️⃣ Checking MCP Server...")
        try:
            response = requests.get(f"{self.mcp_url}/")
            if response.status_code == 200:
                print(f"   ✅ MCP Server running at {self.mcp_url}")
                self.components_ready['mcp'] = True
            else:
                print(f"   ❌ MCP Server returned status {response.status_code}")
                self.components_ready['mcp'] = False
        except Exception as e:
            print(f"   ❌ MCP Server not accessible: {e}")
            print(f"   💡 Start server with: python src/mcp/server.py")
            self.components_ready['mcp'] = False
        
        # Check LLM (Typhoon)
        print("\n3️⃣ Checking LLM (Typhoon via Ollama)...")
        try:
            llm_config = self.config['llm']
            self.llm = TyphoonClient(
                api_url=llm_config['api_url'],
                model=llm_config['model']
            )
            print(f"   ✅ Typhoon LLM connected (model: {llm_config['model']})")
            self.components_ready['llm'] = True
        except Exception as e:
            print(f"   ❌ Typhoon LLM failed: {e}")
            print(f"   💡 Start Ollama with: ollama run {self.config['llm']['model']}")
            self.components_ready['llm'] = False
        
        # Check TTS (VachanaTTS)
        print("\n4️⃣ Checking TTS (VachanaTTS)...")
        try:
            tts_config = self.config['tts']
            self.tts = VachanaTTS(
                default_model=tts_config.get('default_model'),
                speaking_rate=tts_config.get('speaking_rate', 1.0)
            )
            if self.tts.available_models:
                print(f"   ✅ VachanaTTS loaded (model: {self.tts.default_model})")
                self.components_ready['tts'] = True
            else:
                print("   ⚠️  VachanaTTS loaded but no models found")
                self.components_ready['tts'] = False
        except Exception as e:
            print(f"   ❌ VachanaTTS failed: {e}")
            self.components_ready['tts'] = False
        
        # Summary
        print("\n" + "="*70)
        print("📊 COMPONENT STATUS SUMMARY")
        print("="*70)
        all_ready = True
        for component, ready in self.components_ready.items():
            status = "✅ READY" if ready else "❌ NOT READY"
            print(f"{component.upper():15} : {status}")
            if not ready:
                all_ready = False
        
        return all_ready
    
    def run_full_cycle(self, audio_duration: int = 5, session_id: str = None):
        """Run complete STT → MCP → LLM → TTS cycle"""
        
        print("\n" + "="*70)
        print("🤖 FULL INTEGRATION TEST - CONVERSATION CYCLE")
        print("="*70)
        
        # Initialize session
        if session_id is None:
            session_id = f"session_test_{int(time.time())}"
        
        # Step 1: STT - Record and Transcribe
        print("\n" + "-"*70)
        print("1️⃣ SPEECH-TO-TEXT (STT)")
        print("-"*70)
        print(f"🎤 Recording for {audio_duration} seconds...")
        print("   Please say something in Thai!")
        print("   Examples:")
        print("   - 'พาไปห้องแลปหน่อยครับ' (navigation)")
        print("   - 'สวัสดีครับ' (greeting)")
        print("\n   Recording starts in 3 seconds...")
        time.sleep(1)
        print("   3...")
        time.sleep(1)
        print("   2...")
        time.sleep(1)
        print("   1...")
        time.sleep(1)
        print("\n   🔴 RECORDING NOW! SPEAK!")
        
        start_stt = time.time()
        audio_data, sample_rate = record_audio(duration=audio_duration)
        
        # Save audio
        audio_file = f"test_integration_{int(time.time())}.wav"
        save_audio(audio_data, audio_file, sample_rate)
        print(f"\n   ✅ Recording complete! Saved to: {audio_file}")
        
        # Transcribe
        print("   🔄 Transcribing...")
        stt_result = self.stt.transcribe_audio(audio_file)
        stt_time = time.time() - start_stt
        
        transcribed_text = stt_result['text']
        confidence = stt_result['confidence']
        
        print(f"\n   📝 Transcribed Text: '{transcribed_text}'")
        print(f"   📊 Confidence: {confidence:.2%}")
        print(f"   ⏱️  STT Time: {stt_time:.2f}s")
        
        if not transcribed_text.strip():
            print("\n   ⚠️  No speech detected! Aborting test.")
            return None
        
        # Step 1.5: Grammar Correction (NEW)
        print("\n   🔄 Correcting grammar...")
        start_correction = time.time()
        
        from utils.text_corrector import ThaiTextCorrector
        
        correction_config = self.config.get('text_correction', {})
        if correction_config.get('enabled', True):
            corrector = ThaiTextCorrector(
                llm_client=self.llm if self.components_ready.get('llm') else None,
                use_llm=correction_config.get('use_llm', True),
                use_dictionary=correction_config.get('use_dictionary', True),
                llm_temperature=correction_config.get('llm_temperature', 0.3),
                max_corrections=correction_config.get('max_corrections', 5)
            )
            
            correction_result = corrector.correct(
                transcribed_text,
                confidence=confidence,
                skip_if_confident=correction_config.get('skip_if_confident', True),
                confidence_threshold=correction_config.get('confidence_threshold', 0.95)
            )
            
            corrected_text = correction_result['corrected_text']
            corrections_made = correction_result['corrections']
            correction_time = time.time() - start_correction
            
            if corrections_made:
                print(f"   ✏️  Original:  '{transcribed_text}'")
                print(f"   ✅ Corrected: '{corrected_text}'")
                print(f"   📝 Changes: {len(corrections_made)} corrections")
                print(f"   ⏱️  Correction Time: {correction_time:.2f}s")
                
                # Log corrections
                for i, corr in enumerate(corrections_made[:3], 1):  # Show first 3
                    if corr['type'] == 'dictionary':
                        print(f"      {i}. '{corr['wrong']}' → '{corr['correct']}'")
            else:
                print(f"   ✅ No corrections needed")
                corrected_text = transcribed_text
        else:
            print(f"   ⚠️  Grammar correction disabled in config")
            corrected_text = transcribed_text

        
        # Step 2: MCP - Intent Detection
        print("\n" + "-"*70)
        print("2️⃣ MODEL CONTEXT PROTOCOL (MCP)")
        print("-"*70)
        
        start_mcp = time.time()
        
        # Simulate face detection first (optional)
        face_payload = {
            "student_id": "64010001",
            "confidence": 0.98,
            "timestamp": time.time()
        }
        face_response = requests.post(f"{self.mcp_url}/api/v1/perception/face", json=face_payload)
        if face_response.status_code == 200:
            session_id = face_response.json()['session_id']
            print(f"   👤 Face detected: {session_id}")
        
        # Send speech to MCP
        speech_payload = {
            "session_id": session_id,
            "text": transcribed_text,
            "confidence": confidence
        }
        
        mcp_response = requests.post(f"{self.mcp_url}/api/v1/perception/speech", json=speech_payload)
        mcp_time = time.time() - start_mcp
        
        if mcp_response.status_code == 200:
            mcp_result = mcp_response.json()
            intent = mcp_result['intent']
            response_text = mcp_result['response_text']
            should_navigate = mcp_result.get('should_navigate', False)
            nav_goal = mcp_result.get('navigation_goal')
            
            print(f"   🎯 Intent: {intent}")
            print(f"   💬 Response: '{response_text}'")
            if should_navigate:
                print(f"   🗺️  Navigation Goal: {nav_goal}")
            print(f"   ⏱️  MCP Time: {mcp_time:.2f}s")
        else:
            print(f"   ❌ MCP request failed: {mcp_response.status_code}")
            return None
        
        # Step 3: LLM - Enhanced Response (Optional)
        print("\n" + "-"*70)
        print("3️⃣ LARGE LANGUAGE MODEL (LLM) - Optional Enhancement")
        print("-"*70)
        
        if self.components_ready.get('llm', False):
            print("   🔄 Generating enhanced response with Typhoon LLM...")
            start_llm = time.time()
            
            # Build prompt for LLM
            system_prompt = """คุณเป็นหุ่นยนต์ช่วยเหลือในมหาวิทยาลัย พูดภาษาไทยอย่างสุภาพและเป็นมิตร
ตอบสั้นๆ กระชับ ไม่เกิน 2 ประโยค"""
            
            user_prompt = f"""ผู้ใช้พูดว่า: "{transcribed_text}"
Intent ที่ตรวจพบ: {intent}
กรุณาตอบกลับอย่างเหมาะสม"""
            
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
            
            llm_config = self.config['llm']
            llm_response = self.llm.chat(
                messages,
                temperature=llm_config.get('temperature', 0.7),
                max_tokens=llm_config.get('max_tokens', 100)
            )
            llm_time = time.time() - start_llm
            
            if llm_response:
                print(f"   💬 LLM Response: '{llm_response}'")
                print(f"   ⏱️  LLM Time: {llm_time:.2f}s")
                # Use LLM response instead of MCP response
                response_text = llm_response
            else:
                print("   ⚠️  LLM failed, using MCP response")
        else:
            print("   ⚠️  LLM not available, using MCP response")
        
        # Step 4: TTS - Text to Speech
        print("\n" + "-"*70)
        print("4️⃣ TEXT-TO-SPEECH (TTS)")
        print("-"*70)
        
        if self.components_ready.get('tts', False):
            print(f"   🔄 Synthesizing: '{response_text}'")
            start_tts = time.time()
            
            try:
                audio_file, metadata = self.tts.synthesize(response_text)
                tts_time = time.time() - start_tts
                
                print(f"   ✅ Audio generated: {audio_file}")
                print(f"   🎵 Duration: {metadata['duration']:.2f}s")
                print(f"   ⏱️  TTS Time: {tts_time:.2f}s")
                
                # Play audio
                print("\n   🔊 Playing audio response...")
                play_audio_file(audio_file)
                print("   ✅ Audio playback complete!")

                
            except Exception as e:
                print(f"   ❌ TTS failed: {e}")
        else:
            print("   ⚠️  TTS not available, skipping audio synthesis")
        
        # Summary
        print("\n" + "="*70)
        print("📊 INTEGRATION TEST SUMMARY")
        print("="*70)
        print(f"Input (STT):      '{transcribed_text}'")
        print(f"Intent (MCP):     {intent}")
        print(f"Output (TTS):     '{response_text}'")
        print(f"\nTotal Time:       {time.time() - start_stt:.2f}s")
        print("="*70)
        
        return {
            'transcribed_text': transcribed_text,
            'intent': intent,
            'response_text': response_text,
            'session_id': session_id
        }


def main():
    parser = argparse.ArgumentParser(description="Full STT-MCP-LLM-TTS Integration Test")
    parser.add_argument('--check-only', action='store_true', 
                       help='Only check component availability')
    parser.add_argument('--duration', type=int, default=5,
                       help='Audio recording duration in seconds')
    parser.add_argument('--mcp-url', type=str, default='http://localhost:8000',
                       help='MCP server URL')
    
    args = parser.parse_args()
    
    print("\n" + "🤖 "*35)
    print("  FULL INTEGRATION TEST")
    print("  STT → MCP → LLM → TTS")
    print("🤖 "*35)
    
    # Initialize test
    test = FullIntegrationTest(mcp_url=args.mcp_url if args.mcp_url != 'http://localhost:8000' else None)
    
    # Check components
    all_ready = test.check_components()
    
    if args.check_only:
        print("\n✅ Component check complete!")
        sys.exit(0 if all_ready else 1)
    
    if not all_ready:
        print("\n⚠️  Some components are not ready!")
        print("   You can still proceed, but some features may not work.")
        response = input("\n   Continue anyway? (y/n): ")
        if response.lower() != 'y':
            print("\n❌ Test aborted.")
            sys.exit(1)
    
    # Run full cycle
    print("\n" + "="*70)
    print("🚀 STARTING FULL INTEGRATION TEST")
    print("="*70)
    print("\n💡 This test will:")
    print("   1. Record your voice (Thai speech)")
    print("   2. Transcribe to text (Whisper)")
    print("   3. Detect intent (MCP)")
    print("   4. Generate response (LLM/Mock)")
    print("   5. Synthesize speech (VachanaTTS)")
    print("   6. Play audio response")
    
    input("\n📍 Press ENTER when ready to start...")
    
    result = test.run_full_cycle(audio_duration=args.duration)
    
    if result:
        print("\n✅ INTEGRATION TEST PASSED!")
    else:
        print("\n❌ INTEGRATION TEST FAILED!")
        sys.exit(1)


if __name__ == "__main__":
    main()
