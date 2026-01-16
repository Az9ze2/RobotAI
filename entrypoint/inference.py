"""
SAFE Continuous Voice Chat with LLM
Includes timeouts, error handling, and resource cleanup
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from stt.whisper_client import WhisperSTT
from tts.vachana_client import VachanaTTS
import sounddevice as sd
import soundfile as sf
import numpy as np
import requests
import time
import yaml
from datetime import datetime
from loguru import logger

class SafeVoiceChatBot:
    """
    Safe continuous voice conversation with LLM
    Includes timeouts and proper cleanup
    """
    
    def __init__(self):
        """Initialize voice chat bot"""
        print("\n" + "="*70)
        print("🤖 SAFE VOICE CHAT BOT - Initializing...")
        print("="*70)
        
        # Load config
        config_path = Path(__file__).parent.parent / "config" / "settings.yaml"
        with open(config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)
        
        # Initialize STT
        print("\n📥 Loading Whisper STT...")
        self.stt = WhisperSTT(
            model_size=self.config['stt']['model'],
            language=self.config['stt']['language']
        )
        print("✅ STT Ready")
        
        # Initialize TTS
        print("\n📥 Loading VachanaTTS...")
        self.tts = VachanaTTS()
        print("✅ TTS Ready")
        
        # LLM config
        self.llm_url = self.config['llm']['api_url']
        self.llm_model = self.config['llm']['model']
        
        # Check LLM connection
        print(f"\n🔗 Checking LLM connection: {self.llm_url}")
        try:
            response = requests.get(f"{self.llm_url}/api/tags", timeout=3)
            if response.status_code == 200:
                print("✅ LLM Connected (Ollama)")
            else:
                print("⚠️  LLM not responding, will use fallback responses")
                self.llm_url = None
        except Exception as e:
            print(f"⚠️  LLM not available, will use fallback responses")
            self.llm_url = None
        
        # Conversation history (limit to prevent memory issues)
        self.history = []
        self.max_history = 10
        
        print("\n" + "="*70)
        print("✅ Voice Chat Bot Ready!")
        print("="*70)
    
    def safe_record(self, duration: int = 5) -> tuple:
        """
        Safely record audio with timeout
        
        Returns:
            (audio_data, sample_rate) or (None, None) on error
        """
        try:
            sample_rate = 16000
            
            print(f"\n🎤 Recording for {duration} seconds...")
            
            # Record with sounddevice - blocking is safer
            audio_data = sd.rec(
                int(duration * sample_rate),
                samplerate=sample_rate,
                channels=1,
                dtype='float32',
                blocking=True  # Use blocking for reliability
            )
            
            return audio_data.flatten(), sample_rate
            
        except Exception as e:
            logger.error(f"Recording error: {e}")
            try:
                sd.stop()
            except:
                pass
            return None, None
    
    def safe_play(self, audio_file: str) -> bool:
        """
        Safely play audio with timeout
        
        Returns:
            True if successful, False otherwise
        """
        try:
            # Load audio
            audio_data, sample_rate = sf.read(audio_file, dtype='float32')
            
            # Play and wait - blocking is safer
            sd.play(audio_data, sample_rate)
            sd.wait()  # Wait for playback to finish
            
            return True
            
        except Exception as e:
            logger.error(f"Playback error: {e}")
            try:
                sd.stop()
            except:
                pass
            return False
    
    def listen(self, duration: int = 5) -> dict:
        """Listen and transcribe speech"""
        print("   3...")
        time.sleep(1)
        print("   2...")
        time.sleep(1)
        print("   1...")
        time.sleep(1)
        print("\n   🔴 RECORDING NOW! Speak in Thai...")
        
        # Record safely
        audio_data, sample_rate = self.safe_record(duration)
        
        if audio_data is None:
            return {
                'text': '',
                'confidence': 0.0,
                'error': 'Recording failed'
            }
        
        # Save temporarily
        temp_file = f"temp_voice_{int(time.time())}.wav"
        try:
            sf.write(temp_file, audio_data, sample_rate)
            
            # Transcribe
            print("\n🔄 Transcribing...")
            result = self.stt.transcribe_audio(temp_file)
            
            return result
            
        except Exception as e:
            logger.error(f"Transcription error: {e}")
            return {
                'text': '',
                'confidence': 0.0,
                'error': str(e)
            }
        finally:
            # Always clean up
            try:
                Path(temp_file).unlink(missing_ok=True)
            except:
                pass
    
    def get_llm_response(self, user_text: str) -> str:
        """Get response from LLM or fallback"""
        if not self.llm_url:
            return self._get_fallback_response(user_text)
        
        try:
            # Simple context from last 2 exchanges
            context = ""
            if self.history:
                for h in self.history[-2:]:
                    context += f"User: {h['user']}\nAssistant: {h['assistant']}\n"
            
            prompt = f"""คุณเป็นหุ่นยนต์บริการในมหาวิทยาลัย ตอบสั้นๆ กระชับ ไม่เกิน 3 ประโยค

{context}
User: {user_text}
Assistant:"""

            response = requests.post(
                f"{self.llm_url}/api/generate",
                json={
                    "model": self.llm_model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.7,
                        "num_predict": 100
                    }
                },
                timeout=15  # 15 second timeout
            )
            
            if response.status_code == 200:
                result = response.json()
                return result['response'].strip()
            else:
                return self._get_fallback_response(user_text)
                
        except requests.Timeout:
            print("⚠️  LLM timeout, using fallback")
            return self._get_fallback_response(user_text)
        except Exception as e:
            logger.error(f"LLM error: {e}")
            return self._get_fallback_response(user_text)
    
    def _get_fallback_response(self, user_text: str) -> str:
        """Generate fallback response"""
        text_lower = user_text.lower()
        
        if any(word in text_lower for word in ["สวัสดี", "hello", "hi"]):
            return "สวัสดีครับ ยินดีต้อนรับสู่มหาวิทยาลัย มีอะไรให้ช่วยไหมครับ"
        
        elif any(word in text_lower for word in ["ห้องสมุด", "library"]):
            return "ห้องสมุดอยู่ที่ชั้นสามของอาคารหลักครับ เปิดแปดโมงเช้าถึงห้าโมงเย็นครับ"
        
        elif any(word in text_lower for word in ["อาหาร", "กิน", "โรงอาหาร"]):
            return "โรงอาหารกลางอยู่ระหว่างคณะวิทยาศาสตร์กับคณะบริหารธุรกิจครับ"
        
        elif any(word in text_lower for word in ["ห้องน้ำ", "toilet"]):
            return "ห้องน้ำอยู่ด้านขวามือตรงทางเดินหลักครับ"
        
        elif any(word in text_lower for word in ["ขอบคุณ", "thank"]):
            return "ยินดีครับ มีอะไรให้ช่วยเพิ่มเติมไหมครับ"
        
        elif any(word in text_lower for word in ["บาย", "ลาก่อน", "bye"]):
            return "ลาก่อนครับ ขอให้มีความสุข"
        
        else:
            return f"ผมเข้าใจว่าคุณถาม '{user_text}' ครับ ลองถามเกี่ยวกับห้องสมุดหรือโรงอาหารได้ครับ"
    
    def speak(self, text: str) -> bool:
        """Synthesize and play speech"""
        try:
            print("\n🔊 Synthesizing speech...")
            
            # Synthesize
            audio_file, metadata = self.tts.synthesize(text)
            
            # Play safely
            print("▶️  Playing response...")
            success = self.safe_play(audio_file)
            
            # Clean up
            try:
                Path(audio_file).unlink(missing_ok=True)
            except:
                pass
            
            return success
            
        except Exception as e:
            logger.error(f"TTS error: {e}")
            return False
    
    def chat_once(self, turn_num: int) -> bool:
        """Single conversation turn"""
        print("\n" + "="*70)
        print(f"💬 Turn {turn_num}")
        print("="*70)
        
        # Listen
        result = self.listen(duration=5)
        user_text = result.get('text', '').strip()
        confidence = result.get('confidence', 0.0)
        
        # Display transcription
        print("\n" + "-"*70)
        print("📝 YOU SAID:")
        print("-"*70)
        print(f"   {user_text if user_text else '(no speech detected)'}")
        print(f"   Confidence: {confidence:.1%}")
        print("-"*70)
        
        if not user_text:
            print("\n⚠️  No speech detected. Try again.")
            return False
        
        # Check for exit
        if any(word in user_text.lower() for word in ["ออก", "จบ", "stop", "exit"]):
            return True
        
        # Get response
        print("\n🤖 Thinking...")
        assistant_text = self.get_llm_response(user_text)
        
        # Display response
        print("\n" + "-"*70)
        print("🤖 ROBOT SAYS:")
        print("-"*70)
        print(f"   {assistant_text}")
        print("-"*70)
        
        # Speak
        self.speak(assistant_text)
        
        # Save to history (limit size)
        self.history.append({
            'user': user_text,
            'assistant': assistant_text
        })
        if len(self.history) > self.max_history:
            self.history.pop(0)
        
        return False
    
    def run(self):
        """Run conversation loop"""
        print("\n" + "🎙️ "*35)
        print("  VOICE CHAT - CONTINUOUS CONVERSATION")
        print("  Say 'ออก' or 'stop' to end")
        print("🎙️ "*35)
        
        print("\n💡 Tips:")
        print("   - Speak clearly in Thai")
        print("   - Each recording is 5 seconds")
        print("   - Press Ctrl+C to emergency stop")
        
        input("\n Press ENTER to start...")
        
        turn = 1
        max_turns = 20  # Safety limit
        
        try:
            while turn <= max_turns:
                should_exit = self.chat_once(turn)
                
                if should_exit:
                    break
                
                turn += 1
                print("\n⏳ Ready for next question...")
                time.sleep(1)
            
            if turn > max_turns:
                print(f"\n⚠️  Reached maximum {max_turns} turns")
            
            # Summary
            print("\n" + "="*70)
            print("📊 CONVERSATION SUMMARY")
            print("="*70)
            print(f"Total exchanges: {len(self.history)}")
            print("\nConversation:")
            for i, h in enumerate(self.history, 1):
                print(f"\n{i}. YOU: {h['user']}")
                print(f"   ROBOT: {h['assistant']}")
            
            print("\n👋 Goodbye!")
            
        except KeyboardInterrupt:
            print("\n\n⚠️  Stopped by user")
            print(f"Had {len(self.history)} exchanges")
        except Exception as e:
            print(f"\n❌ Error: {e}")
            logger.exception(e)
        finally:
            # Emergency cleanup
            try:
                sd.stop()
            except:
                pass

def main():
    print("\n" + "🤖 "*35)
    print("  ROBOT AI - SAFE VOICE CHAT")
    print("  Thai Conversation with Safety Features")
    print("🤖 "*35)
    
    print("\n✅ Safety Features:")
    print("   - Recording/playback timeouts")
    print("   - Maximum 20 turns limit")
    print("   - Automatic resource cleanup")
    print("   - Error recovery")
    
    try:
        bot = SafeVoiceChatBot()
        bot.run()
    except Exception as e:
        print(f"\n❌ Failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("\n🔒 Cleanup complete")

if __name__ == "__main__":
    main()
