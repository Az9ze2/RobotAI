"""
Continuous Voice Chat with LLM
Interactive conversation using STT (Whisper) + LLM (Ollama/Typhoon) + TTS (VachanaTTS)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from stt.whisper_client import WhisperSTT
from tts.vachana_client import VachanaTTS
from utils.audio_utils import record_audio, play_audio_file, save_audio
import requests
import time
import yaml
from datetime import datetime
from loguru import logger

class VoiceChatBot:
    """
    Continuous voice conversation with LLM
    """
    
    def __init__(self):
        """Initialize voice chat bot"""
        print("\n" + "="*70)
        print("🤖 VOICE CHAT BOT - Initializing...")
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
            response = requests.get(f"{self.llm_url}/api/tags", timeout=5)
            if response.status_code == 200:
                print("✅ LLM Connected (Ollama)")
            else:
                print("⚠️  LLM not responding, will use fallback responses")
                self.llm_url = None
        except Exception as e:
            print(f"⚠️  LLM not available ({e}), will use fallback responses")
            self.llm_url = None
        
        # Conversation history
        self.history = []
        self.session_id = f"voice_chat_{int(time.time())}"
        
        print("\n" + "="*70)
        print("✅ Voice Chat Bot Ready!")
        print("="*70)
    
    def listen(self, duration: int = 5) -> dict:
        """
        Listen and transcribe speech
        
        Args:
            duration: Recording duration in seconds
            
        Returns:
            Transcription result
        """
        print(f"\n🎤 Recording for {duration} seconds...")
        print("   3...")
        time.sleep(1)
        print("   2...")
        time.sleep(1)
        print("   1...")
        time.sleep(1)
        print("\n   🔴 RECORDING NOW! Speak in Thai...")
        
        # Record
        audio_data, sample_rate = record_audio(duration=duration)
        
        # Save temporarily
        temp_file = f"temp_voice_{int(time.time())}.wav"
        save_audio(audio_data, temp_file, sample_rate)
        
        # Transcribe
        print("\n🔄 Transcribing...")
        result = self.stt.transcribe_audio(temp_file)
        
        # Clean up
        Path(temp_file).unlink(missing_ok=True)
        
        return result
    
    def get_llm_response(self, user_text: str) -> str:
        """
        Get response from LLM
        
        Args:
            user_text: User's input text
            
        Returns:
            LLM response text
        """
        if not self.llm_url:
            # Fallback responses
            return self._get_fallback_response(user_text)
        
        try:
            # Prepare context with history
            context = "\n".join([
                f"User: {h['user']}\nAssistant: {h['assistant']}"
                for h in self.history[-3:]  # Last 3 exchanges
            ])
            
            prompt = f"""คุณเป็นหุ่นยนต์บริการในมหาวิทยาลัย พูดภาษาไทยอย่างสุภาพและเป็นกันเอง

ประวัติการสนทนา:
{context}

ผู้ใช้: {user_text}
ผู้ช่วย:"""

            # Call Ollama API
            response = requests.post(
                f"{self.llm_url}/api/generate",
                json={
                    "model": self.llm_model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.7,
                        "max_tokens": 150
                    }
                },
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                return result['response'].strip()
            else:
                return self._get_fallback_response(user_text)
                
        except Exception as e:
            logger.error(f"LLM error: {e}")
            return self._get_fallback_response(user_text)
    
    def _get_fallback_response(self, user_text: str) -> str:
        """Generate fallback response when LLM is unavailable"""
        text_lower = user_text.lower()
        
        # Greeting
        if any(word in text_lower for word in ["สวัสดี", "หวัดดี", "hello", "hi"]):
            return "สวัสดีครับ ยินดีต้อนรับสู่มหาวิทยาลัย ผมเป็นหุ่นยนต์บริการ มีอะไรให้ช่วยไหมครับ"
        
        # Library
        elif any(word in text_lower for word in ["ห้องสมุด", "library", "หนังสือ", "อ่านหนังสือ"]):
            return "ห้องสมุดอยู่ที่ชั้นสามของอาคารหลักครับ เปิดทำการตั้งแต่แปดโมงเช้าถึงห้าโมงเย็น วันจันทร์ถึงศุกร์ครับ"
        
        # Food/Restaurant
        elif any(word in text_lower for word in ["อาหาร", "ร้านอาหาร", "กิน", "โรงอาหาร", "food", "restaurant"]):
            return "โรงอาหารกลางอยู่ระหว่างคณะวิทยาศาสตร์กับคณะบริหารธุรกิจครับ เปิดตั้งแต่เจ็ดโมงเช้าถึงหกโมงเย็นครับ มีอาหารหลากหลายให้เลือก"
        
        # Toilet/Restroom
        elif any(word in text_lower for word in ["ห้องน้ำ", "toilet", "restroom", "น้ำ"]):
            return "ห้องน้ำอยู่ด้านขวามือตรงทางเดินหลักครับ มีทั้งชายและหญิง สะอาดและพร้อมใช้งานครับ"
        
        # Parking
        elif any(word in text_lower for word in ["จอดรถ", "ที่จอดรถ", "parking"]):
            return "ที่จอดรถอยู่ด้านหลังอาคารครับ มีที่จอดรถทั้งรถยนต์และรถจักรยานยนต์ ฟรีสำหรับนักศึกษาและบุคลากรครับ"
        
        # Thanks
        elif any(word in text_lower for word in ["ขอบคุณ", "thank", "ขอบใจ"]):
            return "ยินดีครับ มีอะไรให้ช่วยเพิ่มเติมไหมครับ"
        
        # Goodbye
        elif any(word in text_lower for word in ["บาย", "ลาก่อน", "bye", "goodbye"]):
            return "ลาก่อนครับ ขอให้มีความสุข หากมีคำถามเพิ่มเติมสามารถกลับมาถามได้ทุกเมื่อครับ"
        
        # Default
        else:
            return f"ผมเข้าใจว่าคุณถามเกี่ยวกับ '{user_text}' ครับ ขอโทษที่อาจจะตอบไม่ถูกต้อง ขณะนี้ระบบ AI ไม่พร้อมใช้งาน หากต้องการความช่วยเหลือ กรุณาถามเกี่ยวกับห้องสมุด โรงอาหาร หรือสถานที่ต่างๆ ในมหาวิทยาลัยครับ"
    
    def speak(self, text: str):
        """
        Synthesize and play speech
        
        Args:
            text: Thai text to speak
        """
        print("\n🔊 Synthesizing speech...")
        
        # Synthesize
        audio_file, metadata = self.tts.synthesize(text)
        
        # Play
        print("▶️  Playing response...")
        play_audio_file(audio_file)
        
        # Clean up
        Path(audio_file).unlink(missing_ok=True)
    
    def chat_once(self, duration: int = 5):
        """
        Single conversation turn
        
        Args:
            duration: Recording duration
        """
        print("\n" + "="*70)
        print(f"💬 Conversation Turn {len(self.history) + 1}")
        print("="*70)
        
        # Step 1: Listen
        result = self.listen(duration)
        user_text = result['text'].strip()
        confidence = result['confidence']
        
        # Display transcription
        print("\n" + "-"*70)
        print("📝 YOU SAID:")
        print("-"*70)
        print(f"   {user_text}")
        print(f"   Confidence: {confidence:.1%}")
        print("-"*70)
        
        if not user_text:
            print("\n⚠️  No speech detected. Please speak louder or closer to mic.")
            return False
        
        # Check for exit
        if any(word in user_text.lower() for word in ["ออก", "จบ", "exit", "quit", "stop"]):
            print("\n👋 Ending conversation...")
            return True
        
        # Step 2: Get LLM response
        print("\n🤖 Thinking...")
        start_time = time.time()
        assistant_text = self.get_llm_response(user_text)
        llm_time = time.time() - start_time
        
        # Display response
        print("\n" + "-"*70)
        print("🤖 ROBOT SAYS:")
        print("-"*70)
        print(f"   {assistant_text}")
        print(f"   Processing time: {llm_time:.2f}s")
        print("-"*70)
        
        # Step 3: Speak response
        self.speak(assistant_text)
        
        # Save to history
        self.history.append({
            'user': user_text,
            'assistant': assistant_text,
            'timestamp': datetime.now().isoformat()
        })
        
        print("\n✅ Turn complete!")
        
        return False
    
    def run(self, recording_duration: int = 5):
        """
        Run continuous conversation loop
        
        Args:
            recording_duration: Duration of each recording
        """
        print("\n" + "🎙️ "*35)
        print("  VOICE CHAT - CONTINUOUS CONVERSATION")
        print("  Say 'ออก', 'จบ', or 'stop' to end conversation")
        print("🎙️ "*35)
        
        print("\n💡 Tips:")
        print("   - Speak clearly in Thai")
        print("   - Wait for the countdown before speaking")
        print("   - Each recording is 5 seconds")
        print("   - The robot will respond after processing")
        
        input("\n Press ENTER to start conversation...")
        
        try:
            should_exit = False
            while not should_exit:
                should_exit = self.chat_once(duration=recording_duration)
                
                if not should_exit:
                    print("\n⏳ Ready for next question...")
                    time.sleep(1)
            
            # End of conversation
            print("\n" + "="*70)
            print("📊 CONVERSATION SUMMARY")
            print("="*70)
            print(f"Total exchanges: {len(self.history)}")
            
            print("\nConversation history:")
            for i, h in enumerate(self.history, 1):
                print(f"\n{i}. YOU: {h['user']}")
                print(f"   ROBOT: {h['assistant']}")
            
            print("\n" + "="*70)
            print("👋 Thank you for chatting! Goodbye!")
            print("="*70)
            
        except KeyboardInterrupt:
            print("\n\n⚠️  Interrupted by user")
            print(f"Conversation had {len(self.history)} exchanges")
        except Exception as e:
            print(f"\n❌ Error: {e}")
            import traceback
            traceback.print_exc()

def main():
    """Main entry point"""
    print("\n" + "🤖 "*35)
    print("  ROBOT AI - VOICE CHAT")
    print("  Thai Language Conversation with LLM")
    print("🤖 "*35)
    
    print("\n📋 Features:")
    print("   ✅ Speech-to-Text (Whisper)")
    print("   ✅ AI Conversation (Typhoon LLM)")
    print("   ✅ Text-to-Speech (VachanaTTS)")
    print("   ✅ Continuous conversation")
    print("   ✅ Shows transcription and responses")
    
    try:
        bot = VoiceChatBot()
        bot.run(recording_duration=5)
    except Exception as e:
        print(f"\n❌ Failed to start: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
