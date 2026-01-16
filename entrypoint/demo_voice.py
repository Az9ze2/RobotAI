"""
Voice Interaction Demo
Complete demonstration of STT → Conversation → TTS pipeline
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from stt.whisper_client import WhisperSTT
from tts.vachana_client import VachanaTTS
from utils.audio_utils import record_audio, play_audio_file, save_audio
import requests
import time
from loguru import logger


class VoiceBot:
    """
    Complete voice interaction bot
    """
    
    def __init__(
        self,
        api_url: str = "http://localhost:8000",
        session_id: str = None
    ):
        """
        Initialize voice bot
        
        Args:
            api_url: FastAPI server URL
            session_id: Session ID for conversation context
        """
        self.api_url = api_url
        self.session_id = session_id or f"voice_demo_{int(time.time())}"
        
        logger.info(f"Initializing VoiceBot with session: {self.session_id}")
        
        # Initialize STT and TTS
        try:
            self.stt = WhisperSTT(model_size="small", language="th")
            logger.success("STT initialized")
        except Exception as e:
            logger.error(f"STT initialization failed: {e}")
            self.stt = None
        
        try:
            self.tts = VachanaTTS()
            logger.success("TTS initialized")
        except Exception as e:
            logger.error(f"TTS initialization failed: {e}")
            self.tts = None
        
        # Check API connection
        try:
            response = requests.get(f"{self.api_url}/")
            if response.status_code == 200:
                logger.success(f"Connected to API: {self.api_url}")
            else:
                logger.warning(f"API returned status {response.status_code}")
        except Exception as e:
            logger.error(f"Cannot connect to API: {e}")
            raise
    
    def listen(self, duration: float = 5.0) -> dict:
        """
        Listen and transcribe speech
        
        Args:
            duration: Recording duration in seconds
            
        Returns:
            Transcription result dictionary
        """
        if not self.stt:
            raise RuntimeError("STT not initialized")
        
        print(f"\n🎤 Recording for {duration} seconds... Speak now!")
        
        # Record audio
        audio_data, sample_rate = record_audio(duration=duration)
        
        # Save temporarily
        temp_file = f"temp_recording_{int(time.time())}.wav"
        save_audio(audio_data, temp_file, sample_rate)
        
        # Transcribe
        print("🔄 Transcribing...")
        result = self.stt.transcribe_audio(temp_file)
        
        # Clean up
        Path(temp_file).unlink(missing_ok=True)
        
        return result
    
    def speak(self, text: str):
        """
        Synthesize and play speech
        
        Args:
            text: Thai text to speak
        """
        if not self.tts:
            raise RuntimeError("TTS not initialized")
        
        print(f"🔊 Speaking: {text}")
        
        # Synthesize
        audio_file, metadata = self.tts.synthesize(text)
        
        # Play
        play_audio_file(audio_file)
        
        # Clean up
        Path(audio_file).unlink(missing_ok=True)
    
    def process_text(self, text: str, confidence: float) -> dict:
        """
        Send text to API and get response
        
        Args:
            text: User's text
            confidence: STT confidence
            
        Returns:
            API response dictionary
        """
        print("🤖 Processing with AI...")
        
        response = requests.post(
            f"{self.api_url}/speech/input",
            json={
                "session_id": self.session_id,
                "text": text,
                "confidence": confidence
            }
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            raise Exception(f"API error: {response.status_code} - {response.text}")
    
    def interact_once(self, duration: float = 5.0):
        """
        Single voice interaction cycle
        
        Args:
            duration: Recording duration
        """
        print("\n" + "="*60)
        print("  Voice Interaction Cycle")
        print("="*60)
        
        # Step 1: Listen
        stt_result = self.listen(duration)
        user_text = stt_result["text"]
        confidence = stt_result["confidence"]
        
        print(f"\n📝 You said: {user_text}")
        print(f"   Confidence: {confidence:.2%}")
        
        if not user_text.strip():
            print("⚠️ No speech detected. Try again!")
            return
        
        # Step 2: Process
        api_response = self.process_text(user_text, confidence)
        response_text = api_response["response_text"]
        intent = api_response["intent"]
        
        print(f"\n💬 AI response: {response_text}")
        print(f"   Intent: {intent}")
        
        if api_response.get("should_navigate"):
            nav_goal = api_response.get("navigation_goal", {})
            print(f"🗺️  Navigation: {nav_goal.get('target_location')}")
        
        # Step 3: Speak
        self.speak(response_text)
        
        print("\n✅ Interaction complete!")
    
    def continuous_mode(self, recording_duration: float = 5.0):
        """
        Continuous interaction loop
        
        Args:
            recording_duration: Duration of each recording
        """
        print("\n" + "🎙️ "*30)
        print("  CONTINUOUS VOICE INTERACTION MODE")
        print("  Press Ctrl+C to stop")
        print("🎙️ "*30)
        
        try:
            while True:
                self.interact_once(duration=recording_duration)
                
                # Wait before next interaction
                print("\n⏸️  Ready for next interaction in 2 seconds...")
                time.sleep(2)
                
        except KeyboardInterrupt:
            print("\n\n👋 Stopping continuous mode...")


def demo_full_pipeline():
    """Demonstrate complete voice pipeline"""
    print("\n" + "🤖 "*30)
    print("  VOICE INTERACTION DEMO")
    print("  ROS2 Robot AI Brain")
    print("🤖 "*30)
    
    # Initialize bot
    try:
        bot = VoiceBot()
    except Exception as e:
        print(f"\n❌ Failed to initialize: {e}")
        print("\nMake sure:")
        print("1. API server is running: python api/main.py")
        print("2. VachanaTTS models are installed")
        print("3. Microphone and speakers are connected")
        return
    
    # Run single interaction
    print("\n\n📢 Starting voice interaction demo...")
    print("You will record for 5 seconds, then get AI response.\n")
    
    input("Press Enter to start...")
    
    bot.interact_once(duration=5.0)
    
    # Ask if want continuous mode
    print("\n\nWant to try continuous mode? (y/n): ", end="")
    choice = input().strip().lower()
    
    if choice == 'y':
        bot.continuous_mode()


def demo_stt_only():
    """Test STT only"""
    print("\n=== STT Only Demo ===")
    
    stt = WhisperSTT()
    
    print("Recording 5 seconds...")
    audio_data, sample_rate = record_audio(duration=5.0)
    
    # Save and transcribe
    temp_file = "temp_stt_test.wav"
    save_audio(audio_data, temp_file, sample_rate)
    
    result = stt.transcribe_audio(temp_file)
    
    print(f"\nText: {result['text']}")
    print(f"Confidence: {result['confidence']}")
    
    Path(temp_file).unlink()


def demo_tts_only():
    """Test TTS only"""
    print("\n=== TTS Only Demo ===")
    
    tts = VachanaTTS()
    
    text = "สวัสดีครับ ยินดีต้อนรับสู่ระบบหุ่นยนต์บริการ"
    print(f"Synthesizing: {text}")
    
    audio_file, metadata = tts.synthesize(text)
    print(f"Audio generated: {audio_file}")
    print(f"Duration: {metadata['duration']:.2f}s")
    
    print("Playing...")
    play_audio_file(audio_file)
    
    Path(audio_file).unlink()


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        mode = sys.argv[1].lower()
        
        if mode == "stt":
            demo_stt_only()
        elif mode == "tts":
            demo_tts_only()
        elif mode == "full":
            demo_full_pipeline()
        else:
            print("Usage: python demo_voice.py [stt|tts|full]")
    else:
        # Default: full demo
        demo_full_pipeline()
