"""
Inference Pipeline
End-to-end STT → LLM → TTS inference
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from stt.typhoon_asr_client import TyphoonASR
from tts.vachana_client import VachanaTTS
from llm.ollama_client import OllamaClient


class VoiceInferencePipeline:
    """Complete voice interaction inference pipeline"""
    
    def __init__(self, config: dict):
        """Initialize all components"""
        self.stt = TyphoonASR(
            model_name="scb10x/typhoon-asr-realtime",
            language=config['stt']['language'],
            device="auto"
        )
        self.tts = VachanaTTS()
        self.llm = OllamaClient(
            model=config['llm']['model'],
            api_url=config['llm']['api_url']
        )
    
    def process_audio(self, audio_file: str) -> dict:
        """
        Process audio through full pipeline
        
        Args:
            audio_file: Path to audio file
            
        Returns:
            dict with text, response, audio_file
        """
        # STT
        stt_result = self.stt.transcribe_audio(audio_file)
        user_text = stt_result['text']
        
        # LLM
        response = self.llm.generate(user_text)
        
        # TTS
        audio_out, metadata = self.tts.synthesize(response)
        
        return {
            'input_text': user_text,
            'response_text': response,
            'audio_file': audio_out,
            'confidence': stt_result['confidence']
        }
