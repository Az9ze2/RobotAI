"""
VachanaTTS Client Wrapper
Thai Text-to-Speech using VachanaTTS (VITS models)
"""

import sys
import os
from pathlib import Path
from typing import Dict, Optional, Tuple
from loguru import logger
import numpy as np

# Add VachanaTTS to path
VACHANA_PATH = Path("C:/Users/Win 10 Pro/Desktop/VachanaTTS")
sys.path.insert(0, str(VACHANA_PATH))

try:
    from inference.tts_with_voiceclone import generate_speech, save_audio, get_model_names
    from inference.thaicleantext import clean_thai_text
    VACHANA_AVAILABLE = True
except ImportError as e:
    logger.warning(f"VachanaTTS not available: {e}")
    VACHANA_AVAILABLE = False


class VachanaTTS:
    """
    VachanaTTS Client for Thai Speech Synthesis
    """
    
    def __init__(
        self,
        model_dir: str = None,
        default_model: str = None,
        speaking_rate: float = 1.0
    ):
        """
        Initialize VachanaTTS client
        
        Args:
            model_dir: Directory containing VITS models
            default_model: Default model to use
            speaking_rate: Speaking speed (0.1-2.0)
        """
        if not VACHANA_AVAILABLE:
            raise RuntimeError("VachanaTTS is not available. Check installation.")
        
        # Set model directory
        if model_dir is None:
            self.model_dir = str(VACHANA_PATH / "models")
        else:
            self.model_dir = model_dir
        
        self.speaking_rate = speaking_rate
        
        # Get available models
        try:
            self.available_models = get_model_names(self.model_dir)
            logger.info(f"Found {len(self.available_models)} TTS models")
            
            if not self.available_models:
                logger.warning("No TTS models found! Please download models first.")
                logger.info(f"Place models in: {self.model_dir}")
            
            # Set default model
            if default_model and default_model in self.available_models:
                self.default_model = default_model
            elif self.available_models:
                self.default_model = self.available_models[0]
            else:
                self.default_model = None
                
            logger.success(f"VachanaTTS initialized with model: {self.default_model}")
            
        except Exception as e:
            logger.error(f"Failed to initialize VachanaTTS: {e}")
            raise
    
    def synthesize(
        self,
        text: str,
        model_name: Optional[str] = None,
        speaking_rate: Optional[float] = None,
        output_path: Optional[str] = None
    ) -> Tuple[str, Dict]:
        """
        Synthesize Thai text to speech
        
        Args:
            text: Thai text to synthesize
            model_name: Model to use (default: self.default_model)
            speaking_rate: Speaking speed override
            output_path: Optional output file path
            
        Returns:
            Tuple of (audio_file_path, metadata_dict)
        """
        try:
            if not self.available_models:
                raise RuntimeError("No TTS models available")
            
            # Use default model if not specified
            if model_name is None:
                model_name = self.default_model
            
            if model_name not in self.available_models:
                raise ValueError(f"Model '{model_name}' not found. Available: {self.available_models}")
            
            # Use default speaking rate if not specified
            if speaking_rate is None:
                speaking_rate = self.speaking_rate
            
            logger.info(f"Synthesizing: '{text[:50]}...' with model {model_name}")
            
            # Clean Thai text
            cleaned_text = clean_thai_text(text)
            
            # Generate speech
            sampling_rate, audio_data = generate_speech(
                cleaned_text,
                self.model_dir,
                model_name,
                speaking_rate
            )
            
            # Save audio
            if output_path:
                audio_file = output_path
                # Save manually (implement if needed)
                import scipy.io.wavfile as wavfile
                wavfile.write(audio_file, sampling_rate, audio_data)
            else:
                audio_file = save_audio(sampling_rate, audio_data)
            
            metadata = {
                "text": text,
                "cleaned_text": cleaned_text,
                "model": model_name,
                "speaking_rate": speaking_rate,
                "sampling_rate": sampling_rate,
                "duration": len(audio_data) / sampling_rate,
                "audio_file": audio_file
            }
            
            logger.success(f"Audio generated: {audio_file}")
            
            return audio_file, metadata
            
        except Exception as e:
            logger.error(f"Speech synthesis failed: {e}")
            raise
    
    def synthesize_batch(
        self,
        texts: list,
        model_name: Optional[str] = None,
        speaking_rate: Optional[float] = None
    ) -> list:
        """
        Synthesize multiple texts in batch
        
        Args:
            texts: List of Thai texts
            model_name: Model to use
            speaking_rate: Speaking speed
            
        Returns:
            List of (audio_file, metadata) tuples
        """
        results = []
        
        for text in texts:
            try:
                audio_file, metadata = self.synthesize(
                    text, 
                    model_name, 
                    speaking_rate
                )
                results.append((audio_file, metadata))
            except Exception as e:
                logger.error(f"Failed to synthesize '{text[:30]}...': {e}")
                results.append((None, {"error": str(e)}))
        
        return results
    
    def get_available_models(self) -> list:
        """Get list of available TTS models"""
        return self.available_models
    
    def set_default_model(self, model_name: str):
        """Set default model"""
        if model_name in self.available_models:
            self.default_model = model_name
            logger.info(f"Default model set to: {model_name}")
        else:
            raise ValueError(f"Model '{model_name}' not found")
    
    def get_model_info(self) -> Dict:
        """Get TTS model information"""
        return {
            "model_dir": self.model_dir,
            "default_model": self.default_model,
            "available_models": self.available_models,
            "speaking_rate": self.speaking_rate,
            "vachana_path": str(VACHANA_PATH)
        }


# Convenience function for quick synthesis
def synthesize_text(
    text: str,
    model_name: Optional[str] = None,
    speaking_rate: float = 1.0
) -> str:
    """
    Quick text-to-speech synthesis
    
    Args:
        text: Thai text to synthesize
        model_name: TTS model name
        speaking_rate: Speaking speed
        
    Returns:
        Path to generated audio file
    """
    tts = VachanaTTS(speaking_rate=speaking_rate)
    audio_file, _ = tts.synthesize(text, model_name)
    return audio_file


if __name__ == "__main__":
    # Test the TTS
    import sys
    
    if len(sys.argv) > 1:
        text = " ".join(sys.argv[1:])
        print(f"Synthesizing: {text}")
        
        try:
            audio_file = synthesize_text(text)
            print(f"\nAudio generated: {audio_file}")
            print("Play the audio file to hear the result!")
        except Exception as e:
            print(f"Error: {e}")
    else:
        print("Usage: python vachana_client.py <thai_text>")
        print("\nExample:")
        print('python tts/vachana_client.py "สวัสดีครับ ยินดีต้อนรับสู่ระบบหุ่นยนต์"')
        print("\nNote: Make sure VachanaTTS models are downloaded and placed in the models folder")
