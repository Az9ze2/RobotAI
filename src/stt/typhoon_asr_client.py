"""
Typhoon ASR STT Client
Thai Language Speech-to-Text using Typhoon ASR Real-Time
"""

import numpy as np
import torch
from typing import Dict, Optional
from loguru import logger
import time
import soundfile as sf
import librosa
from pathlib import Path
import os


class TyphoonASR:
    """
    Typhoon ASR Speech-to-Text Client for Thai Language
    """
    
    def __init__(
        self, 
        model_name: str = "scb10x/typhoon-asr-realtime",
        device: str = "auto",
        language: str = "th"
    ):
        """
        Initialize Typhoon ASR
        
        Args:
            model_name: Model name (scb10x/typhoon-asr-realtime)
            device: Device to use (auto, cpu, cuda)
            language: Target language code (th for Thai)
        """
        self.model_name = model_name
        self.language = language
        
        # Auto-detect device
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        logger.info(f"Loading Typhoon ASR model '{model_name}' on {self.device}...")
        
        try:
            # Import NeMo ASR
            import nemo.collections.asr as nemo_asr
            
            # Load model
            self.model = nemo_asr.models.ASRModel.from_pretrained(
                model_name=model_name,
                map_location=torch.device(self.device)
            )
            logger.success(f"Typhoon ASR model loaded successfully")
        except ImportError as e:
            logger.error(f"Failed to import NeMo toolkit. Please install: pip install nemo-toolkit[asr]")
            raise
        except Exception as e:
            logger.error(f"Failed to load Typhoon ASR model: {e}")
            raise
    
    def transcribe_audio(
        self, 
        audio_path: str,
        return_confidence: bool = True
    ) -> Dict:
        """
        Transcribe audio file to text
        
        Args:
            audio_path: Path to audio file (wav, mp3, etc.)
            return_confidence: Whether to calculate confidence score
            
        Returns:
            Dictionary with 'text', 'confidence', 'language', 'duration'
        """
        try:
            start_time = time.time()
            
            # Prepare audio (resample to 16kHz if needed)
            processed_path = self._prepare_audio(audio_path)
            
            # Transcribe with Typhoon ASR
            transcriptions = self.model.transcribe(audio=[processed_path])
            
            duration = time.time() - start_time
            
            # Extract text from result (handle both string and Hypothesis object)
            if transcriptions and len(transcriptions) > 0:
                result = transcriptions[0]
                # Check if it's a Hypothesis object
                if hasattr(result, 'text'):
                    text = result.text
                elif isinstance(result, str):
                    text = result
                else:
                    text = str(result)
            else:
                text = ""
            
            text = text.strip() if text else ""
            
            # Calculate confidence (Typhoon ASR doesn't provide direct confidence)
            # We'll use a heuristic based on text length and processing time
            confidence = self._estimate_confidence(text, duration)
            
            logger.info(f"Transcribed in {duration:.2f}s: '{text[:50]}...'")
            
            # Clean up temporary file if created
            if processed_path != audio_path and os.path.exists(processed_path):
                os.remove(processed_path)
            
            return {
                "text": text,
                "confidence": confidence,
                "language": self.language,
                "duration": duration,
                "segments": []
            }
            
        except Exception as e:
            logger.error(f"Transcription failed: {e}")
            return {
                "text": "",
                "confidence": 0.0,
                "language": self.language,
                "duration": 0.0,
                "error": str(e)
            }
    
    def transcribe_numpy(
        self,
        audio_data: np.ndarray,
        sample_rate: int = 16000
    ) -> Dict:
        """
        Transcribe numpy audio array
        
        Args:
            audio_data: Audio data as numpy array
            sample_rate: Sample rate of audio
            
        Returns:
            Dictionary with transcription results
        """
        try:
            start_time = time.time()
            
            # Save to temporary WAV file
            temp_path = "temp_audio_typhoon.wav"
            
            # Ensure audio is float32 and mono
            if audio_data.dtype != np.float32:
                audio_data = audio_data.astype(np.float32)
            
            # Normalize to [-1, 1]
            if audio_data.max() > 1.0:
                audio_data = audio_data / 32768.0
            
            # Resample to 16kHz if needed
            if sample_rate != 16000:
                audio_data = librosa.resample(audio_data, orig_sr=sample_rate, target_sr=16000)
                sample_rate = 16000
            
            # Save to file
            sf.write(temp_path, audio_data, sample_rate)
            
            # Transcribe
            transcriptions = self.model.transcribe(audio=[temp_path])
            
            duration = time.time() - start_time
            
            # Extract text from result (handle both string and Hypothesis object)
            if transcriptions and len(transcriptions) > 0:
                result = transcriptions[0]
                # Check if it's a Hypothesis object
                if hasattr(result, 'text'):
                    text = result.text
                elif isinstance(result, str):
                    text = result
                else:
                    text = str(result)
            else:
                text = ""
            
            text = text.strip() if text else ""
            
            confidence = self._estimate_confidence(text, duration)
            
            logger.info(f"Transcribed numpy array in {duration:.2f}s")
            
            # Clean up
            if os.path.exists(temp_path):
                os.remove(temp_path)
            
            return {
                "text": text,
                "confidence": confidence,
                "language": self.language,
                "duration": duration
            }
            
        except Exception as e:
            logger.error(f"Numpy transcription failed: {e}")
            return {
                "text": "",
                "confidence": 0.0,
                "language": self.language,
                "duration": 0.0,
                "error": str(e)
            }
    
    def _prepare_audio(self, audio_path: str, target_sr: int = 16000) -> str:
        """
        Prepare audio file for Typhoon ASR (resample to 16kHz if needed)
        
        Args:
            audio_path: Source audio file path
            target_sr: Target sample rate
            
        Returns:
            Path to processed audio file
        """
        try:
            # Check if file exists
            if not os.path.exists(audio_path):
                raise FileNotFoundError(f"Audio file not found: {audio_path}")
            
            # Load audio
            y, sr = librosa.load(audio_path, sr=None)
            
            # If already 16kHz, return original path
            if sr == target_sr:
                return audio_path
            
            # Resample
            y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
            
            # Normalize
            y = y / (max(abs(y)) + 1e-8)
            
            # Save to temporary file
            temp_path = f"processed_{Path(audio_path).stem}.wav"
            sf.write(temp_path, y, target_sr)
            
            return temp_path
            
        except Exception as e:
            logger.error(f"Audio preparation failed: {e}")
            return audio_path
    
    def _estimate_confidence(self, text: str, duration: float) -> float:
        """
        Estimate confidence score based on text characteristics
        
        Args:
            text: Transcribed text
            duration: Processing duration
            
        Returns:
            Confidence score between 0 and 1
        """
        try:
            # Heuristic confidence estimation
            # Factors: text length, processing speed
            
            if not text:
                return 0.0
            
            # Base confidence
            confidence = 0.8
            
            # Adjust based on text length (very short or very long might be less confident)
            text_len = len(text)
            if text_len < 5:
                confidence -= 0.2
            elif text_len > 200:
                confidence -= 0.1
            
            # Adjust based on processing time (faster might be more confident)
            if duration < 0.5:
                confidence += 0.1
            elif duration > 5.0:
                confidence -= 0.1
            
            # Clip to [0, 1]
            confidence = max(0.0, min(1.0, confidence))
            
            return round(confidence, 3)
            
        except Exception as e:
            logger.warning(f"Confidence estimation failed: {e}")
            return 0.5
    
    def get_supported_formats(self) -> list:
        """Get list of supported audio formats"""
        return [
            "wav", "mp3", "m4a", "flac", "ogg", 
            "opus", "webm", "aac"
        ]
    
    def get_model_info(self) -> Dict:
        """Get model information"""
        return {
            "model_name": self.model_name,
            "device": self.device,
            "language": self.language,
            "supported_formats": self.get_supported_formats()
        }


# Convenience function for quick transcription
def transcribe_file(
    audio_path: str,
    model_name: str = "scb10x/typhoon-asr-realtime",
    language: str = "th"
) -> Dict:
    """
    Quick transcription of audio file
    
    Args:
        audio_path: Path to audio file
        model_name: Typhoon ASR model name
        language: Target language
        
    Returns:
        Transcription result dictionary
    """
    stt = TyphoonASR(model_name=model_name, language=language)
    return stt.transcribe_audio(audio_path)


if __name__ == "__main__":
    # Test the STT
    import sys
    
    if len(sys.argv) > 1:
        audio_file = sys.argv[1]
        print(f"Transcribing: {audio_file}")
        
        result = transcribe_file(audio_file)
        
        print(f"\nText: {result['text']}")
        print(f"Confidence: {result['confidence']}")
        print(f"Language: {result['language']}")
        print(f"Duration: {result['duration']:.2f}s")
    else:
        print("Usage: python typhoon_asr_client.py <audio_file>")
        print("\nExample:")
        print("python stt/typhoon_asr_client.py test_audio.wav")
