"""
Whisper STT Client
Thai Language Speech-to-Text using OpenAI Whisper
"""

import whisper
import numpy as np
import torch
from typing import Dict, Optional, Tuple
from loguru import logger
import time
import soundfile as sf


class WhisperSTT:
    """
    Whisper Speech-to-Text Client for Thai Language
    """
    
    def __init__(
        self, 
        model_size: str = "small",
        device: str = "auto",
        language: str = "th"
    ):
        """
        Initialize Whisper STT
        
        Args:
            model_size: Model size (tiny, base, small, medium, large)
            device: Device to use (auto, cpu, cuda)
            language: Target language code (th for Thai)
        """
        self.model_size = model_size
        self.language = language
        
        # Auto-detect device
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        logger.info(f"Loading Whisper {model_size} model on {self.device}...")
        
        try:
            self.model = whisper.load_model(model_size, device=self.device)
            logger.success(f"Whisper model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load Whisper model: {e}")
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
            
            # Try loading audio with soundfile first (no FFmpeg needed)
            try:
                audio_data, sample_rate = sf.read(audio_path, dtype='float32')
                # Convert to mono if stereo
                if len(audio_data.shape) > 1:
                    audio_data = audio_data.mean(axis=1)
                # Resample to 16kHz if needed
                if sample_rate != 16000:
                    from scipy import signal
                    num_samples = int(len(audio_data) * 16000 / sample_rate)
                    audio_data = signal.resample(audio_data, num_samples)
                
                # Transcribe with Whisper using numpy array
                result = self.model.transcribe(
                    audio_data,
                    language=self.language,
                    task="transcribe",
                    verbose=False
                )
            except Exception as e:
                # Fallback to file path (requires FFmpeg)
                logger.warning(f"soundfile failed, trying FFmpeg: {e}")
                result = self.model.transcribe(
                    audio_path,
                    language=self.language,
                    task="transcribe",
                    verbose=False
                )
            
            duration = time.time() - start_time
            
            text = result["text"].strip()
            detected_language = result.get("language", self.language)
            
            # Calculate confidence from log probabilities
            confidence = self._calculate_confidence(result)
            
            logger.info(f"Transcribed in {duration:.2f}s: '{text[:50]}...'")
            
            return {
                "text": text,
                "confidence": confidence,
                "language": detected_language,
                "duration": duration,
                "segments": result.get("segments", [])
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
            
            # Ensure audio is float32 and mono
            if audio_data.dtype != np.float32:
                audio_data = audio_data.astype(np.float32)
            
            # Normalize to [-1, 1]
            if audio_data.max() > 1.0:
                audio_data = audio_data / 32768.0
            
            # Resample to 16kHz if needed
            if sample_rate != 16000:
                audio_data = self._resample(audio_data, sample_rate, 16000)
            
            # Transcribe
            result = self.model.transcribe(
                audio_data,
                language=self.language,
                task="transcribe",
                verbose=False
            )
            
            duration = time.time() - start_time
            
            text = result["text"].strip()
            confidence = self._calculate_confidence(result)
            
            logger.info(f"Transcribed numpy array in {duration:.2f}s")
            
            return {
                "text": text,
                "confidence": confidence,
                "language": result.get("language", self.language),
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
    
    def _calculate_confidence(self, result: Dict) -> float:
        """
        Calculate confidence score from Whisper result
        
        Args:
            result: Whisper transcription result
            
        Returns:
            Confidence score between 0 and 1
        """
        try:
            segments = result.get("segments", [])
            
            if not segments:
                return 0.5  # Default medium confidence
            
            # Average no_speech_prob across segments
            # Lower no_speech_prob = higher confidence
            no_speech_probs = []
            
            for segment in segments:
                no_speech_prob = segment.get("no_speech_prob", 0.5)
                no_speech_probs.append(no_speech_prob)
            
            # Convert to confidence (inverse of no_speech_prob)
            avg_no_speech = np.mean(no_speech_probs)
            confidence = 1.0 - avg_no_speech
            
            # Clip to [0, 1]
            confidence = max(0.0, min(1.0, confidence))
            
            return round(confidence, 3)
            
        except Exception as e:
            logger.warning(f"Confidence calculation failed: {e}")
            return 0.5
    
    def _resample(
        self,
        audio: np.ndarray,
        orig_sr: int,
        target_sr: int
    ) -> np.ndarray:
        """
        Simple resampling (linear interpolation)
        
        For production, use librosa or scipy for better quality
        """
        ratio = target_sr / orig_sr
        new_length = int(len(audio) * ratio)
        return np.interp(
            np.linspace(0, len(audio), new_length),
            np.arange(len(audio)),
            audio
        )
    
    def get_supported_formats(self) -> list:
        """Get list of supported audio formats"""
        return [
            "wav", "mp3", "m4a", "flac", "ogg", 
            "opus", "webm", "aac"
        ]
    
    def get_model_info(self) -> Dict:
        """Get model information"""
        return {
            "model_size": self.model_size,
            "device": self.device,
            "language": self.language,
            "supported_formats": self.get_supported_formats()
        }


# Convenience function for quick transcription
def transcribe_file(
    audio_path: str,
    model_size: str = "small",
    language: str = "th"
) -> Dict:
    """
    Quick transcription of audio file
    
    Args:
        audio_path: Path to audio file
        model_size: Whisper model size
        language: Target language
        
    Returns:
        Transcription result dictionary
    """
    stt = WhisperSTT(model_size=model_size, language=language)
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
        print("Usage: python whisper_client.py <audio_file>")
        print("\nExample:")
        print("python stt/whisper_client.py test_audio.wav")
