"""
Feature Engineering Pipeline
Processes audio features for STT/TTS
"""

import numpy as np
from pathlib import Path


class AudioFeaturePipeline:
    """Extract features from audio for ML models"""
    
    def __init__(self, sample_rate: int = 16000):
        self.sample_rate = sample_rate
    
    def extract_features(self, audio_data: np.ndarray) -> dict:
        """
        Extract audio features
        
        Args:
            audio_data: Raw audio array
            
        Returns:
            Dictionary of features
        """
        features = {
            'duration': len(audio_data) / self.sample_rate,
            'mean_amplitude': float(np.mean(np.abs(audio_data))),
            'max_amplitude': float(np.max(np.abs(audio_data))),
            'energy': float(np.sum(audio_data ** 2))
        }
        return features
    
    def normalize_audio(self, audio_data: np.ndarray) -> np.ndarray:
        """Normalize audio to [-1, 1]"""
        max_val = np.max(np.abs(audio_data))
        if max_val > 0:
            return audio_data / max_val
        return audio_data
