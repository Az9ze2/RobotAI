"""
Audio Utilities
Helper functions for audio recording, conversion, and playback
"""

import numpy as np
import sounddevice as sd
import soundfile as sf
from pathlib import Path
from typing import Tuple, Optional
from loguru import logger
import time


def record_audio(
    duration: float = 5.0,
    sample_rate: int = 16000,
    channels: int = 1,
    device: Optional[int] = None
) -> Tuple[np.ndarray, int]:
    """
    Record audio from microphone
    
    Args:
        duration: Recording duration in seconds
        sample_rate: Sample rate (16000 for Whisper)
        channels: Number of audio channels (1 = mono)
        device: Input device index (None = default)
        
    Returns:
        Tuple of (audio_data, sample_rate)
    """
    try:
        logger.info(f"Recording {duration}s of audio...")
        
        # Record audio
        audio_data = sd.rec(
            int(duration * sample_rate),
            samplerate=sample_rate,
            channels=channels,
            device=device,
            dtype='float32'
        )
        
        # Wait for recording to finish
        sd.wait()
        
        logger.success("Recording complete")
        
        return audio_data.flatten(), sample_rate
        
    except Exception as e:
        logger.error(f"Recording failed: {e}")
        raise


def save_audio(
    audio_data: np.ndarray,
    output_path: str,
    sample_rate: int = 16000
) -> str:
    """
    Save audio data to file
    
    Args:
        audio_data: Audio data as numpy array
        output_path: Output file path
        sample_rate: Sample rate
        
    Returns:
        Path to saved file
    """
    try:
        sf.write(output_path, audio_data, sample_rate)
        logger.info(f"Audio saved to: {output_path}")
        return output_path
    except Exception as e:
        logger.error(f"Failed to save audio: {e}")
        raise


def load_audio(audio_path: str) -> Tuple[np.ndarray, int]:
    """
    Load audio from file
    
    Args:
        audio_path: Path to audio file
        
    Returns:
        Tuple of (audio_data, sample_rate)
    """
    try:
        audio_data, sample_rate = sf.read(audio_path, dtype='float32')
        logger.info(f"Loaded audio: {audio_path}")
        return audio_data, sample_rate
    except Exception as e:
        logger.error(f"Failed to load audio: {e}")
        raise


def play_audio(
    audio_data: np.ndarray,
    sample_rate: int = 16000,
    wait: bool = True
):
    """
    Play audio through speakers
    
    Args:
        audio_data: Audio data to play
        sample_rate: Sample rate
        wait: Wait for playback to finish
    """
    try:
        logger.info("Playing audio...")
        sd.play(audio_data, sample_rate)
        
        if wait:
            sd.wait()
            logger.info("Playback complete")
            
    except Exception as e:
        logger.error(f"Playback failed: {e}")
        raise


def play_audio_file(audio_path: str, wait: bool = True):
    """
    Load and play audio file
    
    Args:
        audio_path: Path to audio file
        wait: Wait for playback to finish
    """
    audio_data, sample_rate = load_audio(audio_path)
    play_audio(audio_data, sample_rate, wait)


def convert_sample_rate(
    audio_data: np.ndarray,
    orig_sr: int,
    target_sr: int
) -> np.ndarray:
    """
    Convert audio sample rate
    
    Args:
        audio_data: Input audio
        orig_sr: Original sample rate
        target_sr: Target sample rate
        
    Returns:
        Resampled audio
    """
    try:
        import scipy.signal as signal
        
        # Calculate resampling ratio
        ratio = target_sr / orig_sr
        new_length = int(len(audio_data) * ratio)
        
        # Resample using scipy
        resampled = signal.resample(audio_data, new_length)
        
        logger.info(f"Resampled from {orig_sr}Hz to {target_sr}Hz")
        return resampled.astype(np.float32)
        
    except ImportError:
        # Fallback to linear interpolation
        logger.warning("scipy not available, using linear interpolation")
        ratio = target_sr / orig_sr
        new_length = int(len(audio_data) * ratio)
        return np.interp(
            np.linspace(0, len(audio_data), new_length),
            np.arange(len(audio_data)),
            audio_data
        ).astype(np.float32)


def get_audio_info(audio_path: str) -> dict:
    """
    Get audio file information
    
    Args:
        audio_path: Path to audio file
        
    Returns:
        Dictionary with audio metadata
    """
    try:
        info = sf.info(audio_path)
        
        return {
            "sample_rate": info.samplerate,
            "channels": info.channels,
            "duration": info.duration,
            "frames": info.frames,
            "format": info.format,
            "subtype": info.subtype
        }
    except Exception as e:
        logger.error(f"Failed to get audio info: {e}")
        raise


def list_audio_devices():
    """Print available audio input/output devices"""
    print("\n=== Available Audio Devices ===")
    devices = sd.query_devices()
    
    for i, device in enumerate(devices):
        print(f"\n[{i}] {device['name']}")
        print(f"    Inputs: {device['max_input_channels']}")
        print(f"    Outputs: {device['max_output_channels']}")
        print(f"    Sample Rate: {device['default_samplerate']}Hz")


def record_until_silence(
    silence_threshold: float = 0.01,
    silence_duration: float = 2.0,
    max_duration: float = 30.0,
    sample_rate: int = 16000,
    chunk_duration: float = 0.1
) -> Tuple[np.ndarray, int]:
    """
    Record audio until silence is detected
    
    Args:
        silence_threshold: Amplitude threshold for silence
        silence_duration: Duration of silence to stop recording
        max_duration: Maximum recording duration
        sample_rate: Sample rate
        chunk_duration: Duration of each recording chunk
        
    Returns:
        Tuple of (audio_data, sample_rate)
    """
    logger.info("Recording... Speak now. Will stop after silence.")
    
    chunks = []
    silent_chunks = 0
    chunks_needed_for_silence = int(silence_duration / chunk_duration)
    max_chunks = int(max_duration / chunk_duration)
    
    try:
        for i in range(max_chunks):
            # Record chunk
            chunk = sd.rec(
                int(chunk_duration * sample_rate),
                samplerate=sample_rate,
                channels=1,
                dtype='float32'
            )
            sd.wait()
            
            chunks.append(chunk.flatten())
            
            # Check if chunk is silent
            if np.abs(chunk).mean() < silence_threshold:
                silent_chunks += 1
            else:
                silent_chunks = 0
            
            # Stop if silence detected
            if silent_chunks >= chunks_needed_for_silence:
                logger.info("Silence detected, stopping...")
                break
        
        # Concatenate all chunks
        audio_data = np.concatenate(chunks)
        logger.success(f"Recorded {len(audio_data)/sample_rate:.2f}s")
        
        return audio_data, sample_rate
        
    except Exception as e:
        logger.error(f"Recording failed: {e}")
        raise


if __name__ == "__main__":
    # Demo: List devices
    list_audio_devices()
    
    # Demo: Record and playback
    print("\n\n=== Audio Recording Demo ===")
    print("Recording 3 seconds...")
    
    audio, sr = record_audio(duration=3.0)
    
    print(f"Recorded {len(audio)} samples at {sr}Hz")
    print("Playing back...")
    
    play_audio(audio, sr)
    
    # Save to file
    output_file = "test_recording.wav"
    save_audio(audio, output_file, sr)
    print(f"Saved to {output_file}")
    
    # Get info
    info = get_audio_info(output_file)
    print(f"\nFile info: {info}")
