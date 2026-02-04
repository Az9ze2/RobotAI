"""
Audio Recorder Utility
Record audio from microphone for STT testing
"""

import sounddevice as sd
import soundfile as sf
import numpy as np
from loguru import logger
import time
from typing import Optional, Tuple


class AudioRecorder:
    """
    Simple audio recorder for microphone input
    """
    
    def __init__(
        self,
        sample_rate: int = 16000,
        channels: int = 1,
        dtype: str = 'float32'
    ):
        """
        Initialize audio recorder
        
        Args:
            sample_rate: Sample rate in Hz (default: 16000)
            channels: Number of channels (1=mono, 2=stereo)
            dtype: Data type for recording
        """
        self.sample_rate = sample_rate
        self.channels = channels
        self.dtype = dtype
        
        logger.info(f"AudioRecorder initialized: {sample_rate}Hz, {channels} channel(s)")
    
    def record(
        self,
        duration: float = 5.0,
        show_countdown: bool = True
    ) -> np.ndarray:
        """
        Record audio from microphone
        
        Args:
            duration: Recording duration in seconds
            show_countdown: Show countdown before recording
            
        Returns:
            Recorded audio as numpy array
        """
        try:
            if show_countdown:
                print(f"\n🎤 Recording will start in...")
                for i in range(3, 0, -1):
                    print(f"   {i}...")
                    time.sleep(1)
                print("   🔴 RECORDING NOW! Speak into your microphone...")
            else:
                print(f"🔴 Recording for {duration} seconds...")
            
            # Record audio
            recording = sd.rec(
                int(duration * self.sample_rate),
                samplerate=self.sample_rate,
                channels=self.channels,
                dtype=self.dtype
            )
            
            # Wait for recording to complete
            sd.wait()
            
            print(f"✅ Recording complete! ({duration}s)")
            
            # Convert to mono if stereo
            if self.channels > 1:
                recording = recording.mean(axis=1)
            
            return recording.flatten()
            
        except Exception as e:
            logger.error(f"Recording failed: {e}")
            raise
    
    def record_and_save(
        self,
        output_path: str,
        duration: float = 5.0,
        show_countdown: bool = True
    ) -> Tuple[np.ndarray, str]:
        """
        Record audio and save to file
        
        Args:
            output_path: Path to save audio file
            duration: Recording duration in seconds
            show_countdown: Show countdown before recording
            
        Returns:
            Tuple of (audio_data, file_path)
        """
        try:
            # Record
            audio_data = self.record(duration, show_countdown)
            
            # Save to file
            sf.write(output_path, audio_data, self.sample_rate)
            logger.info(f"Audio saved to: {output_path}")
            
            return audio_data, output_path
            
        except Exception as e:
            logger.error(f"Record and save failed: {e}")
            raise
    
    def get_device_info(self) -> dict:
        """Get information about available audio devices"""
        try:
            devices = sd.query_devices()
            default_input = sd.query_devices(kind='input')
            
            return {
                'devices': devices,
                'default_input': default_input
            }
        except Exception as e:
            logger.error(f"Failed to query devices: {e}")
            return {}
    
    def list_devices(self):
        """Print list of available audio devices"""
        try:
            print("\n🎤 Available Audio Devices:")
            print("=" * 60)
            devices = sd.query_devices()
            for i, device in enumerate(devices):
                device_type = []
                if device['max_input_channels'] > 0:
                    device_type.append('INPUT')
                if device['max_output_channels'] > 0:
                    device_type.append('OUTPUT')
                
                print(f"{i}: {device['name']}")
                print(f"   Type: {', '.join(device_type)}")
                print(f"   Channels: In={device['max_input_channels']}, Out={device['max_output_channels']}")
                print(f"   Sample Rate: {device['default_samplerate']}Hz")
                print()
            
            default_input = sd.query_devices(kind='input')
            print(f"Default Input Device: {default_input['name']}")
            print("=" * 60)
            
        except Exception as e:
            logger.error(f"Failed to list devices: {e}")


# Convenience function
def record_audio(
    duration: float = 5.0,
    sample_rate: int = 16000,
    output_path: Optional[str] = None,
    show_countdown: bool = True
) -> Tuple[np.ndarray, Optional[str]]:
    """
    Quick audio recording function
    
    Args:
        duration: Recording duration in seconds
        sample_rate: Sample rate in Hz
        output_path: Optional path to save audio file
        show_countdown: Show countdown before recording
        
    Returns:
        Tuple of (audio_data, file_path or None)
    """
    recorder = AudioRecorder(sample_rate=sample_rate)
    
    if output_path:
        return recorder.record_and_save(output_path, duration, show_countdown)
    else:
        audio_data = recorder.record(duration, show_countdown)
        return audio_data, None


if __name__ == "__main__":
    # Test the recorder
    print("🎤 Audio Recorder Test")
    print("=" * 60)
    
    recorder = AudioRecorder()
    recorder.list_devices()
    
    # Record 5 seconds
    audio_data, file_path = record_audio(
        duration=5.0,
        output_path="test_recording.wav",
        show_countdown=True
    )
    
    print(f"\n✅ Test complete!")
    print(f"   Audio shape: {audio_data.shape}")
    print(f"   Saved to: {file_path}")
