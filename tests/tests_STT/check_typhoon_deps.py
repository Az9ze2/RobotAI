"""Check if Typhoon ASR dependencies are installed"""
import sys

print("=" * 60)
print("Checking Typhoon ASR Dependencies")
print("=" * 60)

# Check sounddevice
try:
    import sounddevice as sd
    print(f"✅ sounddevice: {sd.__version__}")
except ImportError:
    print("❌ sounddevice: NOT INSTALLED")
    print("   Install: pip install sounddevice")

# Check NeMo
try:
    import nemo
    print(f"✅ nemo-toolkit: {nemo.__version__}")
except ImportError:
    print("❌ nemo-toolkit: NOT INSTALLED")
    print("   Install: pip install nemo-toolkit[asr]")

# Check librosa
try:
    import librosa
    print(f"✅ librosa: {librosa.__version__}")
except ImportError:
    print("❌ librosa: NOT INSTALLED")
    print("   Install: pip install librosa")

# Check soundfile
try:
    import soundfile as sf
    print(f"✅ soundfile: {sf.__version__}")
except ImportError:
    print("❌ soundfile: NOT INSTALLED")
    print("   Install: pip install soundfile")

# Check torch
try:
    import torch
    print(f"✅ torch: {torch.__version__}")
    print(f"   CUDA available: {torch.cuda.is_available()}")
except ImportError:
    print("❌ torch: NOT INSTALLED")
    print("   Install: pip install torch")

print("=" * 60)
print("\nTo install all Typhoon ASR dependencies:")
print("pip install -r requirements-typhoon.txt")
print("=" * 60)
