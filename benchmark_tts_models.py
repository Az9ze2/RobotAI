"""
Benchmark MMS (VachanaTTS) vs KhanomTan (Coqui TTS)
"""
import requests
import time
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from tts.vachana_client import VachanaTTS

# Test sentences
TEST_SENTENCES = [
    "สวัสดีครับคุณกฤติน มีอะไรให้ผมช่วยไหมครับ",
    "ไปกันเลยครับคุณกฤติน เดินทางไปห้องแล็บทั่วนี้ใช้เวลาประมาณ 10-15 นาทีครับ",
    "ตอนนี้คุณกฤตินอยู่ที่สถาบันเทคโนโลยีพระจอมเกล้าคุณทหารลาดกระบังครับ"
]

def benchmark_mms():
    """Benchmark VachanaTTS (MMS)"""
    print("\n🔊 Benchmarking MMS (VachanaTTS)...")
    print("-" * 60)
    
    tts = VachanaTTS()
    results = []
    
    for i, text in enumerate(TEST_SENTENCES, 1):
        print(f"\nSentence {i}: {text[:50]}...")
        
        start_time = time.time()
        audio_file, metadata = tts.synthesize(text)
        synthesis_time = time.time() - start_time
        
        results.append({
            "text": text,
            "time": synthesis_time,
            "duration": metadata['duration'],
            "file": audio_file
        })
        
        print(f"  ⏱️  Synthesis: {synthesis_time:.3f}s")
        print(f"  🎵 Audio Duration: {metadata['duration']:.3f}s")
        print(f"  📊 Real-time Factor: {synthesis_time / metadata['duration']:.2f}x")
    
    avg_time = sum(r['time'] for r in results) / len(results)
    print(f"\n📊 Average Synthesis Time: {avg_time:.3f}s")
    
    return results

def benchmark_khanomtan():
    """Benchmark KhanomTan (Coqui TTS via Docker)"""
    print("\n🔊 Benchmarking KhanomTan (Coqui TTS)...")
    print("-" * 60)
    
    # Check if service is running
    try:
        response = requests.get("http://localhost:5000/health", timeout=2)
        if response.status_code != 200:
            print("❌ KhanomTan service not responding")
            return None
    except:
        print("❌ KhanomTan service not running!")
        print("   Run: cd docker/tts-service && docker-compose up -d")
        return None
    
    results = []
    
    for i, text in enumerate(TEST_SENTENCES, 1):
        print(f"\nSentence {i}: {text[:50]}...")
        
        try:
            response = requests.post(
                "http://localhost:5000/synthesize",
                json={"text": text},
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                results.append({
                    "text": text,
                    "time": data['synthesis_time'],
                    "file_size": data['file_size']
                })
                
                print(f"  ⏱️  Synthesis: {data['synthesis_time']:.3f}s")
                print(f"  📁 File Size: {data['file_size'] / 1024:.1f} KB")
            else:
                print(f"  ❌ Error: {response.text}")
                
        except Exception as e:
            print(f"  ❌ Error: {e}")
    
    if results:
        avg_time = sum(r['time'] for r in results) / len(results)
        print(f"\n📊 Average Synthesis Time: {avg_time:.3f}s")
    
    return results

def compare_results(mms_results, khanomtan_results):
    """Compare benchmark results"""
    print("\n" + "=" * 60)
    print("📊 BENCHMARK COMPARISON")
    print("=" * 60)
    
    if not khanomtan_results:
        print("\n⚠️  KhanomTan results not available")
        return
    
    mms_avg = sum(r['time'] for r in mms_results) / len(mms_results)
    khanomtan_avg = sum(r['time'] for r in khanomtan_results) / len(khanomtan_results)
    
    print(f"\n🏆 Speed Comparison:")
    print(f"  MMS (VachanaTTS):     {mms_avg:.3f}s")
    print(f"  KhanomTan (Coqui):    {khanomtan_avg:.3f}s")
    
    if mms_avg < khanomtan_avg:
        speedup = khanomtan_avg / mms_avg
        print(f"\n  ✅ MMS is {speedup:.2f}x FASTER")
    else:
        speedup = mms_avg / khanomtan_avg
        print(f"\n  ✅ KhanomTan is {speedup:.2f}x FASTER")
    
    print("\n💡 Quality Assessment:")
    print("  MMS:        ⭐⭐⭐ (Robotic but clear)")
    print("  KhanomTan:  ⭐⭐⭐⭐⭐ (Natural, native-like)")
    print("\n  Note: Listen to the generated audio files to compare quality!")

if __name__ == "__main__":
    print("=" * 60)
    print("🎤 TTS MODEL BENCHMARK")
    print("=" * 60)
    
    # Benchmark MMS
    mms_results = benchmark_mms()
    
    # Benchmark KhanomTan
    khanomtan_results = benchmark_khanomtan()
    
    # Compare
    compare_results(mms_results, khanomtan_results)
    
    print("\n✅ Benchmark complete!")
