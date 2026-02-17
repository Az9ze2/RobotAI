"""
Quick test for KhanomTan TTS service
"""
import requests
import time

def test_khanomtan():
    """Test KhanomTan service"""
    print("Testing KhanomTan TTS service...")
    
    # Check health
    try:
        response = requests.get("http://localhost:5000/health", timeout=2)
        print(f"✅ Service is healthy: {response.json()}")
    except Exception as e:
        print(f"❌ Service not responding: {e}")
        return
    
    # Test synthesis
    test_text = "สวัสดีครับ ยินดีต้อนรับสู่สถาบันเทคโนโลยีพระจอมเกล้าเจ้าคุณทหารลาดกระบัง"
    
    print(f"\nSynthesizing: {test_text}")
    
    try:
        start = time.time()
        response = requests.post(
            "http://localhost:5000/synthesize",
            json={"text": test_text, "speaker": "Tsyncone"},
            timeout=30
        )
        duration = time.time() - start
        
        if response.status_code == 200:
            data = response.json()
            print(f"\n✅ Success!")
            print(f"  Synthesis time: {data['synthesis_time']:.3f}s")
            print(f"  Total time: {duration:.3f}s")
            print(f"  File size: {data['file_size'] / 1024:.1f} KB")
            print(f"  Speaker: {data['speaker']}")
        else:
            print(f"\n❌ Error: {response.text}")
            
    except Exception as e:
        print(f"\n❌ Request failed: {e}")

if __name__ == "__main__":
    test_khanomtan()
