"""
Manual test script to verify MCP server is working
Run this while the server is running in another terminal
"""

import requests
import json

BASE_URL = "http://localhost:8000"

def test_health():
    print("\n=== Testing Health Check ===")
    response = requests.get(f"{BASE_URL}/")
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")

def test_face_detection():
    print("\n=== Testing Face Detection ===")
    payload = {
        "student_id": "64010001",
        "confidence": 0.98,
        "timestamp": 1234567890.0
    }
    response = requests.post(f"{BASE_URL}/api/v1/perception/face", json=payload)
    print(f"Status: {response.status_code}")
    result = response.json()
    print(f"Response: {json.dumps(result, indent=2)}")
    return result.get("session_id")

def test_speech_navigation(session_id):
    print("\n=== Testing Speech (Navigation) ===")
    payload = {
        "session_id": session_id,
        "text": "พาไปห้องแลปหน่อยครับ",
        "confidence": 0.95
    }
    response = requests.post(f"{BASE_URL}/api/v1/perception/speech", json=payload)
    print(f"Status: {response.status_code}")
    result = response.json()
    print(f"Response: {json.dumps(result, indent=2, ensure_ascii=False)}")

def test_speech_conversation(session_id):
    print("\n=== Testing Speech (Conversation) ===")
    payload = {
        "session_id": session_id,
        "text": "สวัสดีครับ",
        "confidence": 0.99
    }
    response = requests.post(f"{BASE_URL}/api/v1/perception/speech", json=payload)
    print(f"Status: {response.status_code}")
    result = response.json()
    print(f"Response: {json.dumps(result, indent=2, ensure_ascii=False)}")

if __name__ == "__main__":
    try:
        test_health()
        session_id = test_face_detection()
        test_speech_navigation(session_id)
        test_speech_conversation(session_id)
        print("\n✅ All manual tests completed!")
    except requests.exceptions.ConnectionError:
        print("\n❌ Error: Cannot connect to server. Make sure the server is running:")
        print("   python src/mcp/server.py")
    except Exception as e:
        print(f"\n❌ Error: {e}")
