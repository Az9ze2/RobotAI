
import pytest
from fastapi.testclient import TestClient
from src.mcp.server import app

client = TestClient(app)

def test_health_check():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"status": "running", "service": "mcp"}

def test_custom_flow():
    # 1. Simulate Face Detection
    face_payload = {
        "student_id": "64010001",
        "confidence": 0.98,
        "timestamp": 1234567890.0
    }
    response = client.post("/api/v1/perception/face", json=face_payload)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    session_id = data["session_id"]
    assert "session_64010001" in session_id

    # 2. Simulate Speech (Navigation Intent)
    speech_payload = {
        "session_id": session_id,
        "text": "พาไปห้องแลปหน่อยครับ",
        "confidence": 0.95
    }
    response = client.post("/api/v1/perception/speech", json=speech_payload)
    assert response.status_code == 200
    data = response.json()
    assert data["intent"] == "navigation"
    assert data["should_navigate"] is True
    assert data["navigation_goal"] == "AI_LAB"
    
    # 3. Simulate Speech (Conversation Intent)
    speech_payload = {
        "session_id": session_id,
        "text": "สวัสดีครับ",
        "confidence": 0.99
    }
    response = client.post("/api/v1/perception/speech", json=speech_payload)
    assert response.status_code == 200
    data = response.json()
    assert data["intent"] == "conversation"
    assert data["should_navigate"] is False

if __name__ == "__main__":
    test_health_check()
    test_custom_flow()
    print("All custom flow tests passed!")
