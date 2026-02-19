"""
Comprehensive API Tests
Tests all endpoints of the ROS2 Robot AI Brain API
"""

import requests
import json
import time
from typing import Dict, Optional

# API Configuration
BASE_URL = "http://localhost:8000"
SESSION_ID = f"test_session_{int(time.time())}"


class TestAPI:
    """Test suite for Robot AI Brain API"""
    
    def __init__(self, base_url: str = BASE_URL):
        self.base_url = base_url
        self.session_id = SESSION_ID
        
    def test_health_check(self) -> bool:
        """Test root health check endpoint"""
        print("\n[TEST] Health Check (GET /)")
        try:
            response = requests.get(f"{self.base_url}/")
            print(f"Status: {response.status_code}")
            print(f"Response: {json.dumps(response.json(), indent=2)}")
            
            assert response.status_code == 200
            assert response.json()["status"] == "running"
            print("✅ Health check passed")
            return True
        except Exception as e:
            print(f"❌ Health check failed: {e}")
            return False
    
    def test_context_update(self) -> bool:
        """Test context update endpoint"""
        print("\n[TEST] Context Update (POST /context/update)")
        try:
            payload = {
                "session_id": self.session_id,
                "student_id": "STD001",
                "student_name": "สมชาย ใจดี",
                "location": "อาคารเรียนรวม 1",
                "environment_data": {
                    "temperature": 28.5,
                    "humidity": 65,
                    "time_of_day": "afternoon"
                }
            }
            
            response = requests.post(
                f"{self.base_url}/context/update",
                json=payload
            )
            
            print(f"Status: {response.status_code}")
            print(f"Response: {json.dumps(response.json(), indent=2)}")
            
            assert response.status_code == 200
            assert response.json()["status"] == "success"
            print("✅ Context update passed")
            return True
        except Exception as e:
            print(f"❌ Context update failed: {e}")
            return False
    
    def test_memory_insert(self) -> bool:
        """Test memory insertion"""
        print("\n[TEST] Memory Insert (POST /memory/insert)")
        try:
            memories = [
                {
                    "text": "นักศึกษาสมชายชอบเรียนวิชาคณิตศาสตร์และมักมาอาคารเรียนรวมทุกวันจันทร์",
                    "memory_type": "student_profile",
                    "student_id": "STD001",
                    "timestamp": int(time.time())
                },
                {
                    "text": "ห้องสมุดตั้งอยู่ที่อาคาร 5 ชั้น 2 เปิดบริการจันทร์-ศุกร์ 08:00-20:00",
                    "memory_type": "knowledge",
                    "student_id": "",
                    "timestamp": int(time.time())
                },
                {
                    "text": "โรงอาหารกลางมีเมนูข้าวผัดกะเพราอร่อยมาก ราคา 45 บาท",
                    "memory_type": "knowledge",
                    "student_id": "",
                    "timestamp": int(time.time())
                }
            ]
            
            for mem in memories:
                response = requests.post(
                    f"{self.base_url}/memory/insert",
                    json=mem
                )
                print(f"Inserted: {mem['text'][:50]}...")
                print(f"Status: {response.status_code}")
                assert response.status_code == 200
            
            print("✅ Memory insert passed")
            return True
        except Exception as e:
            print(f"❌ Memory insert failed: {e}")
            return False
    
    def test_memory_search(self) -> bool:
        """Test memory search"""
        print("\n[TEST] Memory Search (POST /memory/search)")
        try:
            queries = [
                {
                    "query": "ห้องสมุดอยู่ที่ไหน",
                    "top_k": 3,
                    "memory_type": "knowledge"
                },
                {
                    "query": "สมชายชอบวิชาอะไร",
                    "top_k": 3,
                    "student_id": "STD001"
                },
                {
                    "query": "ข้าวผัดกะเพรา",
                    "top_k": 2
                }
            ]
            
            for query in queries:
                response = requests.post(
                    f"{self.base_url}/memory/search",
                    json=query
                )
                
                print(f"\nQuery: {query['query']}")
                print(f"Status: {response.status_code}")
                result = response.json()
                print(f"Found {result['count']} memories")
                
                for i, mem in enumerate(result['memories'], 1):
                    print(f"{i}. Score: {mem['score']:.3f} | {mem['text'][:60]}...")
                
                assert response.status_code == 200
                assert result["status"] == "success"
            
            print("\n✅ Memory search passed")
            return True
        except Exception as e:
            print(f"❌ Memory search failed: {e}")
            return False
    
    def test_speech_input(self) -> bool:
        """Test speech input processing"""
        print("\n[TEST] Speech Input (POST /speech/input)")
        try:
            test_inputs = [
                {
                    "session_id": self.session_id,
                    "text": "สวัสดีครับ",
                    "confidence": 0.95
                },
                {
                    "session_id": self.session_id,
                    "text": "ห้องสมุดอยู่ที่ไหนครับ",
                    "confidence": 0.90
                },
                {
                    "session_id": self.session_id,
                    "text": "พาฉันไปโรงอาหารหน่อย",
                    "confidence": 0.88
                }
            ]
            
            for inp in test_inputs:
                response = requests.post(
                    f"{self.base_url}/speech/input",
                    json=inp
                )
                
                print(f"\nUser: {inp['text']}")
                print(f"Status: {response.status_code}")
                
                if response.status_code == 200:
                    result = response.json()
                    print(f"Bot: {result['response_text']}")
                    print(f"Intent: {result['intent']}")
                    if result['should_navigate']:
                        print(f"Navigation: {result['navigation_goal']}")
                else:
                    print(f"Error: {response.text}")
                
                assert response.status_code == 200
            
            print("\n✅ Speech input passed")
            return True
        except Exception as e:
            print(f"❌ Speech input failed: {e}")
            return False
    
    def test_session_management(self) -> bool:
        """Test session get and delete"""
        print("\n[TEST] Session Management")
        try:
            # Get session
            print(f"\nGET /session/{self.session_id}")
            response = requests.get(f"{self.base_url}/session/{self.session_id}")
            print(f"Status: {response.status_code}")
            
            if response.status_code == 200:
                session = response.json()["session"]
                print(f"Session info: {json.dumps(session, indent=2, ensure_ascii=False)}")
            
            # Delete session
            print(f"\nDELETE /session/{self.session_id}")
            response = requests.delete(f"{self.base_url}/session/{self.session_id}")
            print(f"Status: {response.status_code}")
            print(f"Response: {response.json()}")
            
            assert response.status_code == 200
            print("\n✅ Session management passed")
            return True
        except Exception as e:
            print(f"❌ Session management failed: {e}")
            return False
    
    def run_all_tests(self):
        """Run all tests"""
        print("="*60)
        print("  ROS2 Robot AI Brain - API Test Suite")
        print("="*60)
        
        results = {}
        
        # Run tests in order
        results["health_check"] = self.test_health_check()
        results["context_update"] = self.test_context_update()
        results["memory_insert"] = self.test_memory_insert()
        results["memory_search"] = self.test_memory_search()
        results["speech_input"] = self.test_speech_input()
        results["session_management"] = self.test_session_management()
        
        # Summary
        print("\n" + "="*60)
        print("  Test Summary")
        print("="*60)
        
        passed = sum(1 for v in results.values() if v)
        total = len(results)
        
        for test_name, result in results.items():
            status = "✅ PASS" if result else "❌ FAIL"
            print(f"{status} - {test_name}")
        
        print(f"\nTotal: {passed}/{total} tests passed")
        
        if passed == total:
            print("\n🎉 All tests passed!")
        else:
            print(f"\n⚠️  {total - passed} test(s) failed")
        
        return passed == total


if __name__ == "__main__":
    import sys
    
    # Check if server is running
    try:
        response = requests.get(f"{BASE_URL}/", timeout=2)
    except requests.exceptions.ConnectionError:
        print(f"❌ Error: API server not running at {BASE_URL}")
        print("Please start the server with: python api/main.py")
        sys.exit(1)
    
    # Run tests
    tester = TestAPI()
    success = tester.run_all_tests()
    
    sys.exit(0 if success else 1)
