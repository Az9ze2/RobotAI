"""
System Testing Script
Tests all components of the AI Brain
"""

import requests
import json
import time
from typing import Dict


class SystemTester:
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.session_id = f"test_session_{int(time.time())}"
    
    def test_health_check(self) -> bool:
        """Test API health"""
        print("\n=== Testing Health Check ===")
        try:
            response = requests.get(f"{self.base_url}/")
            print(f"Status: {response.status_code}")
            print(f"Response: {response.json()}")
            return response.status_code == 200
        except Exception as e:
            print(f"❌ Health check failed: {e}")
            return False
    
    def test_context_update(self) -> bool:
        """Test context update"""
        print("\n=== Testing Context Update ===")
        try:
            payload = {
                "session_id": self.session_id,
                "student_id": "STU001",
                "student_name": "สมชาย ใจดี",
                "location": "ตึกวิศวกรรม",
                "environment_data": {
                    "temperature": 28.5,
                    "humidity": 65
                }
            }
            
            response = requests.post(
                f"{self.base_url}/context/update",
                json=payload
            )
            
            print(f"Status: {response.status_code}")
            print(f"Response: {response.json()}")
            return response.status_code == 200
            
        except Exception as e:
            print(f"❌ Context update failed: {e}")
            return False
    
    def test_memory_insert(self) -> bool:
        """Test memory insertion"""
        print("\n=== Testing Memory Insert ===")
        try:
            payload = {
                "text": "นักศึกษาสมชายชอบไปห้องสมุดทุกวันจันทร์",
                "memory_type": "diary",
                "student_id": "STU001"
            }
            
            response = requests.post(
                f"{self.base_url}/memory/insert",
                json=payload
            )
            
            print(f"Status: {response.status_code}")
            print(f"Response: {response.json()}")
            
            # Insert more test memories
            test_memories = [
                "นักศึกษาสมชายมักสั่งกาแฟอเมริกาโน่ร้อน",
                "นักศึกษาสมชายเรียนวิชาปัญญาประดิษฐ์ภาคเรียนนี้",
                "สมชายชอบทำงานกลุ่มที่ชั้น 3 ห้องสมุด"
            ]
            
            for mem in test_memories:
                requests.post(
                    f"{self.base_url}/memory/insert",
                    json={
                        "text": mem,
                        "memory_type": "diary",
                        "student_id": "STU001"
                    }
                )
                time.sleep(0.5)
            
            return response.status_code == 200
            
        except Exception as e:
            print(f"❌ Memory insert failed: {e}")
            return False
    
    def test_memory_search(self) -> bool:
        """Test memory search"""
        print("\n=== Testing Memory Search ===")
        try:
            # Wait for indexing
            time.sleep(2)
            
            payload = {
                "query": "สมชายชอบอะไร",
                "top_k": 3,
                "student_id": "STU001"
            }
            
            response = requests.post(
                f"{self.base_url}/memory/search",
                json=payload
            )
            
            print(f"Status: {response.status_code}")
            result = response.json()
            print(f"Found {result.get('count')} memories:")
            
            for i, mem in enumerate(result.get('memories', []), 1):
                print(f"\n{i}. {mem['text']}")
                print(f"   Score: {mem['score']:.3f}")
            
            return response.status_code == 200
            
        except Exception as e:
            print(f"❌ Memory search failed: {e}")
            return False
    
    def test_conversation(self) -> bool:
        """Test full conversation flow"""
        print("\n=== Testing Conversation ===")
        try:
            # Test queries
            queries = [
                "สวัสดีครับ",
                "ฉันชอบดื่มกาแฟอะไร",
                "พาฉันไปห้องสมุดหน่อย"
            ]
            
            for query in queries:
                print(f"\n👤 User: {query}")
                
                payload = {
                    "session_id": self.session_id,
                    "text": query,
                    "confidence": 0.95
                }
                
                response = requests.post(
                    f"{self.base_url}/speech/input",
                    json=payload
                )
                
                if response.status_code == 200:
                    result = response.json()
                    print(f"🤖 Robot: {result['response_text']}")
                    print(f"   Intent: {result['intent']}")
                    
                    if result['should_navigate']:
                        print(f"   Navigation: {result['navigation_goal']}")
                else:
                    print(f"❌ Failed: {response.status_code}")
                
                time.sleep(1)
            
            return True
            
        except Exception as e:
            print(f"❌ Conversation test failed: {e}")
            return False
    
    def test_session_retrieval(self) -> bool:
        """Test session context retrieval"""
        print("\n=== Testing Session Retrieval ===")
        try:
            response = requests.get(
                f"{self.base_url}/session/{self.session_id}"
            )
            
            print(f"Status: {response.status_code}")
            
            if response.status_code == 200:
                session = response.json()['session']
                print(f"\nSession Info:")
                print(f"  Student: {session['student_name']}")
                print(f"  Location: {session['current_location']}")
                print(f"  Conversation turns: {len(session['conversation_history'])}")
            
            return response.status_code == 200
            
        except Exception as e:
            print(f"❌ Session retrieval failed: {e}")
            return False
    
    def run_all_tests(self):
        """Run complete test suite"""
        print("=" * 60)
        print("ROS2 MCP AI Brain - System Test Suite")
        print("=" * 60)
        
        tests = [
            ("Health Check", self.test_health_check),
            ("Context Update", self.test_context_update),
            ("Memory Insert", self.test_memory_insert),
            ("Memory Search", self.test_memory_search),
            ("Conversation", self.test_conversation),
            ("Session Retrieval", self.test_session_retrieval)
        ]
        
        results = []
        
        for test_name, test_func in tests:
            try:
                result = test_func()
                results.append((test_name, result))
            except Exception as e:
                print(f"\n❌ {test_name} crashed: {e}")
                results.append((test_name, False))
        
        # Summary
        print("\n" + "=" * 60)
        print("TEST SUMMARY")
        print("=" * 60)
        
        passed = sum(1 for _, result in results if result)
        total = len(results)
        
        for test_name, result in results:
            status = "✅ PASS" if result else "❌ FAIL"
            print(f"{status} - {test_name}")
        
        print(f"\nTotal: {passed}/{total} tests passed")
        print("=" * 60)


if __name__ == "__main__":
    tester = SystemTester()
    tester.run_all_tests()