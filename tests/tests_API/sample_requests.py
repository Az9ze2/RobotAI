"""
Sample Request Scripts
Demonstrates how to use the Robot AI Brain API with Thai language examples
"""

import requests
import json
import time
from typing import Dict, List

BASE_URL = "http://localhost:8000"


class SampleRequests:
    """Sample API request demonstrations"""
    
    def __init__(self, base_url: str = BASE_URL):
        self.base_url = base_url
        self.session_id = f"demo_session_{int(time.time())}"
        
    def print_response(self, title: str, response: requests.Response):
        """Pretty print API response"""
        print(f"\n{'='*60}")
        print(f"  {title}")
        print('='*60)
        print(f"Status Code: {response.status_code}")
        if response.status_code == 200:
            print(f"Response:\n{json.dumps(response.json(), indent=2, ensure_ascii=False)}")
        else:
            print(f"Error: {response.text}")
    
    def example_1_simple_conversation(self):
        """Example 1: Simple greeting conversation"""
        print("\n\n" + "🔷"*30)
        print("EXAMPLE 1: Simple Conversation Flow")
        print("🔷"*30)
        
        # Step 1: Update context
        context_data = {
            "session_id": self.session_id,
            "student_id": "STD12345",
            "student_name": "นภัสวรรณ์ สุขใจ",
            "location": "อาคารวิทยาศาสตร์"
        }
        
        response = requests.post(
            f"{self.base_url}/context/update",
            json=context_data
        )
        self.print_response("Step 1: Initialize Context", response)
        
        # Step 2: Send greeting
        speech_data = {
            "session_id": self.session_id,
            "text": "สวัสดีค่ะน้องบอท",
            "confidence": 0.95
        }
        
        response = requests.post(
            f"{self.base_url}/speech/input",
            json=speech_data
        )
        self.print_response("Step 2: User Greeting", response)
        
        # Step 3: Ask about location
        speech_data = {
            "session_id": self.session_id,
            "text": "ตอนนี้ฉันอยู่ที่ไหน",
            "confidence": 0.92
        }
        
        response = requests.post(
            f"{self.base_url}/speech/input",
            json=speech_data
        )
        self.print_response("Step 3: Ask Current Location", response)
    
    def example_2_navigation_request(self):
        """Example 2: Navigation to cafeteria"""
        print("\n\n" + "🔷"*30)
        print("EXAMPLE 2: Navigation Request")
        print("🔷"*30)
        
        # Insert knowledge about cafeteria first
        memory_data = {
            "text": "โรงอาหารกลางอยู่ติดกับอาคารเรียนรวม 2 มีอาหารหลากหลาย ราคาเริ่มต้น 30 บาท เปิด 07:00-18:00",
            "memory_type": "knowledge",
            "student_id": "",
            "timestamp": int(time.time())
        }
        
        response = requests.post(
            f"{self.base_url}/memory/insert",
            json=memory_data
        )
        self.print_response("Step 1: Insert Cafeteria Knowledge", response)
        
        # Navigation request
        speech_data = {
            "session_id": self.session_id,
            "text": "พาฉันไปโรงอาหารกลางหน่อย",
            "confidence": 0.89
        }
        
        response = requests.post(
            f"{self.base_url}/speech/input",
            json=speech_data
        )
        self.print_response("Step 2: Request Navigation", response)
    
    def example_3_memory_and_personalization(self):
        """Example 3: Personalized memory retrieval"""
        print("\n\n" + "🔷"*30)
        print("EXAMPLE 3: Memory and Personalization")
        print("🔷"*30)
        
        student_id = "STD67890"
        student_name = "วิชัย เรียนดี"
        
        # Insert student memories
        memories = [
            {
                "text": f"{student_name} ชอบเล่นบาสเกตบอล และเป็นสมาชิกชมรมกีฬา",
                "memory_type": "student_profile",
                "student_id": student_id
            },
            {
                "text": f"{student_name} มักมาใช้ห้องสมุดทุกวันพุธและศุกร์ เพื่ออ่านหนังสือเตรียมสอบ",
                "memory_type": "diary",
                "student_id": student_id
            },
            {
                "text": f"{student_name} เคยถามเกี่ยวกับสนามบาสเกตบอลและชอบเล่นช่วงเย็น",
                "memory_type": "diary",
                "student_id": student_id
            }
        ]
        
        for mem in memories:
            mem["timestamp"] = int(time.time())
            response = requests.post(
                f"{self.base_url}/memory/insert",
                json=mem
            )
            print(f"Inserted: {mem['text'][:50]}...")
        
        # Update context with student
        context_data = {
            "session_id": f"session_{student_id}",
            "student_id": student_id,
            "student_name": student_name,
            "location": "ลานกีฬา"
        }
        
        response = requests.post(
            f"{self.base_url}/context/update",
            json=context_data
        )
        self.print_response("Step 1: Set Student Context", response)
        
        # Search memories
        search_data = {
            "query": "วิชัยชอบทำอะไร",
            "top_k": 5,
            "student_id": student_id
        }
        
        response = requests.post(
            f"{self.base_url}/memory/search",
            json=search_data
        )
        self.print_response("Step 2: Search Student Memories", response)
        
        # Personalized conversation
        speech_data = {
            "session_id": f"session_{student_id}",
            "text": "สนามบาสเกตบอลว่างไหม",
            "confidence": 0.91
        }
        
        response = requests.post(
            f"{self.base_url}/speech/input",
            json=speech_data
        )
        self.print_response("Step 3: Personalized Response", response)
    
    def example_4_campus_knowledge_qa(self):
        """Example 4: Campus knowledge Q&A"""
        print("\n\n" + "🔷"*30)
        print("EXAMPLE 4: Campus Knowledge Q&A")
        print("🔷"*30)
        
        # Insert various campus knowledge
        knowledge_base = [
            "ห้องสมุดกลางตั้งอยู่ที่อาคาร 7 ชั้น 1-3 เปิดจันทร์-ศุกร์ 08:00-20:00 เสาร์-อาทิตย์ 09:00-17:00",
            "สำนักงานทะเบียนอยู่ที่อาคารเรียนรวม 1 ชั้น 1 เปิดทำการ 08:30-16:30",
            "โรงพยาบาลมหาวิทยาลัยอยู่ติดกับประตูทางเข้าหลัก มีแพทย์และพยาบาลตลอด 24 ชั่วโมง",
            "สนามกีฬากลางมีสนามฟุตบอล บาสเกตบอล และลู่วิ่ง เปิดให้ใช้ฟรีสำหรับนักศึกษา",
            "ศูนย์คอมพิวเตอร์อยู่อาคาร IT ชั้น 2 มี WiFi ความเร็วสูง และให้บริการพิมพ์เอกสาร"
        ]
        
        print("\nInserting campus knowledge...")
        for knowledge in knowledge_base:
            memory_data = {
                "text": knowledge,
                "memory_type": "knowledge",
                "student_id": "",
                "timestamp": int(time.time())
            }
            requests.post(f"{self.base_url}/memory/insert", json=memory_data)
            time.sleep(0.1)  # Small delay for indexing
        
        # Ask questions
        questions = [
            "ห้องสมุดเปิดถึงกี่โมง",
            "ทะเบียนอยู่ที่ไหน",
            "มีสนามกีฬาไหมครับ"
        ]
        
        for question in questions:
            # Search relevant knowledge
            search_data = {
                "query": question,
                "top_k": 2,
                "memory_type": "knowledge"
            }
            
            response = requests.post(
                f"{self.base_url}/memory/search",
                json=search_data
            )
            self.print_response(f"Q: {question}", response)
    
    def example_5_low_confidence_handling(self):
        """Example 5: Low confidence speech handling"""
        print("\n\n" + "🔷"*30)
        print("EXAMPLE 5: Low Confidence Speech Handling")
        print("🔷"*30)
        
        # Low confidence input
        speech_data = {
            "session_id": self.session_id,
            "text": "...ไป...อาคาร...สอง",
            "confidence": 0.55  # Below threshold (0.7)
        }
        
        response = requests.post(
            f"{self.base_url}/speech/input",
            json=speech_data
        )
        self.print_response("Low Confidence Input", response)
    
    def example_6_session_management(self):
        """Example 6: Session management"""
        print("\n\n" + "🔷"*30)
        print("EXAMPLE 6: Session Management")
        print("🔷"*30)
        
        # Create and use session
        session_id = f"managed_session_{int(time.time())}"
        
        # Create session with context
        context_data = {
            "session_id": session_id,
            "student_id": "STD99999",
            "student_name": "ทดสอบ ระบบ",
            "location": "ห้องทดลอง"
        }
        
        response = requests.post(
            f"{self.base_url}/context/update",
            json=context_data
        )
        self.print_response("Step 1: Create Session", response)
        
        # Get session info
        response = requests.get(f"{self.base_url}/session/{session_id}")
        self.print_response("Step 2: Get Session Info", response)
        
        # Clear session
        response = requests.delete(f"{self.base_url}/session/{session_id}")
        self.print_response("Step 3: Clear Session", response)
    
    def run_all_examples(self):
        """Run all example demonstrations"""
        print("\n" + "="*60)
        print("  ROS2 Robot AI Brain - Sample Request Demonstrations")
        print("="*60)
        
        try:
            self.example_1_simple_conversation()
            time.sleep(1)
            
            self.example_2_navigation_request()
            time.sleep(1)
            
            self.example_3_memory_and_personalization()
            time.sleep(1)
            
            self.example_4_campus_knowledge_qa()
            time.sleep(1)
            
            self.example_5_low_confidence_handling()
            time.sleep(1)
            
            self.example_6_session_management()
            
            print("\n\n" + "="*60)
            print("  ✅ All examples completed successfully!")
            print("="*60)
            
        except Exception as e:
            print(f"\n❌ Error running examples: {e}")


if __name__ == "__main__":
    import sys
    
    # Check if server is running
    try:
        response = requests.get(f"{BASE_URL}/", timeout=2)
        print(f"✅ Connected to API server at {BASE_URL}")
    except requests.exceptions.ConnectionError:
        print(f"❌ Error: API server not running at {BASE_URL}")
        print("Please start the server with: python api/main.py")
        sys.exit(1)
    
    # Run examples
    demo = SampleRequests()
    demo.run_all_examples()
