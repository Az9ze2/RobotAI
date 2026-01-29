
from typing import Dict, Optional, List
from loguru import logger
import json
import uuid

try:
    from context_builder import ContextBuilder
    from intent_gate import IntentGate
except ImportError:
    from src.mcp.context_builder import ContextBuilder
    from src.mcp.intent_gate import IntentGate

class MCPAgent:
    def __init__(self):
        """Initialize MCP Agent"""
        self.context_builder = ContextBuilder()
        self.intent_gate = IntentGate()
        # In a real scenario, LLM and Vector DB clients would be initialized here
        logger.info("MCP Agent initialized")

    def process_face_event(self, event_data: Dict) -> Dict:
        """
        Process a face detection event
        
        Args:
            event_data: {
                "student_id": str,
                "confidence": float,
                "timestamp": float
            }
        """
        student_id = event_data.get("student_id")
        # For this prototype, we'll generate a session ID or use a fixed one for testing
        session_id = f"session_{student_id}" if student_id else "unknown_session"
        
        if student_id:
            # Mock looking up student name from ID
            student_name = "Student_" + student_id
            self.context_builder.update_student_identity(session_id, student_id, student_name)
            
        logger.info(f"Processed face event for student: {student_id}")
        return {"status": "success", "session_id": session_id}

    def process_speech_input(self, session_id: str, text: str, confidence: float) -> Dict:
        """
        Process speech input from user
        
        Args:
            session_id: Session identifier
            text: Transcribed speech
            confidence: STT confidence
            
        Returns:
            Dict containing response text and action intent
        """
        # 1. Update Context
        self.context_builder.add_conversation_turn(session_id, "user", text)
        
        # 2. Determine Intent
        context = self.context_builder.get_session(session_id)
        intent_result = self.intent_gate.determine_intent(text, context)
        intent = intent_result["intent"]
        
        # 3. Retrieve Memory (Mocked for now)
        # relevant_memories = self.milvus.search(text)
        relevant_memories = []
        
        # 4. Build LLM Context
        llm_context = self.context_builder.build_llm_context(session_id, relevant_memories)
        
        # 5. Call LLM (Mocked for now)
        llm_response = self._mock_llm_inference(llm_context, intent, text)
        
        # 6. Process Output / Update Context
        self.context_builder.add_conversation_turn(session_id, "assistant", llm_response["speech"])
        
        return {
            "response_text": llm_response["speech"],
            "intent": intent,
            "should_navigate": "nav_goal" in llm_response,
            "navigation_goal": llm_response.get("nav_goal")
        }

    def _mock_llm_inference(self, context: Dict, intent: str, user_text: str) -> Dict:
        """
        Mock LLM inference based on intent
        """
        student_name = context["student_info"]["name"]
        if student_name == "unknown":
            student_name = "นักศึกษา"
            
        if intent == "navigation":
            # Extract basic goal from text using Thai keywords
            goal = "unknown_location"
            goal_name_thai = "จุดหมาย"
            
            # Lab/Laboratory
            if any(word in user_text for word in ["แลป", "ห้องแลป", "ห้องปฏิบัติการ", "lab"]):
                goal = "AI_LAB"
                goal_name_thai = "ห้องแลป"
            # Classroom
            elif any(word in user_text for word in ["ห้องเรียน", "ห้องเรียน", "classroom"]):
                goal = "CLASSROOM"
                goal_name_thai = "ห้องเรียน"
            # Office
            elif any(word in user_text for word in ["ออฟฟิศ", "สำนักงาน", "ห้องทำงาน", "office"]):
                goal = "OFFICE"
                goal_name_thai = "ออฟฟิศ"
                
            return {
                "speech": f"ได้เลยครับคุณ {student_name} เดี๋ยวผมพาไปที่{goal_name_thai}นะครับ",
                "nav_goal": goal
            }
            
        elif intent == "conversation":
            return {
                "speech": f"สวัสดีครับคุณ {student_name} มีอะไรให้ผมช่วยไหมครับ"
            }
            
        else:
            return {
                "speech": "รับทราบครับ"
            }
