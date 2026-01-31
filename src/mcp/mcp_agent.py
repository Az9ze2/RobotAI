
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
        self.context_builder.add_conversation_turn(session_id, "assistant", llm_response.get("speech", ""))
        
        return {
            "intent": intent,
            "should_navigate": llm_response.get("should_navigate", False),
            "navigation_goal": llm_response.get("navigation_goal"),
            "goal_name_thai": llm_response.get("goal_name_thai", "")
        }

    def _mock_llm_inference(self, context: Dict, intent: str, user_text: str) -> Dict:
        """
        Intent detection and goal extraction only
        NO response generation - that's LLM's job
        """
        result = {
            "intent": intent,
            "should_navigate": False,
            "navigation_goal": None
        }
        
        if intent == "navigation":
            # Extract navigation goal from text
            goal = "unknown_location"
            goal_name_thai = "จุดหมาย"
            
            # Lab/Laboratory - check multiple variations
            if any(word in user_text for word in ["แลป", "ห้องแลป", "ห้องปฏิบัติการ", "lab", "laboratory"]):
                goal = "AI_LAB"
                goal_name_thai = "ห้องแลป"
            # Classroom
            elif any(word in user_text for word in ["ห้องเรียน", "classroom"]):
                goal = "CLASSROOM"
                goal_name_thai = "ห้องเรียน"
            # Office
            elif any(word in user_text for word in ["ออฟฟิศ", "ออฟฟิต", "สำนักงาน", "office"]):
                goal = "OFFICE"
                goal_name_thai = "ออฟฟิศ"
            # Library
            elif any(word in user_text for word in ["ห้องสมุด", "หอสมุด", "library"]):
                goal = "LIBRARY"
                goal_name_thai = "ห้องสมุด"
            # Canteen
            elif any(word in user_text for word in ["โรงอาหาร", "โรงกิน", "canteen"]):
                goal = "CANTEEN"
                goal_name_thai = "โรงอาหาร"
            
            result["should_navigate"] = True
            result["navigation_goal"] = goal
            result["goal_name_thai"] = goal_name_thai
        
        # NO response generation - LLM will handle that
        return result
