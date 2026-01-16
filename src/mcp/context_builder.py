"""
MCP Context Builder
Constructs rich context for LLM from various sources
"""

from typing import Dict, List, Optional
from datetime import datetime
from loguru import logger


class ContextBuilder:
    def __init__(self):
        """Initialize context builder"""
        self.session_contexts = {}  # session_id -> context dict
        
    def create_session(self, session_id: str) -> None:
        """Create a new session context"""
        self.session_contexts[session_id] = {
            "session_id": session_id,
            "student_id": None,
            "student_name": None,
            "conversation_history": [],
            "current_location": None,
            "detected_faces": [],
            "environment_info": {},
            "created_at": datetime.now().isoformat(),
            "last_updated": datetime.now().isoformat()
        }
        logger.info(f"Created session: {session_id}")
    
    def update_student_identity(self, session_id: str, 
                                student_id: str, student_name: str) -> None:
        """Update student identity in session"""
        if session_id not in self.session_contexts:
            self.create_session(session_id)
        
        self.session_contexts[session_id]["student_id"] = student_id
        self.session_contexts[session_id]["student_name"] = student_name
        self.session_contexts[session_id]["last_updated"] = datetime.now().isoformat()
        logger.info(f"Updated identity for session {session_id}: {student_name}")
    
    def add_conversation_turn(self, session_id: str, 
                            role: str, content: str) -> None:
        """
        Add a conversation turn
        
        Args:
            session_id: Session identifier
            role: 'user' or 'assistant'
            content: Message content
        """
        if session_id not in self.session_contexts:
            self.create_session(session_id)
        
        turn = {
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat()
        }
        
        self.session_contexts[session_id]["conversation_history"].append(turn)
        
        # Keep only last 10 turns to avoid context overflow
        max_history = 10
        if len(self.session_contexts[session_id]["conversation_history"]) > max_history:
            self.session_contexts[session_id]["conversation_history"] = \
                self.session_contexts[session_id]["conversation_history"][-max_history:]
        
        self.session_contexts[session_id]["last_updated"] = datetime.now().isoformat()
    
    def update_location(self, session_id: str, location: str) -> None:
        """Update robot's current location"""
        if session_id not in self.session_contexts:
            self.create_session(session_id)
        
        self.session_contexts[session_id]["current_location"] = location
        self.session_contexts[session_id]["last_updated"] = datetime.now().isoformat()
    
    def update_environment(self, session_id: str, env_data: Dict) -> None:
        """Update environment information"""
        if session_id not in self.session_contexts:
            self.create_session(session_id)
        
        self.session_contexts[session_id]["environment_info"].update(env_data)
        self.session_contexts[session_id]["last_updated"] = datetime.now().isoformat()
    
    def build_llm_context(self, session_id: str, 
                         retrieved_memories: Optional[List[Dict]] = None) -> Dict:
        """
        Build comprehensive context for LLM
        
        Args:
            session_id: Session identifier
            retrieved_memories: Optional memories from vector DB
            
        Returns:
            Context dictionary ready for LLM
        """
        if session_id not in self.session_contexts:
            logger.warning(f"Session {session_id} not found, creating new")
            self.create_session(session_id)
        
        ctx = self.session_contexts[session_id]
        
        # Build context structure
        llm_context = {
            "student_info": {
                "id": ctx.get("student_id", "unknown"),
                "name": ctx.get("student_name", "unknown")
            },
            "conversation_history": ctx.get("conversation_history", [])[-5:],  # Last 5 turns
            "current_location": ctx.get("current_location", "unknown"),
            "environment": ctx.get("environment_info", {}),
            "relevant_memories": []
        }
        
        # Add retrieved memories if available
        if retrieved_memories:
            llm_context["relevant_memories"] = [
                {
                    "content": mem["text"],
                    "relevance_score": mem["score"],
                    "type": mem["memory_type"]
                }
                for mem in retrieved_memories[:3]  # Top 3 most relevant
            ]
        
        return llm_context
    
    def format_context_as_prompt(self, llm_context: Dict) -> str:
        """
        Convert context dict to natural language prompt
        
        Args:
            llm_context: Context dictionary
            
        Returns:
            Formatted prompt string
        """
        prompt_parts = []
        
        # Student info
        student_name = llm_context["student_info"]["name"]
        if student_name != "unknown":
            prompt_parts.append(f"คุณกำลังพูดคุยกับ {student_name}")
        
        # Location
        location = llm_context.get("current_location")
        if location and location != "unknown":
            prompt_parts.append(f"ตำแหน่งปัจจุบัน: {location}")
        
        # Relevant memories
        memories = llm_context.get("relevant_memories", [])
        if memories:
            prompt_parts.append("\nความทรงจำที่เกี่ยวข้อง:")
            for i, mem in enumerate(memories, 1):
                prompt_parts.append(f"{i}. {mem['content']}")
        
        # Conversation history
        history = llm_context.get("conversation_history", [])
        if history:
            prompt_parts.append("\nบทสนทนาล่าสุด:")
            for turn in history[-3:]:  # Last 3 turns
                role_thai = "นักศึกษา" if turn["role"] == "user" else "หุ่นยนต์"
                prompt_parts.append(f"{role_thai}: {turn['content']}")
        
        return "\n".join(prompt_parts)
    
    def get_session(self, session_id: str) -> Optional[Dict]:
        """Get session context"""
        return self.session_contexts.get(session_id)
    
    def clear_session(self, session_id: str) -> None:
        """Clear a session"""
        if session_id in self.session_contexts:
            del self.session_contexts[session_id]
            logger.info(f"Cleared session: {session_id}")