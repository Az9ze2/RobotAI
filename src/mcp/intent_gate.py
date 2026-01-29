
from typing import Dict, Optional
from loguru import logger

class IntentGate:
    def __init__(self):
        """Initialize Intent Gate"""
        # Thai keyword-based intent mapping
        self.intent_keywords = {
            "navigation": [
                # Basic navigation
                "ไป", "พาไป", "พา", "นำทาง", "ไปที่", "ไปหา",
                # Location queries
                "อยู่ที่ไหน", "ที่ไหน", "ตรงไหน", "แถวไหน",
                # Direction requests
                "ทางไป", "เดินไป", "มาที่", "ไปถึง",
                # Polite forms
                "ช่วยพาไป", "พาไปหน่อย", "ไปด้วย", "ไปให้หน่อย"
            ],
            "conversation": [
                # Greetings
                "สวัสดี", "หวัดดี", "ดีครับ", "ดีค่ะ",
                # Identity questions
                "ชื่ออะไร", "ชื่อไร", "เป็นใคร", "คุณคือใคร",
                # Capability questions
                "ทำอะไรได้", "ทำอะไรได้บ้าง", "ช่วยอะไรได้", "ช่วยอะไรได้บ้าง",
                # General conversation
                "เป็นยังไง", "สบายดี", "ว่าไง", "เป็นไง"
            ],
            "memory": [
                # Memory requests
                "จดจำ", "จำไว้", "บันทึก", "เก็บไว้",
                # Recall requests
                "จำได้ไหม", "จำไหม", "เคยบอก", "เคยพูด",
                # Note taking
                "จดหมายเหตุ", "บันทึกไว้", "เก็บข้อมูล"
            ],
            "system": [
                # Stop commands
                "หยุด", "หยุดนะ", "พอ", "พอแล้ว",
                # Shutdown
                "ปิดเครื่อง", "ปิด", "ออกจากระบบ",
                # Restart
                "เริ่มใหม่", "รีสตาร์ท", "เริ่มต้นใหม่"
            ]
        }

    def determine_intent(self, text: str, context: Optional[Dict] = None) -> Dict:
        """
        Determine intent from user text
        
        Args:
            text: User input text
            context: Optional conversation context
            
        Returns:
            Dictionary containing intent and needs_llm flag
        """
        text = text.lower()
        
        # Default to conversation
        intent = "conversation"
        needs_llm = True
        
        # Check against keywords
        for cat, keywords in self.intent_keywords.items():
            if any(k in text for k in keywords):
                intent = cat
                break
        
        # Refine logic
        if intent == "navigation":
            needs_llm = True  # LLM extracts destination
        elif intent == "system":
            needs_llm = False # System commands might not need LLM
            
        logger.info(f"Determined intent: {intent} (needs_llm={needs_llm}) for text: '{text}'")
        
        return {
            "intent": intent,
            "needs_llm": needs_llm,
            "confidence": 0.8  # Placeholder confidence
        }
