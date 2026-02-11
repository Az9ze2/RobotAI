"""
Script to apply voice chat improvements
Run this to update voice_chat.py with all improvements
"""
import re

# Read the current file
with open('c:/Users/Win 10 Pro/Desktop/RobotAI/entrypoint/voice_chat.py', 'r', encoding='utf-8') as f:
    content = f.read()

# 1. Update main function to include student info
old_main = '''    try:
        bot = SafeVoiceChatBot()
        bot.run()'''

new_main = '''    try:
        # Initialize with student info for personalized responses
        bot = SafeVoiceChatBot(
            student_id="65011356",
            student_name="กฤติน"
        )
        bot.run()'''

content = content.replace(old_main, new_main)

# 2. Update get_llm_response method
# Find the method and replace it
old_method_start = '    def get_llm_response(self, user_text: str) -> str:'
old_method_end = '            return self._get_fallback_response(user_text)\n    \n    def _get_fallback_response'

# Extract everything between start and end
import_re = re.compile(r'(    def get_llm_response.*?)(    def _get_fallback_response)', re.DOTALL)

new_method = '''    def get_llm_response(self, user_text: str) -> str:
        """Get response from LLM with student context"""
        if not self.llm_url:
            return self._get_fallback_response(user_text)
        
        try:
            # Add conversation turn to context
            self.context_builder.add_conversation_turn(
                self.session_id, "user", user_text
            )
            
            # Build context with student info
            llm_context = self.context_builder.build_llm_context(self.session_id)
            context_text = self.context_builder.format_context_as_prompt(llm_context)
            
            prompt = f"""คุณคือ "น้องบอท" หุ่นยนต์บริการในสถาบันเทคโนโลยีพระจอมเกล้าคุณทหารลาดกระบัง
คุณพูดภาษาไทยอย่างเป็นกันเอง ใช้คำลงท้าย "ครับ" เพียงครั้งเดียวต่อประโยค

ตอบสั้นๆ กระชับ ไม่เกิน 2 ประโยค
เรียกชื่อนักศึกษาในทุกการตอบเพื่อสร้างความเป็นกันเอง

กฎการทักทาย (ตามชั้นปี):
- ปี 1: "ยินดีต้อนรับสู่สถาบันเทคโนโลยีพระจอมเกล้าคุณทหารลาดกระบังครับคุณ{{ชื่อ}}"
- ปี 2: "สวัสดีครับคุณ{{ชื่อ}} มีโปรเจคอะไรให้ช่วยไหมครับ"
- ปี 3: "สวัสดีครับคุณ{{ชื่อ}} เตรียมตัวฝึกงานเป็นอย่างไรบ้างครับ"
- ปี 4: "สวัสดีครับคุณ{{ชื่อ}} โปรเจคจบเป็นอย่างไรบ้างครับ"

สำคัญ: ตอบเฉพาะเนื้อหา ห้ามใส่ "น้องบอท:" หรือชื่อหุ่นยนต์นำหน้าคำตอบ

{context_text}

User: {user_text}
Assistant:"""

            response = requests.post(
                f"{self.llm_url}/api/generate",
                json={
                    "model": self.llm_model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.7,
                        "num_predict": 100
                    }
                },
                timeout=30  # Increased timeout
            )
            
            if response.status_code == 200:
                result = response.json()
                llm_response = result.get('response', '').strip()
                
                # Clean up response
                if llm_response.startswith('น้องบอท:'):
                    llm_response = llm_response[len('น้องบอท:'):].strip()
                
                # Remove duplicate ครับครับ
                import re
                llm_response = re.sub(r'(ครับ)\1+', r'\1', llm_response)
                llm_response = re.sub(r'(ค่ะ)\1+', r'\1', llm_response)
                llm_response = re.sub(r'(คะ)\1+', r'\1', llm_response)
                
                if llm_response:
                    # Add assistant response to context
                    self.context_builder.add_conversation_turn(
                        self.session_id, "assistant", llm_response
                    )
                    return llm_response
                else:
                    return self._get_fallback_response(user_text)
            else:
                return self._get_fallback_response(user_text)
                
        except requests.Timeout:
            print("⚠️  LLM timeout, using fallback")
            return self._get_fallback_response(user_text)
        except Exception as e:
            logger.error(f"LLM error: {e}")
            return self._get_fallback_response(user_text)
    
    '''

content = import_re.sub(r'\1' + new_method + r'\2', content)

# Write the updated file
with open('c:/Users/Win 10 Pro/Desktop/RobotAI/entrypoint/voice_chat.py', 'w', encoding='utf-8') as f:
    f.write(content)

print("✅ Voice chat updated successfully!")
print("\nChanges applied:")
print("1. ✅ Main function now initializes with student info (กฤติน, ID: 65011356)")
print("2. ✅ get_llm_response uses MCP context builder")
print("3. ✅ Improved prompt with year-based greetings")
print("4. ✅ Response cleaning (removes duplicates and prefix)")
print("5. ✅ Increased timeout to 30s")
