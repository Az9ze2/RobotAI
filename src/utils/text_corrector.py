"""
Thai Text Corrector
Post-processing for STT output to fix grammar and common transcription errors
"""

from typing import Dict, List, Optional, Any
from loguru import logger
import time


class ThaiTextCorrector:
    """
    Corrects Thai text from STT using dictionary and LLM-based approaches
    """
    
    # Common phonetic errors from STT
    COMMON_CORRECTIONS = {
        # Lab/Laboratory variations
        "แหลก": "แลป",
        "แล็บ": "แลป",
        "แร็บ": "แลป",
        
        # Polite particles (order matters - longer first)
        "นาย": "หน่อย",
        "น่อย": "หน่อย",
        
        # Common locations
        "โรงกิน": "โรงอาหาร",
        "ห้องเลียน": "ห้องเรียน",
        "ห้องสมุท": "ห้องสมุด",
        "หอสมุท": "ห้องสมุด",
        
        # Office variations
        "ออฟฟิต": "ออฟฟิศ",
        "ออฟฟิส": "ออฟฟิศ",
        
        # Common filler words to remove
        "เอ่อ": "",
        "อืม": "",
        "อ่า": "",
    }
    
    def __init__(
        self,
        llm_client=None,
        use_llm: bool = True,
        use_dictionary: bool = True,
        llm_temperature: float = 0.3,
        max_corrections: int = 5
    ):
        """
        Initialize Thai Text Corrector
        
        Args:
            llm_client: LLM client for advanced corrections
            use_llm: Enable LLM-based correction
            use_dictionary: Enable dictionary-based correction
            llm_temperature: Temperature for LLM (lower = more conservative)
            max_corrections: Maximum corrections per text
        """
        self.llm_client = llm_client
        self.use_llm = use_llm and llm_client is not None
        self.use_dictionary = use_dictionary
        self.llm_temperature = llm_temperature
        self.max_corrections = max_corrections
        
        logger.info(f"ThaiTextCorrector initialized (LLM: {self.use_llm}, Dict: {self.use_dictionary})")
    
    def correct(
        self,
        text: str,
        confidence: float = 1.0,
        skip_if_confident: bool = True,
        confidence_threshold: float = 0.95
    ) -> Dict[str, Any]:
        """
        Correct Thai text from STT
        
        Args:
            text: Original text from STT
            confidence: STT confidence score
            skip_if_confident: Skip correction if confidence is high
            confidence_threshold: Threshold for skipping
            
        Returns:
            Dictionary with corrected text and metadata
        """
        start_time = time.time()
        corrections_made = []
        
        # Skip if confidence is very high
        if skip_if_confident and confidence >= confidence_threshold:
            logger.info(f"Skipping correction (confidence: {confidence:.2%} >= {confidence_threshold:.2%})")
            return {
                'original_text': text,
                'corrected_text': text,
                'corrections': [],
                'method': 'skipped',
                'duration': time.time() - start_time
            }
        
        corrected_text = text
        
        # Step 1: Dictionary-based correction
        if self.use_dictionary:
            dict_result = self._dictionary_correction(corrected_text)
            if dict_result['changed']:
                corrected_text = dict_result['text']
                corrections_made.extend(dict_result['corrections'])
                logger.info(f"Dictionary corrections: {len(dict_result['corrections'])}")
        
        # Step 2: LLM-based correction (if enabled and dictionary found issues)
        if self.use_llm and (corrections_made or confidence < 0.9):
            llm_result = self._llm_correction(corrected_text, text)
            if llm_result['changed']:
                corrected_text = llm_result['text']
                corrections_made.append({
                    'type': 'llm',
                    'original': text,
                    'corrected': corrected_text
                })
                logger.info("LLM correction applied")
        
        duration = time.time() - start_time
        
        # Limit corrections
        if len(corrections_made) > self.max_corrections:
            logger.warning(f"Too many corrections ({len(corrections_made)}), using original text")
            corrected_text = text
            corrections_made = []
        
        result = {
            'original_text': text,
            'corrected_text': corrected_text,
            'corrections': corrections_made,
            'method': 'dictionary+llm' if self.use_llm else 'dictionary',
            'duration': duration
        }
        
        if corrected_text != text:
            logger.success(f"Corrected: '{text}' -> '{corrected_text}'")
        
        return result
    
    def _dictionary_correction(self, text: str) -> Dict[str, Any]:
        """
        Apply dictionary-based corrections
        
        Args:
            text: Text to correct
            
        Returns:
            Dictionary with corrected text and changes
        """
        corrected = text
        corrections = []
        
        for wrong, correct in self.COMMON_CORRECTIONS.items():
            if wrong in corrected:
                # Replace the word
                new_text = corrected.replace(wrong, correct)
                
                if new_text != corrected:
                    corrections.append({
                        'type': 'dictionary',
                        'wrong': wrong,
                        'correct': correct,
                        'position': corrected.find(wrong)
                    })
                    corrected = new_text
        
        # Clean up extra spaces
        corrected = ' '.join(corrected.split())
        
        return {
            'text': corrected,
            'changed': corrected != text,
            'corrections': corrections
        }
    
    def _llm_correction(self, text: str, original: str) -> Dict[str, Any]:
        """
        Apply LLM-based correction
        
        Args:
            text: Text to correct (after dictionary)
            original: Original text from STT
            
        Returns:
            Dictionary with corrected text and metadata
        """
        try:
            # Build correction prompt
            system_prompt = """คุณเป็นผู้ช่วยแก้ไขไวยากรณ์ภาษาไทย
งาน: แก้ไขข้อความที่ได้จากระบบ Speech-to-Text ให้ถูกต้อง

กฎสำคัญ:
1. แก้ไขเฉพาะคำที่ผิด ห้ามเปลี่ยนความหมายหรือเจตนาของผู้พูด
2. แก้คำที่เสียงคล้ายกัน (เช่น "แหลก" -> "แลป", "นาย" -> "หน่อย")
3. ลบคำซ้ำซ้อนและคำอุดช่อง (เอ่อ, อืม)
4. รักษาน้ำเสียงและความสุภาพเดิม
5. ตอบเฉพาะข้อความที่แก้แล้ว ไม่ต้องอธิบาย ไม่ต้องใส่เครื่องหมายคำพูด"""
            
            user_prompt = f"""ข้อความต้นฉบับ: {text}

ข้อความที่แก้แล้ว:"""
            
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
            
            # Call LLM
            corrected = self.llm_client.chat(
                messages,
                temperature=self.llm_temperature,
                max_tokens=100
            )
            
            if corrected:
                # Clean up response
                corrected = corrected.strip().strip('"').strip("'")
                
                # Validate: should be similar length and structure
                if len(corrected) > len(text) * 2 or len(corrected) < len(text) * 0.5:
                    logger.warning("LLM correction length mismatch, using original")
                    return {'text': text, 'changed': False}
                
                return {
                    'text': corrected,
                    'changed': corrected != text
                }
            else:
                logger.warning("LLM correction failed, using original")
                return {'text': text, 'changed': False}
                
        except Exception as e:
            logger.error(f"LLM correction error: {e}")
            return {'text': text, 'changed': False}
    
    def add_correction(self, wrong: str, correct: str):
        """
        Add a new correction to the dictionary
        
        Args:
            wrong: Incorrect word
            correct: Correct word
        """
        self.COMMON_CORRECTIONS[wrong] = correct
        logger.info(f"Added correction: '{wrong}' -> '{correct}'")
    
    def get_stats(self) -> Dict[str, int]:
        """Get correction statistics"""
        return {
            'dictionary_entries': len(self.COMMON_CORRECTIONS),
            'llm_enabled': self.use_llm,
            'dictionary_enabled': self.use_dictionary
        }


# Convenience function
def correct_thai_text(
    text: str,
    llm_client=None,
    confidence: float = 1.0
) -> str:
    """
    Quick correction of Thai text
    
    Args:
        text: Text to correct
        llm_client: Optional LLM client
        confidence: STT confidence
        
    Returns:
        Corrected text
    """
    corrector = ThaiTextCorrector(llm_client)
    result = corrector.correct(text, confidence)
    return result['corrected_text']


if __name__ == "__main__":
    # Test the corrector
    print("\n=== Thai Text Corrector Test ===\n")
    
    # Test without LLM (dictionary only)
    corrector = ThaiTextCorrector(use_llm=False)
    
    test_cases = [
        "พาไปห้องแหลกนายครับ",
        "เอ่อ พาไป ห้องสมุด ด้วย ครับ",
        "โรงกินอยู่ที่ไหน",
        "สวัสดีครับ"
    ]
    
    for test in test_cases:
        result = corrector.correct(test, confidence=0.9)
        print(f"Original:  '{result['original_text']}'")
        print(f"Corrected: '{result['corrected_text']}'")
        print(f"Changes:   {len(result['corrections'])}")
        print()
