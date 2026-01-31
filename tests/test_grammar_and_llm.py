"""
Separate unit tests for Grammar Checker and LLM
Tests the fixes for location corrections and male gender usage
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from utils.text_corrector import ThaiTextCorrector
from llm.typhoon_client import TyphoonClient


def test_grammar_checker_location_corrections():
    """Test grammar checker fixes location names correctly"""
    print("\n" + "="*70)
    print("🧪 TEST 1: Grammar Checker - Location Corrections")
    print("="*70)
    
    corrector = ThaiTextCorrector(use_llm=False, use_dictionary=True)
    
    test_cases = [
        ("พาไปห้องรับหน่อยครับ", "ห้องแลป"),
        ("พาไปห้องราบหน่อยครับ", "ห้องแลป"),
        ("พาไปห้องลับหน่อยครับ", "ห้องแลป"),
        ("พาไปห้องแหลกหน่อยครับ", "ห้องแลป"),
        ("โรงกินอยู่ที่ไหน", "โรงอาหาร"),
        ("ไปห้องสมุทหน่อย", "ห้องสมุด"),
    ]
    
    passed = 0
    failed = 0
    
    for original, expected_word in test_cases:
        result = corrector.correct(original, confidence=0.8)
        corrected = result['corrected_text']
        
        if expected_word in corrected:
            print(f"✅ PASS: '{original}'")
            print(f"   → '{corrected}'")
            print(f"   ✓ Contains '{expected_word}'")
            passed += 1
        else:
            print(f"❌ FAIL: '{original}'")
            print(f"   → '{corrected}'")
            print(f"   ✗ Expected '{expected_word}' but not found")
            failed += 1
        print()
    
    print(f"\n📊 Results: {passed} passed, {failed} failed out of {len(test_cases)}")
    return passed, failed


def test_llm_male_gender():
    """Test LLM uses male polite particles (ครับ) only"""
    print("\n" + "="*70)
    print("🧪 TEST 2: LLM - Male Gender Usage")
    print("="*70)
    
    try:
        llm = TyphoonClient()
        print("✅ LLM connected\n")
    except Exception as e:
        print(f"❌ LLM not available: {e}")
        return 0, 0
    
    # Test prompt with male robot persona
    system_prompt = """คุณเป็นหุ่นยนต์ช่วยเหลือเพศชาย ในมหาวิทยาลัย
ใช้ "ครับ" เท่านั้น ห้ามใช้ "ค่ะ" หรือ "คะ"
พูดสั้นๆ กระชับ ไม่เกิน 1-2 ประโยค
ตอบเฉพาะในขอบเขตที่ถูกถาม ไม่ต้องอธิบายเกินความจำเป็น
ห้ามพูดเกินบทบาท"""
    
    test_cases = [
        {
            "user": "พาไปห้องแลปหน่อยครับ",
            "intent": "navigation",
            "description": "Navigation request"
        },
        {
            "user": "สวัสดีครับ",
            "intent": "conversation",
            "description": "Greeting"
        },
        {
            "user": "ขอบคุณครับ",
            "intent": "conversation",
            "description": "Thank you"
        }
    ]
    
    passed = 0
    failed = 0
    
    for test in test_cases:
        user_prompt = f"""ผู้ใช้พูดว่า: "{test['user']}"
Intent: {test['intent']}

ตอบกลับสั้นๆ เหมาะสม ใช้ "ครับ" เท่านั้น"""
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        response = llm.chat(messages, temperature=0.3, max_tokens=100)
        
        print(f"Test: {test['description']}")
        print(f"Input: '{test['user']}'")
        print(f"Response: '{response}'")
        
        # Check for male particles
        has_male = "ครับ" in response
        has_female = "ค่ะ" in response or "คะ" in response
        is_concise = len(response) < 100  # Rough check for conciseness
        
        if has_male and not has_female and is_concise:
            print(f"✅ PASS:")
            print(f"   ✓ Uses 'ครับ' (male)")
            print(f"   ✓ No 'ค่ะ/คะ' (female)")
            print(f"   ✓ Concise ({len(response)} chars)")
            passed += 1
        else:
            print(f"❌ FAIL:")
            if not has_male:
                print(f"   ✗ Missing 'ครับ'")
            if has_female:
                print(f"   ✗ Contains 'ค่ะ/คะ' (female)")
            if not is_concise:
                print(f"   ✗ Too long ({len(response)} chars)")
            failed += 1
        print()
    
    print(f"\n📊 Results: {passed} passed, {failed} failed out of {len(test_cases)}")
    return passed, failed


def test_llm_correction_with_examples():
    """Test LLM-based grammar correction with specific examples"""
    print("\n" + "="*70)
    print("🧪 TEST 3: LLM Grammar Correction")
    print("="*70)
    
    try:
        llm = TyphoonClient()
        corrector = ThaiTextCorrector(llm_client=llm, use_llm=True, use_dictionary=True)
        print("✅ LLM corrector initialized\n")
    except Exception as e:
        print(f"❌ LLM not available: {e}")
        return 0, 0
    
    test_cases = [
        {
            "original": "เอ่อ พาไป ห้องแลป หน่อย ครับ",
            "should_remove": "เอ่อ",
            "description": "Remove filler words"
        },
        {
            "original": "พาไปห้องรับหน่อยครับ",
            "should_contain": "แลป",
            "description": "Fix location name"
        }
    ]
    
    passed = 0
    failed = 0
    
    for test in test_cases:
        result = corrector.correct(test['original'], confidence=0.8)
        corrected = result['corrected_text']
        
        print(f"Test: {test['description']}")
        print(f"Original:  '{test['original']}'")
        print(f"Corrected: '{corrected}'")
        
        success = True
        if 'should_remove' in test:
            if test['should_remove'] not in corrected:
                print(f"✅ PASS: Removed '{test['should_remove']}'")
                passed += 1
            else:
                print(f"❌ FAIL: Did not remove '{test['should_remove']}'")
                failed += 1
                success = False
        
        if 'should_contain' in test and success:
            if test['should_contain'] in corrected:
                print(f"✅ PASS: Contains '{test['should_contain']}'")
                passed += 1
            else:
                print(f"❌ FAIL: Missing '{test['should_contain']}'")
                failed += 1
        
        print()
    
    print(f"\n📊 Results: {passed} passed, {failed} failed")
    return passed, failed


def main():
    """Run all tests"""
    print("\n" + "🤖"*35)
    print("  GRAMMAR CHECKER & LLM UNIT TESTS")
    print("🤖"*35)
    
    total_passed = 0
    total_failed = 0
    
    # Test 1: Grammar Checker Dictionary
    p1, f1 = test_grammar_checker_location_corrections()
    total_passed += p1
    total_failed += f1
    
    # Test 2: LLM Male Gender
    p2, f2 = test_llm_male_gender()
    total_passed += p2
    total_failed += f2
    
    # Test 3: LLM Grammar Correction
    p3, f3 = test_llm_correction_with_examples()
    total_passed += p3
    total_failed += f3
    
    # Final summary
    print("\n" + "="*70)
    print("📊 FINAL SUMMARY")
    print("="*70)
    print(f"Total Passed: {total_passed}")
    print(f"Total Failed: {total_failed}")
    print(f"Success Rate: {total_passed/(total_passed+total_failed)*100:.1f}%")
    print("="*70)
    
    if total_failed == 0:
        print("\n✅ ALL TESTS PASSED!")
    else:
        print(f"\n⚠️  {total_failed} TEST(S) FAILED")
    
    return 0 if total_failed == 0 else 1


if __name__ == "__main__":
    exit(main())
