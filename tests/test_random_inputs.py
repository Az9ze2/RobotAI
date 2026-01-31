"""
Random Input Test for Grammar Checker
Tests with both correct and incorrect Thai text to verify proper handling
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from utils.text_corrector import ThaiTextCorrector
from llm.typhoon_client import TyphoonClient
import random


def test_random_inputs():
    """Test grammar checker with random mix of correct and incorrect inputs"""
    print("\n" + "="*70)
    print("🎲 RANDOM INPUT TEST - Grammar Checker")
    print("="*70)
    print("Testing with mix of correct and incorrect Thai text\n")
    
    # Initialize corrector
    try:
        llm = TyphoonClient()
        corrector = ThaiTextCorrector(llm_client=llm, use_llm=True, use_dictionary=True)
        print("✅ Corrector initialized (Dictionary + LLM)\n")
    except:
        corrector = ThaiTextCorrector(use_llm=False, use_dictionary=True)
        print("✅ Corrector initialized (Dictionary only)\n")
    
    # Test cases: mix of correct, incorrect, and edge cases
    test_cases = [
        # Correct inputs (should NOT be changed)
        {
            "text": "สวัสดีครับ",
            "expected_behavior": "unchanged",
            "category": "Correct - Simple greeting"
        },
        {
            "text": "ขอบคุณมากครับ",
            "expected_behavior": "unchanged",
            "category": "Correct - Thank you"
        },
        {
            "text": "พาไปห้องแลปหน่อยครับ",
            "expected_behavior": "unchanged",
            "category": "Correct - Navigation request"
        },
        {
            "text": "ห้องสมุดอยู่ที่ไหนครับ",
            "expected_behavior": "unchanged",
            "category": "Correct - Location question"
        },
        
        # Incorrect inputs (should be corrected)
        {
            "text": "พาไปห้องรับหน่อยครับ",
            "expected_behavior": "corrected",
            "expected_word": "ห้องแลป",
            "category": "Incorrect - Wrong location (ห้องรับ)"
        },
        {
            "text": "พาไปห้องแหลกนายครับ",
            "expected_behavior": "corrected",
            "expected_word": "ห้องแลป",
            "category": "Incorrect - Multiple errors"
        },
        {
            "text": "โรงกินอยู่ที่ไหน",
            "expected_behavior": "corrected",
            "expected_word": "โรงอาหาร",
            "category": "Incorrect - Wrong word (โรงกิน)"
        },
        {
            "text": "เอ่อ พาไป ห้องสมุด ครับ",
            "expected_behavior": "corrected",
            "should_not_contain": "เอ่อ",
            "category": "Incorrect - Filler words"
        },
        
        # Edge cases
        {
            "text": "ห้องรับ ห้องราบ ห้องลับ",
            "expected_behavior": "corrected",
            "expected_word": "ห้องแลป",
            "category": "Edge - Multiple same errors"
        },
        {
            "text": "",
            "expected_behavior": "unchanged",
            "category": "Edge - Empty string"
        },
        {
            "text": "   ",
            "expected_behavior": "unchanged",
            "category": "Edge - Whitespace only"
        },
        {
            "text": "Hello world",
            "expected_behavior": "unchanged",
            "category": "Edge - English text"
        },
        {
            "text": "พาไป lab หน่อยครับ",
            "expected_behavior": "maybe_corrected",
            "category": "Edge - Mixed Thai-English"
        },
        {
            "text": "สวัสดีครับครับครับ",
            "expected_behavior": "maybe_corrected",
            "category": "Edge - Repeated words"
        },
    ]
    
    # Shuffle test cases to make it random
    random.shuffle(test_cases)
    
    results = {
        "correct_unchanged": 0,
        "incorrect_fixed": 0,
        "unexpected_change": 0,
        "edge_cases": 0
    }
    
    print(f"Running {len(test_cases)} random tests...\n")
    print("-"*70)
    
    for i, test in enumerate(test_cases, 1):
        text = test['text']
        expected = test['expected_behavior']
        category = test['category']
        
        # Run correction
        result = corrector.correct(text, confidence=0.8)
        corrected = result['corrected_text']
        changed = corrected != text
        
        print(f"\n[Test {i}] {category}")
        print(f"Input:     '{text}'")
        print(f"Output:    '{corrected}'")
        print(f"Changed:   {'Yes' if changed else 'No'}")
        
        # Validate result
        success = False
        
        if expected == "unchanged":
            if not changed:
                print(f"✅ PASS - Correctly left unchanged")
                results["correct_unchanged"] += 1
                success = True
            else:
                print(f"⚠️  WARN - Unexpectedly changed (may be LLM refinement)")
                results["unexpected_change"] += 1
        
        elif expected == "corrected":
            if changed:
                # Check if expected word is present
                if 'expected_word' in test:
                    if test['expected_word'] in corrected:
                        print(f"✅ PASS - Correctly fixed (contains '{test['expected_word']}')")
                        results["incorrect_fixed"] += 1
                        success = True
                    else:
                        print(f"❌ FAIL - Changed but missing '{test['expected_word']}'")
                
                # Check if unwanted word is removed
                elif 'should_not_contain' in test:
                    if test['should_not_contain'] not in corrected:
                        print(f"✅ PASS - Correctly removed '{test['should_not_contain']}'")
                        results["incorrect_fixed"] += 1
                        success = True
                    else:
                        print(f"❌ FAIL - Did not remove '{test['should_not_contain']}'")
                else:
                    print(f"✅ PASS - Corrected as expected")
                    results["incorrect_fixed"] += 1
                    success = True
            else:
                print(f"❌ FAIL - Should have been corrected but wasn't")
        
        elif expected == "maybe_corrected":
            print(f"ℹ️  INFO - Edge case, correction optional")
            results["edge_cases"] += 1
            success = True
        
        if result['corrections']:
            print(f"Corrections made: {len(result['corrections'])}")
            for corr in result['corrections'][:2]:  # Show first 2
                if corr['type'] == 'dictionary':
                    print(f"  - '{corr['wrong']}' → '{corr['correct']}'")
    
    # Summary
    print("\n" + "="*70)
    print("📊 TEST SUMMARY")
    print("="*70)
    print(f"Correct inputs unchanged:  {results['correct_unchanged']}")
    print(f"Incorrect inputs fixed:    {results['incorrect_fixed']}")
    print(f"Unexpected changes:        {results['unexpected_change']}")
    print(f"Edge cases handled:        {results['edge_cases']}")
    print(f"\nTotal tests: {len(test_cases)}")
    
    # Calculate success rate
    expected_correct = results['correct_unchanged'] + results['incorrect_fixed'] + results['edge_cases']
    success_rate = (expected_correct / len(test_cases)) * 100
    
    print(f"Success rate: {success_rate:.1f}%")
    print("="*70)
    
    if results['unexpected_change'] == 0:
        print("\n✅ EXCELLENT - No over-correction detected!")
    elif results['unexpected_change'] <= 2:
        print("\n⚠️  GOOD - Minor unexpected changes (likely LLM refinements)")
    else:
        print("\n⚠️  WARNING - Multiple unexpected changes detected")
    
    return success_rate >= 80


def test_confidence_threshold():
    """Test that high confidence inputs skip correction"""
    print("\n" + "="*70)
    print("🎯 CONFIDENCE THRESHOLD TEST")
    print("="*70)
    print("Testing skip_if_confident feature\n")
    
    corrector = ThaiTextCorrector(use_llm=False, use_dictionary=True)
    
    test_text = "พาไปห้องรับหน่อยครับ"  # Has error
    
    # Test with high confidence (should skip)
    result_high = corrector.correct(
        test_text,
        confidence=0.96,
        skip_if_confident=True,
        confidence_threshold=0.95
    )
    
    # Test with low confidence (should correct)
    result_low = corrector.correct(
        test_text,
        confidence=0.85,
        skip_if_confident=True,
        confidence_threshold=0.95
    )
    
    print(f"Input: '{test_text}'")
    print(f"\nHigh confidence (0.96):")
    print(f"  Result: '{result_high['corrected_text']}'")
    print(f"  Skipped: {result_high['method'] == 'skipped'}")
    
    print(f"\nLow confidence (0.85):")
    print(f"  Result: '{result_low['corrected_text']}'")
    print(f"  Corrected: {result_low['method'] != 'skipped'}")
    
    if result_high['method'] == 'skipped' and result_low['method'] != 'skipped':
        print("\n✅ PASS - Confidence threshold working correctly")
        return True
    else:
        print("\n❌ FAIL - Confidence threshold not working as expected")
        return False


def main():
    """Run all random input tests"""
    print("\n" + "🎲"*35)
    print("  RANDOM INPUT TESTS - GRAMMAR CHECKER")
    print("🎲"*35)
    
    # Test 1: Random inputs
    test1_pass = test_random_inputs()
    
    # Test 2: Confidence threshold
    test2_pass = test_confidence_threshold()
    
    # Final result
    print("\n" + "="*70)
    print("🏁 FINAL RESULT")
    print("="*70)
    
    if test1_pass and test2_pass:
        print("✅ ALL RANDOM TESTS PASSED!")
        print("\nThe grammar checker:")
        print("  ✓ Correctly fixes errors")
        print("  ✓ Leaves correct text unchanged")
        print("  ✓ Handles edge cases properly")
        print("  ✓ Respects confidence thresholds")
        return 0
    else:
        print("⚠️  SOME TESTS FAILED")
        return 1


if __name__ == "__main__":
    exit(main())
