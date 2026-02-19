"""
Unit tests for Thai Text Corrector
Tests dictionary-based and LLM-based grammar corrections
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import pytest
from utils.text_corrector import ThaiTextCorrector, correct_thai_text
from llm.typhoon_client import TyphoonClient


class TestDictionaryCorrection:
    """Test dictionary-based corrections"""
    
    def setup_method(self):
        """Setup test corrector without LLM"""
        self.corrector = ThaiTextCorrector(use_llm=False, use_dictionary=True)
    
    def test_lab_correction(self):
        """Test lab/laboratory phonetic corrections"""
        # แหลก -> แลป
        result = self.corrector.correct("พาไปห้องแหลกนายครับ")
        assert "แลป" in result['corrected_text']
        assert "แหลก" not in result['corrected_text']
        assert len(result['corrections']) > 0
    
    def test_polite_particle_correction(self):
        """Test polite particle corrections"""
        # นาย -> หน่อย
        result = self.corrector.correct("พาไปห้องแลปนายครับ")
        assert "หน่อย" in result['corrected_text']
        assert "นาย" not in result['corrected_text']
    
    def test_multiple_corrections(self):
        """Test multiple corrections in one text"""
        result = self.corrector.correct("พาไปห้องแหลกนายครับ")
        # Should correct both แหลก and นาย
        assert "แลป" in result['corrected_text']
        assert "หน่อย" in result['corrected_text']
        assert len(result['corrections']) >= 2
    
    def test_canteen_correction(self):
        """Test canteen location correction"""
        result = self.corrector.correct("โรงกินอยู่ที่ไหน")
        assert "โรงอาหาร" in result['corrected_text']
        assert "โรงกิน" not in result['corrected_text']
    
    def test_library_correction(self):
        """Test library location correction"""
        result = self.corrector.correct("พาไปห้องสมุท")
        assert "ห้องสมุด" in result['corrected_text']
    
    def test_filler_word_removal(self):
        """Test removal of filler words"""
        result = self.corrector.correct("เอ่อ พาไป ห้องสมุด ครับ")
        # Should remove เอ่อ
        assert "เอ่อ" not in result['corrected_text']
        assert "ห้องสมุด" in result['corrected_text']
    
    def test_no_correction_needed(self):
        """Test that correct text is unchanged"""
        original = "สวัสดีครับ"
        result = self.corrector.correct(original)
        assert result['corrected_text'] == original
        assert len(result['corrections']) == 0
    
    def test_high_confidence_skip(self):
        """Test skipping correction for high confidence"""
        result = self.corrector.correct(
            "พาไปห้องแหลกนายครับ",
            confidence=0.96,
            skip_if_confident=True,
            confidence_threshold=0.95
        )
        # Should skip correction
        assert result['method'] == 'skipped'
        assert result['corrected_text'] == "พาไปห้องแหลกนายครับ"
    
    def test_low_confidence_correction(self):
        """Test correction for low confidence"""
        result = self.corrector.correct(
            "พาไปห้องแหลกนายครับ",
            confidence=0.85,
            skip_if_confident=True,
            confidence_threshold=0.95
        )
        # Should apply correction
        assert result['method'] != 'skipped'
        assert "แลป" in result['corrected_text']


class TestLLMCorrection:
    """Test LLM-based corrections"""
    
    def setup_method(self):
        """Setup test corrector with LLM"""
        try:
            self.llm = TyphoonClient()
            self.corrector = ThaiTextCorrector(
                llm_client=self.llm,
                use_llm=True,
                use_dictionary=True,
                llm_temperature=0.3
            )
            self.llm_available = True
        except Exception as e:
            self.llm_available = False
            pytest.skip(f"LLM not available: {e}")
    
    def test_llm_grammar_correction(self):
        """Test LLM-based grammar correction"""
        if not self.llm_available:
            pytest.skip("LLM not available")
        
        # Test with text that has grammar issues
        result = self.corrector.correct(
            "เอ่อ พาไป เอ่อ ห้องสมุด ด้วย ครับ",
            confidence=0.85
        )
        
        # Should remove filler words and normalize
        assert "เอ่อ" not in result['corrected_text']
        assert "ห้องสมุด" in result['corrected_text']
    
    def test_llm_with_dictionary(self):
        """Test LLM working with dictionary corrections"""
        if not self.llm_available:
            pytest.skip("LLM not available")
        
        result = self.corrector.correct(
            "พาไปห้องแหลกนายครับ",
            confidence=0.85
        )
        
        # Dictionary should fix แหลก and นาย first
        # Then LLM might further refine
        assert "แลป" in result['corrected_text']
        assert "แหลก" not in result['corrected_text']


class TestCorrectionMetadata:
    """Test correction metadata and logging"""
    
    def setup_method(self):
        self.corrector = ThaiTextCorrector(use_llm=False)
    
    def test_correction_tracking(self):
        """Test that corrections are tracked"""
        result = self.corrector.correct("พาไปห้องแหลกนายครับ")
        
        # Should have correction metadata
        assert 'corrections' in result
        assert len(result['corrections']) > 0
        
        # Check correction details
        for corr in result['corrections']:
            assert 'type' in corr
            assert corr['type'] == 'dictionary'
            assert 'wrong' in corr
            assert 'correct' in corr
    
    def test_original_text_preserved(self):
        """Test that original text is preserved in result"""
        original = "พาไปห้องแหลกนายครับ"
        result = self.corrector.correct(original)
        
        assert result['original_text'] == original
        assert result['corrected_text'] != original
    
    def test_duration_tracking(self):
        """Test that duration is tracked"""
        result = self.corrector.correct("สวัสดีครับ")
        
        assert 'duration' in result
        assert result['duration'] >= 0


class TestMaxCorrections:
    """Test maximum corrections limit"""
    
    def test_max_corrections_limit(self):
        """Test that max corrections limit is enforced"""
        corrector = ThaiTextCorrector(use_llm=False, max_corrections=2)
        
        # Create text with many errors
        text = "แหลก นาย โรงกิน ห้องสมุท ออฟฟิต"
        result = corrector.correct(text)
        
        # Should limit corrections
        assert 'corrections' in result


class TestConvenienceFunction:
    """Test convenience function"""
    
    def test_correct_thai_text_function(self):
        """Test the convenience function"""
        corrected = correct_thai_text("พาไปห้องแหลกนายครับ")
        
        assert "แลป" in corrected
        assert "แหลก" not in corrected


class TestAddCustomCorrection:
    """Test adding custom corrections"""
    
    def test_add_correction(self):
        """Test adding a new correction to dictionary"""
        corrector = ThaiTextCorrector(use_llm=False)
        
        # Add custom correction
        corrector.add_correction("ทดสอบผิด", "ทดสอบถูก")
        
        # Test it works
        result = corrector.correct("นี่คือทดสอบผิด")
        assert "ทดสอบถูก" in result['corrected_text']


class TestEdgeCases:
    """Test edge cases"""
    
    def setup_method(self):
        self.corrector = ThaiTextCorrector(use_llm=False)
    
    def test_empty_string(self):
        """Test with empty string"""
        result = self.corrector.correct("")
        assert result['corrected_text'] == ""
        assert len(result['corrections']) == 0
    
    def test_whitespace_only(self):
        """Test with whitespace only"""
        result = self.corrector.correct("   ")
        assert result['corrected_text'].strip() == ""
    
    def test_english_text(self):
        """Test with English text (should not change)"""
        original = "Hello world"
        result = self.corrector.correct(original)
        assert result['corrected_text'] == original
    
    def test_mixed_thai_english(self):
        """Test with mixed Thai-English"""
        result = self.corrector.correct("พาไป lab นายครับ")
        # Should correct Thai parts
        assert "หน่อย" in result['corrected_text']


def test_stats():
    """Test getting corrector statistics"""
    corrector = ThaiTextCorrector(use_llm=False)
    stats = corrector.get_stats()
    
    assert 'dictionary_entries' in stats
    assert stats['dictionary_entries'] > 0
    assert 'llm_enabled' in stats
    assert 'dictionary_enabled' in stats


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "--tb=short"])
