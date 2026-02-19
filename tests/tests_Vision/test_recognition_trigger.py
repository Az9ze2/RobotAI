"""
Unit tests for recognition trigger.
"""

import pytest
import time
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from vision.recognition_trigger import RecognitionTrigger, TriggerDecision


class TestRecognitionTrigger:
    """Test suite for recognition trigger."""
    
    @pytest.fixture
    def trigger(self):
        """Create trigger instance."""
        return RecognitionTrigger(
            min_track_frames=3,
            cooldown_seconds=5.0,
            require_attention=True
        )
    
    def test_trigger_initialization(self, trigger):
        """Test trigger initialization."""
        assert trigger.min_track_frames == 3
        assert trigger.cooldown_seconds == 5.0
        assert trigger.require_attention == True
        assert len(trigger.last_recognition_time) == 0
    
    def test_track_too_new(self, trigger):
        """Test trigger with track that's too new."""
        decision = trigger.should_trigger(
            track_id=1,
            confidence=0.9,
            track_age=2,  # Less than min_track_frames
            is_looking=True
        )
        
        assert decision.should_trigger == False
        assert "too new" in decision.reason.lower()
        assert decision.track_id == 1
    
    def test_student_not_looking(self, trigger):
        """Test trigger when student not looking."""
        decision = trigger.should_trigger(
            track_id=1,
            confidence=0.9,
            track_age=5,
            is_looking=False  # Not looking
        )
        
        assert decision.should_trigger == False
        assert "not looking" in decision.reason.lower()
    
    def test_successful_trigger(self, trigger):
        """Test successful trigger."""
        decision = trigger.should_trigger(
            track_id=1,
            confidence=0.9,
            track_age=5,
            is_looking=True
        )
        
        assert decision.should_trigger == True
        assert "met" in decision.reason.lower()
        assert decision.track_id == 1
        assert decision.confidence == 0.9
    
    def test_cooldown_period(self, trigger):
        """Test cooldown period."""
        # First trigger - should succeed
        decision1 = trigger.should_trigger(
            track_id=1,
            confidence=0.9,
            track_age=5,
            is_looking=True
        )
        assert decision1.should_trigger == True
        
        # Immediate second trigger - should fail (cooldown)
        decision2 = trigger.should_trigger(
            track_id=1,
            confidence=0.9,
            track_age=6,
            is_looking=True
        )
        assert decision2.should_trigger == False
        assert "cooldown" in decision2.reason.lower()
    
    def test_cooldown_expiry(self, trigger):
        """Test cooldown expiry."""
        # Use shorter cooldown for testing
        trigger.cooldown_seconds = 0.5
        
        # First trigger
        decision1 = trigger.should_trigger(
            track_id=1,
            confidence=0.9,
            track_age=5,
            is_looking=True
        )
        assert decision1.should_trigger == True
        
        # Wait for cooldown to expire
        time.sleep(0.6)
        
        # Second trigger - should succeed
        decision2 = trigger.should_trigger(
            track_id=1,
            confidence=0.9,
            track_age=6,
            is_looking=True
        )
        assert decision2.should_trigger == True
    
    def test_multiple_tracks(self, trigger):
        """Test with multiple tracks."""
        # Trigger for track 1
        decision1 = trigger.should_trigger(
            track_id=1,
            confidence=0.9,
            track_age=5,
            is_looking=True
        )
        assert decision1.should_trigger == True
        
        # Trigger for track 2 - should succeed (different track)
        decision2 = trigger.should_trigger(
            track_id=2,
            confidence=0.85,
            track_age=4,
            is_looking=True
        )
        assert decision2.should_trigger == True
        
        # Trigger for track 1 again - should fail (cooldown)
        decision3 = trigger.should_trigger(
            track_id=1,
            confidence=0.9,
            track_age=6,
            is_looking=True
        )
        assert decision3.should_trigger == False
    
    def test_require_attention_disabled(self):
        """Test with attention requirement disabled."""
        trigger = RecognitionTrigger(
            min_track_frames=3,
            cooldown_seconds=5.0,
            require_attention=False  # Disabled
        )
        
        # Should trigger even when not looking
        decision = trigger.should_trigger(
            track_id=1,
            confidence=0.9,
            track_age=5,
            is_looking=False
        )
        
        assert decision.should_trigger == True
    
    def test_reset_track(self, trigger):
        """Test resetting specific track."""
        # Trigger for track 1
        trigger.should_trigger(1, 0.9, 5, True)
        assert 1 in trigger.last_recognition_time
        
        # Reset track 1
        trigger.reset_track(1)
        assert 1 not in trigger.last_recognition_time
    
    def test_reset_all(self, trigger):
        """Test resetting all tracks."""
        # Trigger for multiple tracks
        trigger.should_trigger(1, 0.9, 5, True)
        trigger.should_trigger(2, 0.85, 4, True)
        assert len(trigger.last_recognition_time) == 2
        
        # Reset all
        trigger.reset_all()
        assert len(trigger.last_recognition_time) == 0


class TestTriggerDecision:
    """Test suite for TriggerDecision dataclass."""
    
    def test_trigger_decision_creation(self):
        """Test creating TriggerDecision."""
        decision = TriggerDecision(
            should_trigger=True,
            reason="Test reason",
            track_id=1,
            confidence=0.95
        )
        
        assert decision.should_trigger == True
        assert decision.reason == "Test reason"
        assert decision.track_id == 1
        assert decision.confidence == 0.95


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
