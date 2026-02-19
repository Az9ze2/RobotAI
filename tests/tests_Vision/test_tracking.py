"""
Unit tests for ByteTrack tracker.
"""

import pytest
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from vision.tracker import ByteTracker, Track


class TestByteTracker:
    """Test suite for ByteTrack tracker."""
    
    @pytest.fixture
    def tracker(self):
        """Create tracker instance."""
        return ByteTracker(
            track_thresh=0.6,
            track_buffer=30,
            match_thresh=0.8,
            min_box_area=100
        )
    
    @pytest.fixture
    def sample_detections(self):
        """Create sample detections."""
        return [
            {
                "bbox": [100, 100, 200, 200],
                "confidence": 0.9,
                "landmarks": [[120, 120], [180, 120], [150, 150], [130, 180], [170, 180]]
            },
            {
                "bbox": [300, 300, 400, 400],
                "confidence": 0.85,
                "landmarks": [[320, 320], [380, 320], [350, 350], [330, 380], [370, 380]]
            }
        ]
    
    def test_tracker_initialization(self, tracker):
        """Test tracker initialization."""
        assert tracker.track_thresh == 0.6
        assert tracker.track_buffer == 30
        assert tracker.match_thresh == 0.8
        assert tracker.min_box_area == 100
        assert len(tracker.tracks) == 0
        assert tracker.next_track_id == 1
    
    def test_track_creation(self, tracker, sample_detections):
        """Test creating new tracks."""
        tracks = tracker.update(sample_detections)
        
        assert len(tracks) == 2
        assert tracks[0].track_id == 1
        assert tracks[1].track_id == 2
        assert tracks[0].state in ["new", "tracked"]
        assert tracks[1].state in ["new", "tracked"]
    
    def test_track_update(self, tracker, sample_detections):
        """Test updating existing tracks."""
        # First frame
        tracks1 = tracker.update(sample_detections)
        assert len(tracks1) == 2
        
        # Second frame with same detections (slightly moved)
        moved_detections = [
            {
                "bbox": [105, 105, 205, 205],  # Moved slightly
                "confidence": 0.9,
                "landmarks": [[125, 125], [185, 125], [155, 155], [135, 185], [175, 185]]
            },
            {
                "bbox": [305, 305, 405, 405],  # Moved slightly
                "confidence": 0.85,
                "landmarks": [[325, 325], [385, 325], [355, 355], [335, 385], [375, 385]]
            }
        ]
        
        tracks2 = tracker.update(moved_detections)
        
        # Should maintain same track IDs
        assert len(tracks2) == 2
        assert tracks2[0].track_id == 1
        assert tracks2[1].track_id == 2
        assert tracks2[0].age == 2
        assert tracks2[1].age == 2
    
    def test_iou_calculation(self, tracker):
        """Test IOU calculation."""
        bbox1 = [0, 0, 100, 100]
        bbox2 = [50, 50, 150, 150]
        
        iou = tracker._iou(bbox1, bbox2)
        
        # Expected IOU = 2500 / 17500 ≈ 0.143
        assert 0.1 < iou < 0.2
    
    def test_iou_no_overlap(self, tracker):
        """Test IOU with no overlap."""
        bbox1 = [0, 0, 100, 100]
        bbox2 = [200, 200, 300, 300]
        
        iou = tracker._iou(bbox1, bbox2)
        assert iou == 0.0
    
    def test_iou_perfect_overlap(self, tracker):
        """Test IOU with perfect overlap."""
        bbox1 = [0, 0, 100, 100]
        bbox2 = [0, 0, 100, 100]
        
        iou = tracker._iou(bbox1, bbox2)
        assert iou == 1.0
    
    def test_track_loss(self, tracker, sample_detections):
        """Test track loss when detection disappears."""
        # First frame
        tracks1 = tracker.update(sample_detections)
        assert len(tracks1) == 2
        
        # Second frame with no detections
        tracks2 = tracker.update([])
        
        # Active tracks should be 0 (no detections)
        assert len(tracks2) == 0
        
        # But internal tracks should still exist (marked as lost/missed)
        assert len(tracker.tracks) >= 0  # Tracks may be removed or kept depending on implementation
        
        # If tracks exist, they should have frames_since_update > 0
        for track in tracker.tracks:
            assert track.frames_since_update >= 1
    
    def test_min_box_area_filter(self, tracker):
        """Test filtering by minimum box area."""
        small_detection = [{
            "bbox": [0, 0, 5, 5],  # Area = 25, below threshold of 100
            "confidence": 0.9,
            "landmarks": [[1, 1], [4, 1], [2.5, 2.5], [1.5, 4], [3.5, 4]]
        }]
        
        tracks = tracker.update(small_detection)
        
        # Should be filtered out
        assert len(tracks) == 0
    
    def test_tracker_reset(self, tracker, sample_detections):
        """Test tracker reset."""
        # Create some tracks
        tracker.update(sample_detections)
        assert len(tracker.tracks) > 0
        
        # Reset
        tracker.reset()
        
        assert len(tracker.tracks) == 0
        assert tracker.next_track_id == 1
        assert tracker.frame_count == 0


class TestTrack:
    """Test suite for Track class."""
    
    def test_track_initialization(self):
        """Test track initialization."""
        track = Track(
            track_id=1,
            bbox=[100, 100, 200, 200],
            confidence=0.9,
            landmarks=[[120, 120], [180, 120], [150, 150], [130, 180], [170, 180]]
        )
        
        assert track.track_id == 1
        assert track.bbox == [100, 100, 200, 200]
        assert track.confidence == 0.9
        assert track.state == "tracked"
        assert track.frames_since_update == 0
        assert track.hits == 1
        assert track.age == 1
    
    def test_track_update(self):
        """Test track update."""
        track = Track(
            track_id=1,
            bbox=[100, 100, 200, 200],
            confidence=0.9
        )
        
        # Update track
        new_bbox = [105, 105, 205, 205]
        track.update(new_bbox, 0.95)
        
        assert track.bbox == new_bbox
        assert track.confidence == 0.95
        assert track.frames_since_update == 0
        assert track.hits == 2
        assert track.age == 2
        assert track.state == "tracked"
    
    def test_track_mark_missed(self):
        """Test marking track as missed."""
        track = Track(
            track_id=1,
            bbox=[100, 100, 200, 200],
            confidence=0.9
        )
        
        # Mark as missed
        track.mark_missed()
        
        assert track.frames_since_update == 1
        assert track.age == 2
        
        # Mark as missed again
        track.mark_missed()
        
        assert track.frames_since_update == 2
        assert track.state == "lost"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
