"""
Integration tests for complete face recognition pipeline.
"""

import pytest
import numpy as np
import sys
from pathlib import Path
from unittest.mock import Mock, MagicMock

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from vision.pipeline import FaceRecognitionPipeline


class TestFaceRecognitionPipeline:
    """Test suite for complete face recognition pipeline."""
    
    @pytest.fixture
    def mock_components(self):
        """Create mock components for testing."""
        # Mock detector
        detector = Mock()
        detector.detect = Mock(return_value=[
            {
                "bbox": [100, 100, 200, 200],
                "confidence": 0.9,
                "landmarks": [[120, 120], [180, 120], [150, 150], [130, 180], [170, 180]]
            }
        ])
        
        # Mock tracker
        tracker = Mock()
        mock_track = Mock()
        mock_track.track_id = 1
        mock_track.bbox = [100, 100, 200, 200]
        mock_track.confidence = 0.9
        mock_track.landmarks = [[120, 120], [180, 120], [150, 150], [130, 180], [170, 180]]
        mock_track.age = 5
        tracker.update = Mock(return_value=[mock_track])
        tracker.reset = Mock()
        
        # Mock head pose estimator
        head_pose_estimator = Mock()
        head_pose_estimator.estimate_simple = Mock(return_value={
            "yaw": 5.0,
            "pitch": 3.0,
            "roll": 2.0,
            "is_looking": True
        })
        
        # Mock recognition trigger
        recognition_trigger = Mock()
        mock_decision = Mock()
        mock_decision.should_trigger = True
        mock_decision.reason = "All conditions met"
        recognition_trigger.should_trigger = Mock(return_value=mock_decision)
        recognition_trigger.reset_all = Mock()
        
        # Mock face recognizer
        face_recognizer = Mock()
        mock_embedding = np.random.randn(512).astype(np.float32)
        mock_embedding = mock_embedding / np.linalg.norm(mock_embedding)
        face_recognizer.extract_embedding = Mock(return_value=mock_embedding)
        
        # Mock student database
        student_db = Mock()
        student_db.identify_student = Mock(return_value={
            "student_id": "STU001",
            "name": "John Doe",
            "similarity": 0.85
        })
        student_db.get_stats = Mock(return_value={
            "total_students": 5,
            "total_embeddings": 15
        })
        
        return {
            "detector": detector,
            "tracker": tracker,
            "head_pose_estimator": head_pose_estimator,
            "recognition_trigger": recognition_trigger,
            "face_recognizer": face_recognizer,
            "student_db": student_db
        }
    
    def test_pipeline_initialization(self, mock_components):
        """Test pipeline initialization."""
        pipeline = FaceRecognitionPipeline(**mock_components)
        
        assert pipeline.detector is not None
        assert pipeline.tracker is not None
        assert pipeline.head_pose_estimator is not None
        assert pipeline.recognition_trigger is not None
        assert pipeline.face_recognizer is not None
        assert pipeline.student_db is not None
    
    def test_process_frame_with_identification(self, mock_components):
        """Test processing frame with successful identification."""
        pipeline = FaceRecognitionPipeline(**mock_components)
        
        # Create test image
        image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
        
        # Process frame
        results = pipeline.process_frame(image)
        
        # Verify results
        assert len(results) == 1
        result = results[0]
        
        assert result["track_id"] == 1
        assert result["bbox"] == [100, 100, 200, 200]
        assert result["confidence"] == 0.9
        assert result["head_pose"]["is_looking"] == True
        assert result["student_id"] == "STU001"
        assert result["student_name"] == "John Doe"
        assert result["similarity"] == 0.85
    
    def test_process_frame_not_looking(self, mock_components):
        """Test processing frame when student not looking."""
        # Modify head pose to not looking
        mock_components["head_pose_estimator"].estimate_simple = Mock(return_value={
            "yaw": 45.0,
            "pitch": 30.0,
            "roll": 5.0,
            "is_looking": False
        })
        
        # Modify trigger to not trigger
        mock_decision = Mock()
        mock_decision.should_trigger = False
        mock_decision.reason = "Student not looking"
        mock_components["recognition_trigger"].should_trigger = Mock(return_value=mock_decision)
        
        pipeline = FaceRecognitionPipeline(**mock_components)
        
        image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
        results = pipeline.process_frame(image)
        
        # Should have result but no identification
        assert len(results) == 1
        assert results[0]["student_id"] is None
        assert results[0]["head_pose"]["is_looking"] == False
    
    def test_process_frame_no_match(self, mock_components):
        """Test processing frame with no matching student."""
        # Modify database to return no match
        mock_components["student_db"].identify_student = Mock(return_value=None)
        
        pipeline = FaceRecognitionPipeline(**mock_components)
        
        image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
        results = pipeline.process_frame(image)
        
        # Should have result but no identification
        assert len(results) == 1
        assert results[0]["student_id"] is None
    
    def test_process_frame_no_detections(self, mock_components):
        """Test processing frame with no detections."""
        # Modify detector to return no detections
        mock_components["detector"].detect = Mock(return_value=[])
        mock_components["tracker"].update = Mock(return_value=[])
        
        pipeline = FaceRecognitionPipeline(**mock_components)
        
        image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
        results = pipeline.process_frame(image)
        
        # Should have no results
        assert len(results) == 0
    
    def test_pipeline_reset(self, mock_components):
        """Test pipeline reset."""
        pipeline = FaceRecognitionPipeline(**mock_components)
        
        # Reset pipeline
        pipeline.reset()
        
        # Verify reset was called on components
        mock_components["tracker"].reset.assert_called_once()
        mock_components["recognition_trigger"].reset_all.assert_called_once()
    
    def test_multiple_faces(self, mock_components):
        """Test processing frame with multiple faces."""
        # Modify tracker to return multiple tracks
        mock_track1 = Mock()
        mock_track1.track_id = 1
        mock_track1.bbox = [100, 100, 200, 200]
        mock_track1.confidence = 0.9
        mock_track1.landmarks = [[120, 120], [180, 120], [150, 150], [130, 180], [170, 180]]
        mock_track1.age = 5
        
        mock_track2 = Mock()
        mock_track2.track_id = 2
        mock_track2.bbox = [300, 300, 400, 400]
        mock_track2.confidence = 0.85
        mock_track2.landmarks = [[320, 320], [380, 320], [350, 350], [330, 380], [370, 380]]
        mock_track2.age = 4
        
        mock_components["tracker"].update = Mock(return_value=[mock_track1, mock_track2])
        
        pipeline = FaceRecognitionPipeline(**mock_components)
        
        image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
        results = pipeline.process_frame(image)
        
        # Should have results for both faces
        assert len(results) == 2
        assert results[0]["track_id"] == 1
        assert results[1]["track_id"] == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
