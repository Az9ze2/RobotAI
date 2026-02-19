"""
Unit tests for head pose estimator.
"""

import pytest
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from vision.head_pose import HeadPoseEstimator


class TestHeadPoseEstimator:
    """Test suite for head pose estimator."""
    
    @pytest.fixture
    def estimator(self):
        """Create estimator instance."""
        return HeadPoseEstimator(
            yaw_threshold=30.0,
            pitch_threshold=20.0,
            roll_threshold=45.0
        )
    
    @pytest.fixture
    def frontal_landmarks(self):
        """Landmarks for frontal face."""
        return [
            [100, 150],  # Left eye
            [200, 150],  # Right eye
            [150, 200],  # Nose
            [120, 250],  # Left mouth
            [180, 250]   # Right mouth
        ]
    
    @pytest.fixture
    def left_turned_landmarks(self):
        """Landmarks for face turned left."""
        return [
            [80, 150],   # Left eye (closer to nose)
            [200, 150],  # Right eye
            [130, 200],  # Nose (shifted left)
            [100, 250],  # Left mouth
            [170, 250]   # Right mouth
        ]
    
    @pytest.fixture
    def right_turned_landmarks(self):
        """Landmarks for face turned right."""
        return [
            [100, 150],  # Left eye
            [220, 150],  # Right eye (farther from nose)
            [170, 200],  # Nose (shifted right)
            [130, 250],  # Left mouth
            [200, 250]   # Right mouth
        ]
    
    def test_estimator_initialization(self, estimator):
        """Test estimator initialization."""
        assert estimator.yaw_threshold == 30.0
        assert estimator.pitch_threshold == 20.0
        assert estimator.roll_threshold == 45.0
    
    def test_frontal_face_simple(self, estimator, frontal_landmarks):
        """Test frontal face detection (simple method)."""
        result = estimator.estimate_simple(frontal_landmarks)
        
        assert "yaw" in result
        assert "pitch" in result
        assert "roll" in result
        assert "is_looking" in result
        
        # Frontal face should have small angles
        assert abs(result["yaw"]) < 15.0
        assert abs(result["roll"]) < 15.0
        assert result["is_looking"] == True
    
    def test_left_turned_face_simple(self, estimator, left_turned_landmarks):
        """Test left-turned face detection (simple method)."""
        result = estimator.estimate_simple(left_turned_landmarks)
        
        # Left turn should have negative yaw
        assert result["yaw"] < 0
    
    def test_right_turned_face_simple(self, estimator, right_turned_landmarks):
        """Test right-turned face detection (simple method)."""
        result = estimator.estimate_simple(right_turned_landmarks)
        
        # Right turn should have positive yaw
        assert result["yaw"] > 0
    
    def test_invalid_landmarks(self, estimator):
        """Test with invalid number of landmarks."""
        invalid_landmarks = [[100, 150], [200, 150]]  # Only 2 landmarks
        
        result = estimator.estimate_simple(invalid_landmarks)
        
        assert result["yaw"] == 0.0
        assert result["pitch"] == 0.0
        assert result["roll"] == 0.0
        assert result["is_looking"] == False
    
    def test_looking_threshold(self):
        """Test looking threshold logic."""
        # Strict thresholds
        strict_estimator = HeadPoseEstimator(
            yaw_threshold=10.0,
            pitch_threshold=10.0,
            roll_threshold=10.0
        )
        
        # Relaxed thresholds
        relaxed_estimator = HeadPoseEstimator(
            yaw_threshold=50.0,
            pitch_threshold=50.0,
            roll_threshold=50.0
        )
        
        landmarks = [
            [100, 150],
            [200, 150],
            [150, 200],
            [120, 250],
            [180, 250]
        ]
        
        strict_result = strict_estimator.estimate_simple(landmarks)
        relaxed_result = relaxed_estimator.estimate_simple(landmarks)
        
        # Both should detect frontal face, but relaxed should be more lenient
        assert isinstance(strict_result["is_looking"], bool)
        assert isinstance(relaxed_result["is_looking"], bool)
    
    def test_tilted_head(self, estimator):
        """Test head with roll (tilt)."""
        # Tilted landmarks (head tilted right)
        tilted_landmarks = [
            [120, 130],  # Left eye (higher)
            [180, 170],  # Right eye (lower)
            [150, 200],  # Nose
            [130, 240],  # Left mouth
            [170, 260]   # Right mouth
        ]
        
        result = estimator.estimate_simple(tilted_landmarks)
        
        # Should detect roll
        assert abs(result["roll"]) > 5.0
    
    def test_rotation_matrix_to_euler(self, estimator):
        """Test rotation matrix to Euler angles conversion."""
        # Identity matrix (no rotation)
        R = np.eye(3)
        yaw, pitch, roll = estimator._rotation_matrix_to_euler_angles(R)
        
        assert abs(yaw) < 1.0
        assert abs(pitch) < 1.0
        assert abs(roll) < 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
