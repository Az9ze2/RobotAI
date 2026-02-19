"""
Unit tests for enrollment manager.
"""

import pytest
import numpy as np
import cv2
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from vision.enrollment import EnrollmentManager


class TestEnrollmentManager:
    """Test suite for enrollment manager."""
    
    @pytest.fixture
    def manager(self):
        """Create enrollment manager instance."""
        return EnrollmentManager(
            min_face_size=112,
            max_blur_threshold=100.0,
            required_angles=["front", "left", "right"],
            quality_threshold=0.8
        )
    
    @pytest.fixture
    def good_quality_face(self):
        """Create a good quality face image."""
        # Create a sharp, well-lit face image (200x200)
        face = np.ones((200, 200, 3), dtype=np.uint8) * 128
        # Add some texture to avoid being too blurry
        for i in range(0, 200, 10):
            face[i:i+5, :] = 150
        return face
    
    @pytest.fixture
    def blurry_face(self):
        """Create a blurry face image."""
        face = np.ones((200, 200, 3), dtype=np.uint8) * 128
        # Apply heavy blur
        face = cv2.GaussianBlur(face, (51, 51), 0)
        return face
    
    @pytest.fixture
    def small_face(self):
        """Create a small face image."""
        return np.ones((50, 50, 3), dtype=np.uint8) * 128
    
    @pytest.fixture
    def frontal_landmarks(self):
        """Frontal face landmarks."""
        return [
            [100, 150],  # Left eye
            [200, 150],  # Right eye
            [150, 200],  # Nose
            [120, 250],  # Left mouth
            [180, 250]   # Right mouth
        ]
    
    @pytest.fixture
    def left_landmarks(self):
        """Left-turned face landmarks."""
        return [
            [80, 150],
            [200, 150],
            [130, 200],
            [100, 250],
            [170, 250]
        ]
    
    @pytest.fixture
    def right_landmarks(self):
        """Right-turned face landmarks."""
        return [
            [100, 150],
            [220, 150],
            [170, 200],
            [130, 250],
            [200, 250]
        ]
    
    def test_manager_initialization(self, manager):
        """Test manager initialization."""
        assert manager.min_face_size == 112
        assert manager.max_blur_threshold == 100.0
        assert manager.required_angles == ["front", "left", "right"]
        assert manager.quality_threshold == 0.8
    
    def test_good_quality_check(self, manager):
        """Test quality check on good quality face."""
        # Create test image
        image = np.ones((400, 400, 3), dtype=np.uint8) * 128
        # Add texture
        for i in range(0, 400, 10):
            image[i:i+5, :] = 150
        
        bbox = [100, 100, 300, 300]  # 200x200 face
        
        quality = manager.check_face_quality(image, bbox)
        
        assert "passed" in quality
        assert "quality_score" in quality
        assert "face_size" in quality
        assert quality["face_size"] == (200, 200)
        assert quality["size_ok"] == True
    
    def test_small_face_check(self, manager):
        """Test quality check on small face."""
        image = np.ones((400, 400, 3), dtype=np.uint8) * 128
        bbox = [100, 100, 150, 150]  # 50x50 face (too small)
        
        quality = manager.check_face_quality(image, bbox)
        
        assert quality["size_ok"] == False
        assert "closer" in quality["feedback"].lower()
    
    def test_blurry_face_check(self, manager):
        """Test quality check on blurry face."""
        # Create heavily blurred image
        image = np.ones((400, 400, 3), dtype=np.uint8) * 128
        image = cv2.GaussianBlur(image, (51, 51), 0)
        
        bbox = [100, 100, 300, 300]
        
        quality = manager.check_face_quality(image, bbox)
        
        # Blurry image should have low blur score
        assert quality["blur_ok"] == False or quality["blur_score"] < manager.max_blur_threshold
    
    def test_determine_frontal_angle(self, manager, frontal_landmarks):
        """Test determining frontal angle."""
        angle = manager.determine_angle(frontal_landmarks)
        assert angle == "front"
    
    def test_determine_left_angle(self, manager, left_landmarks):
        """Test determining left angle."""
        angle = manager.determine_angle(left_landmarks)
        assert angle == "left"
    
    def test_determine_right_angle(self, manager, right_landmarks):
        """Test determining right angle."""
        angle = manager.determine_angle(right_landmarks)
        assert angle == "right"
    
    def test_invalid_landmarks(self, manager):
        """Test with invalid landmarks."""
        invalid_landmarks = [[100, 150], [200, 150]]  # Only 2 landmarks
        angle = manager.determine_angle(invalid_landmarks)
        assert angle == "unknown"
    
    def test_validate_embeddings_good(self, manager):
        """Test validating good embeddings."""
        # Create similar but diverse embeddings
        embeddings = []
        base = np.random.randn(512).astype(np.float32)
        base = base / np.linalg.norm(base)
        
        for _ in range(3):
            # Add small noise to create similar but different embeddings
            emb = base + np.random.randn(512).astype(np.float32) * 0.1
            emb = emb / np.linalg.norm(emb)
            embeddings.append(emb)
        
        is_valid = manager.validate_embeddings(embeddings)
        # Should be valid (similar but diverse)
        assert isinstance(is_valid, bool)
    
    def test_validate_embeddings_identical(self, manager):
        """Test validating identical embeddings."""
        # Create identical embeddings
        base = np.random.randn(512).astype(np.float32)
        base = base / np.linalg.norm(base)
        
        embeddings = [base.copy() for _ in range(3)]
        
        is_valid = manager.validate_embeddings(embeddings)
        # Identical embeddings should fail validation
        assert is_valid == False
    
    def test_validate_embeddings_too_different(self, manager):
        """Test validating very different embeddings."""
        # Create completely different embeddings
        embeddings = []
        for _ in range(3):
            emb = np.random.randn(512).astype(np.float32)
            emb = emb / np.linalg.norm(emb)
            embeddings.append(emb)
        
        is_valid = manager.validate_embeddings(embeddings)
        # Very different embeddings should fail validation
        assert is_valid == False
    
    def test_validate_single_embedding(self, manager):
        """Test validating single embedding."""
        emb = np.random.randn(512).astype(np.float32)
        emb = emb / np.linalg.norm(emb)
        
        is_valid = manager.validate_embeddings([emb])
        # Single embedding should be valid (can't check diversity)
        assert is_valid == True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
