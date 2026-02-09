"""
Student enrollment manager.

This module handles the enrollment process for new students,
including quality checks and multi-angle capture.
"""

import numpy as np
import cv2
from typing import List, Dict, Optional, Tuple
from loguru import logger


class EnrollmentManager:
    """
    Manage student enrollment process.
    
    Features:
    - Quality checks (blur, face size, lighting)
    - Multi-angle capture guidance
    - Embedding generation and validation
    - Database integration
    """
    
    def __init__(
        self,
        min_face_size: int = 112,
        max_blur_threshold: float = 100.0,
        required_angles: List[str] = None,
        quality_threshold: float = 0.8
    ):
        """
        Initialize enrollment manager.
        
        Args:
            min_face_size: Minimum face size (pixels)
            max_blur_threshold: Maximum blur value (lower = sharper)
            required_angles: Required capture angles (e.g., ["front", "left", "right"])
            quality_threshold: Minimum quality score (0-1)
        """
        self.min_face_size = min_face_size
        self.max_blur_threshold = max_blur_threshold
        self.required_angles = required_angles or ["front", "left", "right"]
        self.quality_threshold = quality_threshold
        
        logger.info(f"EnrollmentManager initialized with required angles: {self.required_angles}")
    
    def check_face_quality(
        self,
        image: np.ndarray,
        bbox: List[float],
        landmarks: Optional[List[List[float]]] = None
    ) -> Dict:
        """
        Check quality of detected face.
        
        Args:
            image: Input image
            bbox: Face bounding box
            landmarks: Optional facial landmarks
        
        Returns:
            Dictionary with quality metrics and pass/fail status
        """
        x1, y1, x2, y2 = map(int, bbox)
        face = image[y1:y2, x1:x2]
        
        # Check face size
        face_width = x2 - x1
        face_height = y2 - y1
        size_ok = face_width >= self.min_face_size and face_height >= self.min_face_size
        
        # Check blur (using Laplacian variance)
        gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
        blur_ok = blur_score >= self.max_blur_threshold
        
        # Check lighting (using histogram)
        brightness = np.mean(gray)
        lighting_ok = 50 < brightness < 200  # Reasonable brightness range
        
        # Overall quality score
        quality_score = 0.0
        if size_ok:
            quality_score += 0.3
        if blur_ok:
            quality_score += 0.4
        if lighting_ok:
            quality_score += 0.3
        
        passed = quality_score >= self.quality_threshold
        
        return {
            "passed": passed,
            "quality_score": quality_score,
            "face_size": (face_width, face_height),
            "size_ok": size_ok,
            "blur_score": blur_score,
            "blur_ok": blur_ok,
            "brightness": brightness,
            "lighting_ok": lighting_ok,
            "feedback": self._generate_feedback(size_ok, blur_ok, lighting_ok)
        }
    
    def _generate_feedback(
        self,
        size_ok: bool,
        blur_ok: bool,
        lighting_ok: bool
    ) -> str:
        """Generate user-friendly feedback."""
        issues = []
        
        if not size_ok:
            issues.append("Move closer to camera")
        if not blur_ok:
            issues.append("Hold still, image is blurry")
        if not lighting_ok:
            issues.append("Improve lighting")
        
        if not issues:
            return "Good quality!"
        
        return " | ".join(issues)
    
    def determine_angle(self, landmarks: List[List[float]]) -> str:
        """
        Determine face angle from landmarks.
        
        Args:
            landmarks: 5-point facial landmarks
        
        Returns:
            Angle label ("front", "left", "right")
        """
        if len(landmarks) != 5:
            return "unknown"
        
        # Simple angle detection based on eye positions
        left_eye = np.array(landmarks[0])
        right_eye = np.array(landmarks[1])
        nose = np.array(landmarks[2])
        
        # Calculate distances from nose to each eye
        left_dist = np.linalg.norm(nose - left_eye)
        right_dist = np.linalg.norm(nose - right_eye)
        
        # Determine angle based on distance ratio
        # When face turns left, nose moves closer to left eye (ratio < 1)
        # When face turns right, nose moves closer to right eye (ratio > 1)
        # When frontal, distances are approximately equal (ratio ≈ 1)
        ratio = left_dist / (right_dist + 1e-6)
        
        # Adjusted thresholds based on test landmarks:
        # Frontal: ratio ≈ 1.0
        # Left turn: ratio ≈ 0.71 (nose closer to left eye)
        # Right turn: ratio ≈ 1.40 (nose closer to right eye)
        if 0.85 < ratio < 1.15:
            return "front"
        elif ratio <= 0.85:
            return "left"  # Nose closer to left eye = turned left
        else:
            return "right"  # Nose closer to right eye = turned right
    
    def validate_embeddings(self, embeddings: List[np.ndarray]) -> bool:
        """
        Validate that embeddings are diverse enough.
        
        Args:
            embeddings: List of face embeddings
        
        Returns:
            True if embeddings are valid
        """
        if len(embeddings) < 2:
            return True  # Can't check diversity with < 2 embeddings
        
        # Check pairwise similarities
        similarities = []
        for i in range(len(embeddings)):
            for j in range(i + 1, len(embeddings)):
                sim = np.dot(embeddings[i], embeddings[j])
                similarities.append(sim)
        
        # Embeddings should be similar (same person) but not identical
        avg_similarity = np.mean(similarities)
        min_similarity = np.min(similarities)
        max_similarity = np.max(similarities)
        
        # Reject if embeddings are too identical (likely same image)
        if min_similarity > 0.98:
            logger.warning(f"Embeddings too identical: min_similarity={min_similarity:.3f}")
            return False
        
        # Reject if embeddings are too different (likely different people)
        if avg_similarity < 0.5:
            logger.warning(f"Embeddings too different: avg_similarity={avg_similarity:.3f}")
            return False
        
        # Accept if embeddings are reasonably similar (0.5-0.98 range)
        return True
    
    def __repr__(self) -> str:
        return (
            f"EnrollmentManager(min_face_size={self.min_face_size}, "
            f"required_angles={self.required_angles})"
        )
