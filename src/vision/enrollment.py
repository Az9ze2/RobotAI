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
        landmarks: Optional[List[List[float]]] = None,
        expected_angle: Optional[str] = None
    ) -> Dict:
        """
        Check quality of detected face.
        
        Args:
            image: Input image
            bbox: Face bounding box
            landmarks: Optional facial landmarks
            expected_angle: Expected face angle (e.g., "straight", "left", "right", "up", "down")
        
        Returns:
            Dictionary with quality metrics and pass/fail status
        """
        x1, y1, x2, y2 = map(int, bbox)
        
        # Validate bbox is within image bounds
        h, w = image.shape[:2]
        if x1 < 0 or y1 < 0 or x2 > w or y2 > h or x2 <= x1 or y2 <= y1:
            return {
                "passed": False,
                "quality_score": 0.0,
                "face_size": (0, 0),
                "size_ok": False,
                "blur_score": 0.0,
                "blur_ok": False,
                "brightness": 0.0,
                "lighting_ok": False,
                "angle_ok": False,
                "feedback": "Invalid face region"
            }
        
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
        
        # Check angle if landmarks provided and expected_angle specified
        angle_ok = True
        detected_angle = None
        if expected_angle and landmarks:
            detected_angle = self.determine_angle(landmarks)
            angle_ok = detected_angle == expected_angle
        
        # Overall quality score
        quality_score = 0.0
        if size_ok:
            quality_score += 0.25
        if blur_ok:
            quality_score += 0.35
        if lighting_ok:
            quality_score += 0.25
        if angle_ok:
            quality_score += 0.15
        
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
            "angle_ok": angle_ok,
            "detected_angle": detected_angle,
            "feedback": self._generate_feedback(size_ok, blur_ok, lighting_ok, angle_ok, expected_angle, detected_angle)
        }
    
    def _generate_feedback(
        self,
        size_ok: bool,
        blur_ok: bool,
        lighting_ok: bool,
        angle_ok: bool = True,
        expected_angle: Optional[str] = None,
        detected_angle: Optional[str] = None
    ) -> str:
        """Generate user-friendly feedback."""
        issues = []
        
        if not size_ok:
            issues.append("Move closer to camera")
        if not blur_ok:
            issues.append("Hold still, image is blurry")
        if not lighting_ok:
            issues.append("Improve lighting")
        if not angle_ok and expected_angle and detected_angle:
            issues.append(f"Wrong angle: {detected_angle} (need {expected_angle})")
        
        if not issues:
            return "Good quality!"
        
        return " | ".join(issues)
    
    def determine_angle(self, landmarks: List[List[float]]) -> str:
        """
        Determine face angle from landmarks.
        
        Args:
            landmarks: 5-point facial landmarks
        
        Returns:
            Angle label ("straight", "left", "right", "up", "down")
        """
        if len(landmarks) != 5:
            return "unknown"
        
        # Extract landmarks
        left_eye = np.array(landmarks[0])
        right_eye = np.array(landmarks[1])
        nose = np.array(landmarks[2])
        left_mouth = np.array(landmarks[3])
        right_mouth = np.array(landmarks[4])
        
        # Calculate eye center and mouth center
        eye_center = (left_eye + right_eye) / 2
        mouth_center = (left_mouth + right_mouth) / 2
        
        # Calculate distances for horizontal angle (left/right/straight)
        left_dist = np.linalg.norm(nose - left_eye)
        right_dist = np.linalg.norm(nose - right_eye)
        horizontal_ratio = left_dist / (right_dist + 1e-6)
        
        # Calculate vertical position for up/down detection
        # When looking up, nose moves up relative to mouth (smaller distance)
        # When looking down, nose moves down relative to eyes (larger distance)
        face_height = np.linalg.norm(eye_center - mouth_center)
        nose_to_eyes = np.linalg.norm(nose - eye_center)
        vertical_ratio = nose_to_eyes / (face_height + 1e-6)
        
        # Determine vertical angle (up/down) first - these are more extreme
        # Looking up: nose closer to eyes (vertical_ratio < 0.35)
        # Looking down: nose farther from eyes (vertical_ratio > 0.55)
        if vertical_ratio < 0.35:
            return "up"
        elif vertical_ratio > 0.55:
            return "down"
        
        # If not up/down, check horizontal angle (left/right/straight)
        # Frontal: ratio ≈ 1.0
        # Left turn: ratio ≈ 0.71 (nose closer to left eye)
        # Right turn: ratio ≈ 1.40 (nose closer to right eye)
        if 0.85 < horizontal_ratio < 1.15:
            return "straight"
        elif horizontal_ratio <= 0.85:
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
        
        logger.info(f"Embedding validation: avg={avg_similarity:.3f}, min={min_similarity:.3f}, max={max_similarity:.3f}")
        
        # Reject if embeddings are too identical (likely same image)
        if min_similarity > 0.98:
            logger.warning(f"Embeddings too identical: min_similarity={min_similarity:.3f}")
            return False
        
        # Reject if embeddings are too different (likely different people)
        # Relaxed threshold for multi-angle captures (down from 0.5 to 0.3)
        if avg_similarity < 0.3:
            logger.warning(f"Embeddings too different: avg_similarity={avg_similarity:.3f}")
            return False
        
        # Accept if embeddings are reasonably similar (0.3-0.98 range)
        logger.info("✓ Embeddings validated successfully")
        return True
    
    def __repr__(self) -> str:
        return (
            f"EnrollmentManager(min_face_size={self.min_face_size}, "
            f"required_angles={self.required_angles})"
        )
