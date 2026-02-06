"""
Complete face recognition pipeline.

This module integrates all components into a complete pipeline:
detection → tracking → pose → trigger → recognition → identification
"""

import numpy as np
import cv2
from typing import List, Dict, Optional
from loguru import logger

from .detector import SCRFDDetector
from .tracker import ByteTracker, Track
from .head_pose import HeadPoseEstimator
from .recognition_trigger import RecognitionTrigger
from .recognizer import FaceRecognizer
from .student_db import StudentDatabase


class FaceRecognitionPipeline:
    """
    Complete face recognition pipeline.
    
    Integrates all components for end-to-end face detection,
    tracking, and student identification.
    """
    
    def __init__(
        self,
        detector: SCRFDDetector,
        tracker: ByteTracker,
        head_pose_estimator: HeadPoseEstimator,
        recognition_trigger: RecognitionTrigger,
        face_recognizer: FaceRecognizer,
        student_db: StudentDatabase
    ):
        """
        Initialize pipeline with all components.
        
        Args:
            detector: SCRFD face detector
            tracker: ByteTrack tracker
            head_pose_estimator: Head pose estimator
            recognition_trigger: Recognition trigger
            face_recognizer: ArcFace face recognizer
            student_db: Student database
        """
        self.detector = detector
        self.tracker = tracker
        self.head_pose_estimator = head_pose_estimator
        self.recognition_trigger = recognition_trigger
        self.face_recognizer = face_recognizer
        self.student_db = student_db
        
        logger.info("FaceRecognitionPipeline initialized")
    
    def process_frame(self, image: np.ndarray) -> List[Dict]:
        """
        Process a single frame through the complete pipeline.
        
        Args:
            image: Input image (BGR format)
        
        Returns:
            List of results for each detected face:
                - track_id: Track ID
                - bbox: Bounding box
                - confidence: Detection confidence
                - landmarks: Facial landmarks
                - head_pose: Head pose angles and attention status
                - student_id: Identified student ID (if triggered and found)
                - student_name: Student name (if identified)
                - similarity: Similarity score (if identified)
        """
        results = []
        
        # Step 1: Detect faces
        detections = self.detector.detect(image)
        
        # Step 2: Update tracker
        tracks = self.tracker.update(detections)
        
        # Step 3: Process each track
        for track in tracks:
            result = {
                "track_id": track.track_id,
                "bbox": track.bbox,
                "confidence": track.confidence,
                "landmarks": track.landmarks,
                "head_pose": None,
                "student_id": None,
                "student_name": None,
                "similarity": None
            }
            
            # Step 4: Estimate head pose
            if track.landmarks:
                head_pose = self.head_pose_estimator.estimate_simple(track.landmarks)
                result["head_pose"] = head_pose
                
                # Step 5: Check if should trigger recognition
                trigger_decision = self.recognition_trigger.should_trigger(
                    track_id=track.track_id,
                    confidence=track.confidence,
                    track_age=track.age,
                    is_looking=head_pose["is_looking"]
                )
                
                # Step 6: If triggered, perform recognition and identification
                if trigger_decision.should_trigger:
                    try:
                        # Extract face embedding
                        embedding = self.face_recognizer.extract_embedding(
                            image,
                            track.bbox,
                            track.landmarks
                        )
                        
                        # Identify student
                        student_info = self.student_db.identify_student(embedding)
                        
                        if student_info:
                            result["student_id"] = student_info["student_id"]
                            result["student_name"] = student_info["name"]
                            result["similarity"] = student_info["similarity"]
                            
                            logger.info(
                                f"Identified student: {student_info['student_id']} "
                                f"({student_info['name']}) with similarity {student_info['similarity']:.3f}"
                            )
                        else:
                            logger.debug(f"Track {track.track_id}: No matching student found")
                    
                    except Exception as e:
                        logger.error(f"Error during recognition: {e}")
            
            results.append(result)
        
        return results
    
    def reset(self):
        """Reset pipeline state."""
        self.tracker.reset()
        self.recognition_trigger.reset_all()
        logger.info("Pipeline reset")
    
    def __repr__(self) -> str:
        return (
            f"FaceRecognitionPipeline("
            f"detector={self.detector.__class__.__name__}, "
            f"tracker={self.tracker.__class__.__name__}, "
            f"students={self.student_db.get_stats().get('total_students', 0)})"
        )
