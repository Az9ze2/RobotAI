"""
Vision module for face detection, tracking, and recognition.

This module provides:
- Face detection using SCRFD (InsightFace) - supports both ONNX and PyTorch
- Multi-face tracking using ByteTrack
- Head pose estimation from facial landmarks
- Face recognition using ArcFace
- Student identification with Milvus vector database
"""

from .detector import SCRFDDetector
from .detector_pytorch import SCRFDPyTorchDetector
from .detector_factory import create_scrfd_detector
from .tracker import ByteTracker
from .head_pose import HeadPoseEstimator
from .recognition_trigger import RecognitionTrigger
from .recognizer import FaceRecognizer
from .student_db import StudentDatabase
from .enrollment import EnrollmentManager
from .pipeline import FaceRecognitionPipeline

__all__ = [
    "SCRFDDetector",
    "SCRFDPyTorchDetector",
    "create_scrfd_detector",
    "ByteTracker",
    "HeadPoseEstimator",
    "RecognitionTrigger",
    "FaceRecognizer",
    "StudentDatabase",
    "EnrollmentManager",
    "FaceRecognitionPipeline",
]
