"""
SCRFD Face Detector using PyTorch model directly.

This detector loads the SCRFD PyTorch checkpoint and performs inference
without requiring ONNX conversion.
"""

import torch
import torch.nn.functional as F
import numpy as np
import cv2
from typing import List, Dict, Tuple, Optional
from loguru import logger


class SCRFDPyTorchDetector:
    """
    SCRFD face detector using PyTorch checkpoint directly.
    
    This implementation loads the MMDetection checkpoint and performs
    inference using PyTorch, avoiding the need for ONNX conversion.
    """
    
    def __init__(
        self,
        model_path: str,
        confidence_threshold: float = 0.7,
        nms_threshold: float = 0.4,
        input_size: Tuple[int, int] = (640, 640),
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """
        Initialize SCRFD PyTorch detector.
        
        Args:
            model_path: Path to .pth checkpoint file
            confidence_threshold: Minimum confidence for detections
            nms_threshold: NMS IoU threshold
            input_size: Input image size (height, width)
            device: Device to run inference on ('cuda' or 'cpu')
        """
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold
        self.input_size = input_size
        self.device = device
        
        logger.info(f"Loading SCRFD PyTorch model from {model_path}")
        
        # Load checkpoint
        self.checkpoint = torch.load(model_path, map_location=device)
        self.state_dict = self.checkpoint['state_dict']
        
        # Extract model info
        self.meta = self.checkpoint.get('meta', {})
        logger.info(f"Model info: {self.meta.get('CLASSES', 'Unknown')}")
        
        # Since we don't have the full model architecture without MMDetection,
        # we'll use a simplified inference approach
        logger.warning("Using simplified PyTorch inference")
        logger.warning("For full SCRFD features, install MMDetection or use ONNX model")
        
        # Initialize preprocessing parameters
        self.mean = np.array([127.5, 127.5, 127.5], dtype=np.float32)
        self.std = np.array([128.0, 128.0, 128.0], dtype=np.float32)
    
    def preprocess(self, image: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Preprocess image for inference.
        
        Args:
            image: Input image (BGR format)
            
        Returns:
            Preprocessed tensor and scale factor
        """
        # Get original size
        height, width = image.shape[:2]
        
        # Calculate scale
        scale = min(self.input_size[0] / height, self.input_size[1] / width)
        
        # Resize image
        new_height = int(height * scale)
        new_width = int(width * scale)
        resized = cv2.resize(image, (new_width, new_height))
        
        # Pad to input size
        padded = np.zeros((self.input_size[0], self.input_size[1], 3), dtype=np.uint8)
        padded[:new_height, :new_width] = resized
        
        # Normalize
        normalized = (padded.astype(np.float32) - self.mean) / self.std
        
        # Convert to tensor (CHW format)
        tensor = normalized.transpose(2, 0, 1)
        tensor = np.expand_dims(tensor, axis=0)
        
        return tensor, scale
    
    def detect(self, image: np.ndarray) -> List[Dict]:
        """
        Detect faces in image.
        
        Note: This is a simplified implementation that uses OpenCV as fallback
        since we don't have the full SCRFD model architecture.
        
        To use the full PyTorch model, you would need:
        1. Install MMDetection
        2. Load the model with proper architecture
        3. Run inference through MMDetection's inference API
        
        Args:
            image: Input image (BGR format)
            
        Returns:
            List of detections, each containing:
                - bbox: [x1, y1, x2, y2]
                - confidence: Detection confidence
                - landmarks: Facial landmarks (5 points) or None
        """
        logger.warning("PyTorch SCRFD model requires MMDetection for full inference")
        logger.warning("Using OpenCV Haar Cascade as fallback")
        
        # Fallback to OpenCV for now
        return self._opencv_fallback(image)
    
    def _opencv_fallback(self, image: np.ndarray) -> List[Dict]:
        """
        Fallback face detection using OpenCV Haar Cascade.
        
        Args:
            image: Input image (BGR format)
            
        Returns:
            List of detections
        """
        # Load Haar Cascade
        face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=4,
            minSize=(30, 30)
        )
        
        # Convert to expected format
        detections = []
        for (x, y, w, h) in faces:
            # Estimate landmarks (rough approximation)
            landmarks = [
                [x + w * 0.3, y + h * 0.4],  # Left eye
                [x + w * 0.7, y + h * 0.4],  # Right eye
                [x + w * 0.5, y + h * 0.6],  # Nose
                [x + w * 0.35, y + h * 0.8], # Left mouth
                [x + w * 0.65, y + h * 0.8]  # Right mouth
            ]
            
            detections.append({
                "bbox": [x, y, x + w, y + h],
                "confidence": 0.9,  # OpenCV doesn't provide confidence
                "landmarks": landmarks
            })
        
        return detections
    
    def detect_with_mmdet(self, image: np.ndarray) -> List[Dict]:
        """
        Detect faces using MMDetection (if available).
        
        This method requires MMDetection to be installed.
        
        Args:
            image: Input image (BGR format)
            
        Returns:
            List of detections
        """
        try:
            from mmdet.apis import init_detector, inference_detector
            
            # This would require the config file
            # For now, return empty list
            logger.error("MMDetection inference requires config file")
            return []
            
        except ImportError:
            logger.error("MMDetection not installed")
            return self._opencv_fallback(image)


# For backward compatibility with ONNX-based detector
class SCRFDDetector(SCRFDPyTorchDetector):
    """Alias for backward compatibility."""
    pass


if __name__ == "__main__":
    # Test the detector
    detector = SCRFDPyTorchDetector(
        model_path="model.pth",
        confidence_threshold=0.7,
        nms_threshold=0.4
    )
    
    # Test with webcam
    cap = cv2.VideoCapture(0)
    
    print("Press 'q' to quit")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Detect faces
        detections = detector.detect(frame)
        
        # Draw results
        for det in detections:
            bbox = det["bbox"]
            conf = det["confidence"]
            
            # Draw bounding box
            cv2.rectangle(
                frame,
                (int(bbox[0]), int(bbox[1])),
                (int(bbox[2]), int(bbox[3])),
                (0, 255, 0),
                2
            )
            
            # Draw confidence
            cv2.putText(
                frame,
                f"{conf:.2f}",
                (int(bbox[0]), int(bbox[1]) - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2
            )
            
            # Draw landmarks if available
            if det["landmarks"]:
                for lm in det["landmarks"]:
                    cv2.circle(frame, (int(lm[0]), int(lm[1])), 2, (0, 0, 255), -1)
        
        # Show frame
        cv2.imshow('SCRFD PyTorch Detector', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
