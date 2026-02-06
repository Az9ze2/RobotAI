"""
SCRFD Face Detector with ONNX Runtime and TensorRT support.

This module implements face detection using the SCRFD model from InsightFace,
optimized for NVIDIA Jetson Orin with FP16 precision and TensorRT execution provider.
"""

import numpy as np
import cv2
from typing import List, Tuple, Optional, Dict
import onnxruntime as ort
from loguru import logger


class SCRFDDetector:
    """
    SCRFD face detector with ONNX Runtime and TensorRT support.
    
    Features:
    - FP16 precision for Tensor Core optimization
    - TensorRT execution provider for maximum performance
    - Returns bounding boxes, landmarks (5 keypoints), and confidence scores
    - Configurable confidence and NMS thresholds
    """
    
    def __init__(
        self,
        model_path: str,
        confidence_threshold: float = 0.7,
        nms_threshold: float = 0.4,
        input_size: Tuple[int, int] = (640, 640),
        use_tensorrt: bool = True,
        device: str = "cuda"
    ):
        """
        Initialize SCRFD detector.
        
        Args:
            model_path: Path to SCRFD ONNX model
            confidence_threshold: Minimum confidence for detections
            nms_threshold: NMS IoU threshold
            input_size: Model input size (width, height)
            use_tensorrt: Use TensorRT execution provider
            device: Device to run on ("cuda" or "cpu")
        """
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold
        self.input_size = input_size
        self.use_tensorrt = use_tensorrt
        self.device = device
        
        # Initialize ONNX Runtime session
        self.session = self._create_session()
        
        # Get input/output names
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [output.name for output in self.session.get_outputs()]
        
        logger.info(f"SCRFD detector initialized with model: {model_path}")
        logger.info(f"Input size: {input_size}, Confidence threshold: {confidence_threshold}")
    
    def _create_session(self) -> ort.InferenceSession:
        """Create ONNX Runtime session with appropriate execution providers."""
        providers = []
        
        if self.device == "cuda":
            if self.use_tensorrt:
                # TensorRT execution provider for maximum performance
                providers.append((
                    "TensorrtExecutionProvider",
                    {
                        "trt_fp16_enable": True,  # Enable FP16 for Tensor Cores
                        "trt_engine_cache_enable": True,
                        "trt_engine_cache_path": "./models/trt_cache",
                    }
                ))
            
            # CUDA execution provider as fallback
            providers.append("CUDAExecutionProvider")
        
        # CPU as final fallback
        providers.append("CPUExecutionProvider")
        
        session_options = ort.SessionOptions()
        session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        
        try:
            session = ort.InferenceSession(
                self.model_path,
                sess_options=session_options,
                providers=providers
            )
            logger.info(f"ONNX Runtime providers: {session.get_providers()}")
            return session
        except Exception as e:
            logger.error(f"Failed to create ONNX Runtime session: {e}")
            raise
    
    def preprocess(self, image: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Preprocess image for SCRFD model.
        
        Args:
            image: Input image (BGR format)
        
        Returns:
            Preprocessed image tensor and scale factor
        """
        # Get original image size
        img_h, img_w = image.shape[:2]
        
        # Calculate scale to fit input size while maintaining aspect ratio
        scale = min(self.input_size[0] / img_w, self.input_size[1] / img_h)
        
        # Resize image
        new_w = int(img_w * scale)
        new_h = int(img_h * scale)
        resized = cv2.resize(image, (new_w, new_h))
        
        # Create padded image
        padded = np.zeros((self.input_size[1], self.input_size[0], 3), dtype=np.uint8)
        padded[:new_h, :new_w] = resized
        
        # Convert to RGB and normalize
        rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
        normalized = rgb.astype(np.float32) / 255.0
        
        # Transpose to CHW format and add batch dimension
        transposed = normalized.transpose(2, 0, 1)
        batched = np.expand_dims(transposed, axis=0)
        
        return batched, scale
    
    def postprocess(
        self,
        outputs: List[np.ndarray],
        scale: float,
        original_shape: Tuple[int, int]
    ) -> List[Dict]:
        """
        Postprocess model outputs to get detections.
        
        Args:
            outputs: Model outputs
            scale: Scale factor from preprocessing
            original_shape: Original image shape (height, width)
        
        Returns:
            List of detections, each containing:
                - bbox: [x1, y1, x2, y2]
                - confidence: float
                - landmarks: [[x, y], ...] (5 keypoints)
        """
        detections = []
        
        # Parse outputs based on SCRFD model structure
        # Typically: [bboxes, scores, landmarks]
        if len(outputs) >= 3:
            bboxes = outputs[0]
            scores = outputs[1]
            landmarks = outputs[2]
            
            # Filter by confidence
            valid_indices = scores > self.confidence_threshold
            
            if np.any(valid_indices):
                valid_bboxes = bboxes[valid_indices]
                valid_scores = scores[valid_indices]
                valid_landmarks = landmarks[valid_indices]
                
                # Apply NMS
                keep_indices = self._nms(valid_bboxes, valid_scores, self.nms_threshold)
                
                # Scale back to original image size
                for idx in keep_indices:
                    bbox = valid_bboxes[idx] / scale
                    kpts = valid_landmarks[idx].reshape(-1, 2) / scale
                    
                    detection = {
                        "bbox": bbox.tolist(),
                        "confidence": float(valid_scores[idx]),
                        "landmarks": kpts.tolist()
                    }
                    detections.append(detection)
        
        return detections
    
    def _nms(self, boxes: np.ndarray, scores: np.ndarray, threshold: float) -> List[int]:
        """
        Non-Maximum Suppression.
        
        Args:
            boxes: Bounding boxes [N, 4] (x1, y1, x2, y2)
            scores: Confidence scores [N]
            threshold: IoU threshold
        
        Returns:
            Indices of boxes to keep
        """
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]
        
        areas = (x2 - x1) * (y2 - y1)
        order = scores.argsort()[::-1]
        
        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            
            w = np.maximum(0.0, xx2 - xx1)
            h = np.maximum(0.0, yy2 - yy1)
            inter = w * h
            
            iou = inter / (areas[i] + areas[order[1:]] - inter)
            
            inds = np.where(iou <= threshold)[0]
            order = order[inds + 1]
        
        return keep
    
    def detect(self, image: np.ndarray) -> List[Dict]:
        """
        Detect faces in image.
        
        Args:
            image: Input image (BGR format)
        
        Returns:
            List of detections with bboxes, confidences, and landmarks
        """
        # Preprocess
        input_tensor, scale = self.preprocess(image)
        
        # Run inference
        outputs = self.session.run(self.output_names, {self.input_name: input_tensor})
        
        # Postprocess
        detections = self.postprocess(outputs, scale, image.shape[:2])
        
        logger.debug(f"Detected {len(detections)} faces")
        return detections
    
    def __repr__(self) -> str:
        return (
            f"SCRFDDetector(model={self.model_path}, "
            f"input_size={self.input_size}, "
            f"confidence_threshold={self.confidence_threshold})"
        )
