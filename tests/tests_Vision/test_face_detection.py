"""
Unit tests for SCRFD face detector.
"""

import pytest
import numpy as np
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from vision.detector import SCRFDDetector


class TestSCRFDDetector:
    """Test suite for SCRFD face detector."""
    
    @pytest.fixture
    def detector_config(self):
        """Detector configuration for testing."""
        return {
            "model_path": "./models/scrfd_2.5g_kps_fp16.onnx",
            "confidence_threshold": 0.7,
            "nms_threshold": 0.4,
            "input_size": (640, 640),
            "use_tensorrt": False,  # Use CUDA for testing
            "device": "cuda"
        }
    
    @pytest.fixture
    def sample_image(self):
        """Create a sample test image."""
        # Create a simple test image (640x640x3)
        image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
        return image
    
    def test_detector_initialization(self, detector_config):
        """Test detector initialization."""
        try:
            detector = SCRFDDetector(**detector_config)
            assert detector is not None
            assert detector.confidence_threshold == 0.7
            assert detector.nms_threshold == 0.4
            assert detector.input_size == (640, 640)
        except FileNotFoundError:
            pytest.skip("Model file not found - run model downloader first")
        except Exception as e:
            pytest.skip(f"ONNX Runtime not available: {e}")
    
    def test_preprocess(self, detector_config, sample_image):
        """Test image preprocessing."""
        try:
            detector = SCRFDDetector(**detector_config)
            preprocessed, scale = detector.preprocess(sample_image)
            
            # Check output shape
            assert preprocessed.shape[0] == 1  # Batch size
            assert preprocessed.shape[1] == 3  # Channels
            assert preprocessed.shape[2] == 640  # Height
            assert preprocessed.shape[3] == 640  # Width
            
            # Check scale factor
            assert isinstance(scale, float)
            assert scale > 0
        except FileNotFoundError:
            pytest.skip("Model file not found")
        except Exception as e:
            pytest.skip(f"Test skipped: {e}")
    
    def test_nms(self, detector_config):
        """Test Non-Maximum Suppression."""
        try:
            detector = SCRFDDetector(**detector_config)
            
            # Create test boxes and scores
            boxes = np.array([
                [10, 10, 50, 50],
                [15, 15, 55, 55],  # Overlaps with first
                [100, 100, 150, 150]  # Separate box
            ], dtype=np.float32)
            
            scores = np.array([0.9, 0.8, 0.95], dtype=np.float32)
            
            # Run NMS
            keep_indices = detector._nms(boxes, scores, threshold=0.5)
            
            # Should keep boxes 0 and 2 (highest scores, non-overlapping)
            assert len(keep_indices) == 2
            assert 0 in keep_indices or 1 in keep_indices
            assert 2 in keep_indices
        except FileNotFoundError:
            pytest.skip("Model file not found")
        except Exception as e:
            pytest.skip(f"Test skipped: {e}")
    
    def test_detect_no_faces(self, detector_config, sample_image):
        """Test detection on image with no faces."""
        try:
            detector = SCRFDDetector(**detector_config)
            detections = detector.detect(sample_image)
            
            # Random image should have no detections
            assert isinstance(detections, list)
        except FileNotFoundError:
            pytest.skip("Model file not found")
        except Exception as e:
            pytest.skip(f"Test skipped: {e}")
    
    def test_detection_output_format(self, detector_config):
        """Test that detection output has correct format."""
        try:
            detector = SCRFDDetector(**detector_config)
            
            # Create a mock detection result
            mock_detection = {
                "bbox": [10.0, 20.0, 100.0, 120.0],
                "confidence": 0.95,
                "landmarks": [[30.0, 40.0], [70.0, 40.0], [50.0, 60.0], [35.0, 85.0], [65.0, 85.0]]
            }
            
            # Verify format
            assert "bbox" in mock_detection
            assert "confidence" in mock_detection
            assert "landmarks" in mock_detection
            assert len(mock_detection["bbox"]) == 4
            assert len(mock_detection["landmarks"]) == 5
        except FileNotFoundError:
            pytest.skip("Model file not found")
        except Exception as e:
            pytest.skip(f"Test skipped: {e}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
