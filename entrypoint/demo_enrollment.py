"""
Student enrollment demo script.

Interactive script for enrolling students with webcam.
"""

import cv2
import numpy as np
import argparse
import yaml
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from vision import (
    SCRFDDetector,
    FaceRecognizer,
    StudentDatabase,
    EnrollmentManager
)
from loguru import logger


def load_config(config_path: str = "./config/local.yaml") -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="Enroll a new student")
    parser.add_argument("--student_id", type=str, required=True, help="Student ID")
    parser.add_argument("--name", type=str, required=True, help="Student name")
    parser.add_argument("--config", type=str, default="./config/local.yaml", help="Config file")
    
    args = parser.parse_args()
    
    # Load config
    logger.info("Loading configuration...")
    config = load_config(args.config)
    vision_config = config["vision"]
    milvus_config = config["milvus"]
    
    # Initialize components
    logger.info("Initializing components...")
    
    detector = SCRFDDetector(
        model_path=vision_config["detection"]["model_path"],
        confidence_threshold=vision_config["detection"]["confidence_threshold"],
        nms_threshold=vision_config["detection"]["nms_threshold"],
        input_size=tuple(vision_config["detection"]["input_size"]),
        use_tensorrt=vision_config["detection"]["use_tensorrt"],
        device=vision_config["detection"]["device"]
    )
    
    recognizer = FaceRecognizer(
        model_path=vision_config["face_recognition"]["model_path"],
        embedding_dim=vision_config["face_recognition"]["embedding_dim"],
        use_tensorrt=vision_config["face_recognition"]["use_tensorrt"],
        device=vision_config["face_recognition"]["device"]
    )
    
    student_db = StudentDatabase(
        collection_name=vision_config["student_db"]["collection_name"],
        similarity_threshold=vision_config["student_db"]["similarity_threshold"],
        top_k=vision_config["student_db"]["top_k"],
        embeddings_per_student=vision_config["student_db"]["embeddings_per_student"],
        index_type=vision_config["student_db"]["index_type"],
        metric_type=vision_config["student_db"]["metric_type"],
        host=milvus_config["host"],
        port=milvus_config["port"]
    )
    
    enrollment_mgr = EnrollmentManager(
        min_face_size=vision_config["enrollment"]["min_face_size"],
        max_blur_threshold=vision_config["enrollment"]["max_blur_threshold"],
        required_angles=vision_config["enrollment"]["required_angles"],
        quality_threshold=vision_config["enrollment"]["quality_threshold"]
    )
    
    # Open webcam
    logger.info("Opening webcam...")
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        logger.error("Failed to open webcam")
        return
    
    # Collect embeddings
    embeddings = []
    angles_captured = set()
    required_angles = set(vision_config["enrollment"]["required_angles"])
    
    logger.info(f"Enrolling student: {args.student_id} ({args.name})")
    logger.info(f"Required angles: {required_angles}")
    logger.info("Press 'c' to capture, 'q' to quit")
    
    while len(angles_captured) < len(required_angles):
        ret, frame = cap.read()
        if not ret:
            break
        
        # Detect faces
        detections = detector.detect(frame)
        
        # Draw detections and feedback
        display = frame.copy()
        
        if len(detections) > 0:
            det = detections[0]  # Use first detection
            bbox = det["bbox"]
            landmarks = det["landmarks"]
            
            # Check quality
            quality = enrollment_mgr.check_face_quality(frame, bbox, landmarks)
            
            # Determine angle
            angle = enrollment_mgr.determine_angle(landmarks)
            
            # Draw bbox
            x1, y1, x2, y2 = map(int, bbox)
            color = (0, 255, 0) if quality["passed"] else (0, 0, 255)
            cv2.rectangle(display, (x1, y1), (x2, y2), color, 2)
            
            # Draw landmarks
            for lm in landmarks:
                cv2.circle(display, (int(lm[0]), int(lm[1])), 2, (255, 0, 0), -1)
            
            # Display feedback
            feedback_text = [
                f"Angle: {angle} {'✓' if angle in angles_captured else ''}",
                f"Quality: {quality['quality_score']:.2f}",
                quality["feedback"]
            ]
            
            y_offset = y1 - 10
            for text in feedback_text:
                cv2.putText(display, text, (x1, y_offset), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                y_offset -= 20
        
        # Display progress
        progress_text = f"Captured: {angles_captured} | Remaining: {required_angles - angles_captured}"
        cv2.putText(display, progress_text, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        cv2.imshow("Enrollment", display)
        
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('c') and len(detections) > 0:
            det = detections[0]
            quality = enrollment_mgr.check_face_quality(frame, det["bbox"], det["landmarks"])
            angle = enrollment_mgr.determine_angle(det["landmarks"])
            
            if quality["passed"] and angle in required_angles and angle not in angles_captured:
                # Extract embedding
                embedding = recognizer.extract_embedding(frame, det["bbox"], det["landmarks"])
                embeddings.append(embedding)
                angles_captured.add(angle)
                logger.info(f"Captured {angle} angle! ({len(angles_captured)}/{len(required_angles)})")
            else:
                if not quality["passed"]:
                    logger.warning(f"Quality check failed: {quality['feedback']}")
                elif angle in angles_captured:
                    logger.warning(f"Angle {angle} already captured")
                else:
                    logger.warning(f"Angle {angle} not in required angles")
        
        elif key == ord('q'):
            logger.info("Enrollment cancelled")
            cap.release()
            cv2.destroyAllWindows()
            return
    
    cap.release()
    cv2.destroyAllWindows()
    
    # Validate embeddings
    if not enrollment_mgr.validate_embeddings(embeddings):
        logger.warning("Embeddings may not be diverse enough, but proceeding anyway...")
    
    # Enroll student
    logger.info(f"Enrolling student with {len(embeddings)} embeddings...")
    success = student_db.enroll_student(args.student_id, args.name, embeddings)
    
    if success:
        logger.info(f"✓ Successfully enrolled {args.student_id} ({args.name})")
    else:
        logger.error("✗ Enrollment failed")


if __name__ == "__main__":
    main()
