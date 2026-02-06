"""
Live face recognition demo script.

Real-time face detection, tracking, and student identification from webcam.
"""

import cv2
import numpy as np
import argparse
import yaml
from pathlib import Path
import sys
import time

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from vision import FaceRecognitionPipeline
from vision import (
    SCRFDDetector,
    ByteTracker,
    HeadPoseEstimator,
    RecognitionTrigger,
    FaceRecognizer,
    StudentDatabase
)
from loguru import logger


def load_config(config_path: str = "./config/local.yaml") -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def draw_results(frame: np.ndarray, results: list) -> np.ndarray:
    """
    Draw detection and recognition results on frame.
    
    Args:
        frame: Input frame
        results: List of results from pipeline
    
    Returns:
        Frame with visualizations
    """
    display = frame.copy()
    
    for result in results:
        bbox = result["bbox"]
        track_id = result["track_id"]
        confidence = result["confidence"]
        head_pose = result.get("head_pose")
        student_id = result.get("student_id")
        student_name = result.get("student_name")
        similarity = result.get("similarity")
        
        # Draw bounding box
        x1, y1, x2, y2 = map(int, bbox)
        
        # Color based on identification status
        if student_id:
            color = (0, 255, 0)  # Green for identified
        elif head_pose and head_pose.get("is_looking"):
            color = (0, 255, 255)  # Yellow for looking but not identified
        else:
            color = (255, 0, 0)  # Blue for not looking
        
        cv2.rectangle(display, (x1, y1), (x2, y2), color, 2)
        
        # Draw landmarks
        if result.get("landmarks"):
            for lm in result["landmarks"]:
                cv2.circle(display, (int(lm[0]), int(lm[1])), 2, (255, 0, 0), -1)
        
        # Prepare text
        text_lines = [f"Track #{track_id}"]
        
        if student_id:
            text_lines.append(f"ID: {student_id}")
            text_lines.append(f"Name: {student_name}")
            text_lines.append(f"Similarity: {similarity:.3f}")
        else:
            text_lines.append("Unknown")
        
        if head_pose:
            text_lines.append(f"Yaw: {head_pose['yaw']:.1f}°")
            text_lines.append(f"Pitch: {head_pose['pitch']:.1f}°")
            if head_pose["is_looking"]:
                text_lines.append("👁 Looking")
        
        # Draw text
        y_offset = y1 - 10
        for text in text_lines:
            cv2.putText(display, text, (x1, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            y_offset -= 20
    
    return display


def main():
    parser = argparse.ArgumentParser(description="Live face recognition demo")
    parser.add_argument("--config", type=str, default="./config/local.yaml", help="Config file")
    parser.add_argument("--camera", type=int, default=0, help="Camera index")
    parser.add_argument("--show-fps", action="store_true", help="Show FPS")
    
    args = parser.parse_args()
    
    # Load config
    logger.info("Loading configuration...")
    config = load_config(args.config)
    vision_config = config["vision"]
    milvus_config = config["milvus"]
    
    # Initialize components
    logger.info("Initializing pipeline components...")
    
    detector = SCRFDDetector(
        model_path=vision_config["detection"]["model_path"],
        confidence_threshold=vision_config["detection"]["confidence_threshold"],
        nms_threshold=vision_config["detection"]["nms_threshold"],
        input_size=tuple(vision_config["detection"]["input_size"]),
        use_tensorrt=vision_config["detection"]["use_tensorrt"],
        device=vision_config["detection"]["device"]
    )
    
    tracker = ByteTracker(
        track_thresh=vision_config["tracking"]["track_thresh"],
        track_buffer=vision_config["tracking"]["track_buffer"],
        match_thresh=vision_config["tracking"]["match_thresh"],
        min_box_area=vision_config["tracking"]["min_box_area"]
    )
    
    head_pose_estimator = HeadPoseEstimator(
        yaw_threshold=vision_config["head_pose"]["yaw_threshold"],
        pitch_threshold=vision_config["head_pose"]["pitch_threshold"],
        roll_threshold=vision_config["head_pose"]["roll_threshold"]
    )
    
    recognition_trigger = RecognitionTrigger(
        min_track_frames=vision_config["recognition"]["min_track_frames"],
        cooldown_seconds=vision_config["recognition"]["cooldown_seconds"],
        require_attention=vision_config["recognition"]["require_attention"]
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
    
    # Create pipeline
    pipeline = FaceRecognitionPipeline(
        detector=detector,
        tracker=tracker,
        head_pose_estimator=head_pose_estimator,
        recognition_trigger=recognition_trigger,
        face_recognizer=recognizer,
        student_db=student_db
    )
    
    logger.info(f"Pipeline initialized: {pipeline}")
    logger.info(f"Student database: {student_db}")
    
    # Open webcam
    logger.info(f"Opening camera {args.camera}...")
    cap = cv2.VideoCapture(args.camera)
    
    if not cap.isOpened():
        logger.error("Failed to open camera")
        return
    
    logger.info("Starting live recognition. Press 'q' to quit, 'r' to reset")
    
    # FPS calculation
    fps_start_time = time.time()
    fps_frame_count = 0
    fps = 0.0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Process frame
        start_time = time.time()
        results = pipeline.process_frame(frame)
        process_time = (time.time() - start_time) * 1000  # ms
        
        # Draw results
        display = draw_results(frame, results)
        
        # Calculate FPS
        fps_frame_count += 1
        if time.time() - fps_start_time >= 1.0:
            fps = fps_frame_count / (time.time() - fps_start_time)
            fps_frame_count = 0
            fps_start_time = time.time()
        
        # Display performance metrics
        if args.show_fps:
            metrics_text = [
                f"FPS: {fps:.1f}",
                f"Process: {process_time:.1f}ms",
                f"Faces: {len(results)}"
            ]
            
            y_offset = 30
            for text in metrics_text:
                cv2.putText(display, text, (10, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                y_offset += 30
        
        cv2.imshow("Face Recognition", display)
        
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            break
        elif key == ord('r'):
            pipeline.reset()
            logger.info("Pipeline reset")
    
    cap.release()
    cv2.destroyAllWindows()
    logger.info("Demo ended")


if __name__ == "__main__":
    main()
