"""
Student Enrollment Demo

Interactive demo for enrolling students with multi-angle face capture.
Captures 5 angles: straight, left, right, up, down
"""

import cv2
import sys
import numpy as np
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from vision.detector_factory import create_scrfd_detector
from vision.recognizer import FaceRecognizer
from vision.enrollment import EnrollmentManager
from vision.database import EnrollmentDatabase


class EnrollmentDemo:
    """Interactive enrollment demo with 5-angle capture."""
    
    def __init__(self):
        """Initialize enrollment demo."""
        print("=" * 60)
        print("STUDENT ENROLLMENT SYSTEM")
        print("=" * 60)
        print("\nInitializing components...")
        
        # Initialize detector
        print("  Loading face detector...")
        self.detector = create_scrfd_detector(
            model_path="models/buffalo_l/det_10g.onnx",
            confidence_threshold=0.5,
            nms_threshold=0.4,
            input_size=(640, 640),
            device="cpu"
        )
        
        # Initialize recognizer
        print("  Loading face recognizer...")
        self.recognizer = FaceRecognizer(
            model_path="models/arcface_r100_v1_fp16.onnx",
            device="cpu"
        )
        
        # Initialize enrollment manager
        print("  Loading enrollment manager...")
        self.enrollment_mgr = EnrollmentManager(
            min_face_size=112,
            max_blur_threshold=100.0,
            required_angles=["straight", "left", "right", "up", "down"],
            quality_threshold=0.7
        )
        
        # Initialize database
        print("  Loading enrollment database...")
        self.db = EnrollmentDatabase("data/enrollments.json")
        
        print(f"\n✓ Initialization complete!")
        print(f"  Current enrollments: {len(self.db)}")
        print()
    
    def capture_angle(self, cap, angle_name: str, angle_instruction: str):
        """
        Capture face from a specific angle.
        
        Args:
            cap: OpenCV video capture
            angle_name: Name of angle (e.g., "straight", "left")
            angle_instruction: Instruction to show user
        
        Returns:
            Tuple of (embedding, image) or (None, None) if failed
        """
        print(f"\n📸 Capturing: {angle_name.upper()}")
        print(f"   {angle_instruction}")
        print("   Press SPACE when ready, ESC to cancel")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("   ❌ Failed to read frame")
                return None, None
            
            # Detect faces
            detections = self.detector.detect(frame)
            
            # Draw UI
            display = frame.copy()
            h, w = display.shape[:2]
            
            # Draw instruction panel
            cv2.rectangle(display, (0, 0), (w, 100), (0, 0, 0), -1)
            cv2.putText(display, f"Angle: {angle_name.upper()}", (10, 30),
                       cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 255, 255), 2)
            cv2.putText(display, angle_instruction, (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            cv2.putText(display, "SPACE = Capture | ESC = Cancel", (10, 85),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            
            if len(detections) == 0:
                # No face detected
                cv2.putText(display, "NO FACE DETECTED", (w//2 - 150, h//2),
                           cv2.FONT_HERSHEY_DUPLEX, 1.0, (0, 0, 255), 2)
            elif len(detections) > 1:
                # Multiple faces
                cv2.putText(display, "MULTIPLE FACES - Please be alone", (w//2 - 250, h//2),
                           cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 165, 255), 2)
            else:
                # Single face detected
                det = detections[0]
                bbox = det['bbox']
                landmarks = det.get('landmarks')
                
                # Draw bbox
                x1, y1, x2, y2 = map(int, bbox)
                cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Draw landmarks
                if landmarks:
                    for lx, ly in landmarks:
                        cv2.circle(display, (int(lx), int(ly)), 3, (0, 0, 255), -1)
                
                # Check quality
                quality = self.enrollment_mgr.check_face_quality(frame, bbox, landmarks, expected_angle=angle_name)
                
                # Display quality feedback
                feedback_y = y2 + 30
                if quality['passed']:
                    cv2.putText(display, "✓ GOOD QUALITY - Press SPACE", (x1, feedback_y),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                else:
                    cv2.putText(display, f"✗ {quality['feedback']}", (x1, feedback_y),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                
                # Show quality metrics
                metrics_y = feedback_y + 25
                cv2.putText(display, f"Quality: {quality['quality_score']:.2f}", (x1, metrics_y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            
            cv2.imshow("Enrollment - Capture Face", display)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == 27:  # ESC
                print("   ⚠️  Cancelled")
                return None, None
            elif key == 32:  # SPACE
                if len(detections) == 1:
                    det = detections[0]
                    quality = self.enrollment_mgr.check_face_quality(
                        frame, det['bbox'], det.get('landmarks'), expected_angle=angle_name
                    )
                    
                    if quality['passed']:
                        # Extract embedding
                        try:
                            embedding = self.recognizer.extract_embedding(
                                frame, det['bbox'], det.get('landmarks')
                            )
                            print(f"   ✓ Captured {angle_name} successfully!")
                            return embedding, frame
                        except Exception as e:
                            print(f"   ❌ Failed to extract embedding: {e}")
                    else:
                        print(f"   ⚠️  Quality check failed: {quality['feedback']}")
                else:
                    print("   ⚠️  Please ensure exactly one face is visible")
    
    def enroll_student(self):
        """Run the enrollment process for a new student."""
        print("\n" + "=" * 60)
        print("NEW STUDENT ENROLLMENT")
        print("=" * 60)
        
        # Define angles to capture
        angles = [
            ("straight", "Look straight at the camera"),
            ("left", "Turn your head LEFT (your left)"),
            ("right", "Turn your head RIGHT (your right)"),
            ("up", "Tilt your head UP (look at ceiling)"),
            ("down", "Tilt your head DOWN (look at floor)")
        ]
        
        # Open webcam
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("❌ Failed to open webcam")
            return
        
        print("\nStarting face capture...")
        print("Follow the on-screen instructions for each angle")
        
        embeddings = []
        captured_images = []
        
        # Capture each angle
        for angle_name, instruction in angles:
            embedding, image = self.capture_angle(cap, angle_name, instruction)
            
            if embedding is None:
                print("\n⚠️  Enrollment cancelled")
                cap.release()
                cv2.destroyAllWindows()
                return
            
            embeddings.append(embedding)
            captured_images.append(image)
        
        cap.release()
        cv2.destroyAllWindows()
        
        # Validate embeddings
        print("\n🔍 Validating embeddings...")
        if not self.enrollment_mgr.validate_embeddings(embeddings):
            print("❌ Embedding validation failed - please try again")
            return
        
        print("✓ Embeddings validated successfully!")
        
        # Get student information
        print("\n" + "=" * 60)
        print("STUDENT INFORMATION")
        print("=" * 60)
        
        student_id = input("\nEnter Student ID: ").strip()
        if not student_id:
            print("❌ Student ID cannot be empty")
            return
        
        # Check if student already exists
        if self.db.get_student(student_id):
            overwrite = input(f"⚠️  Student {student_id} already exists. Overwrite? (y/n): ").strip().lower()
            if overwrite != 'y':
                print("❌ Enrollment cancelled")
                return
        
        student_name = input("Enter Student Name: ").strip()
        if not student_name:
            print("❌ Student name cannot be empty")
            return
        
        # Optional metadata
        grade = input("Enter Grade (optional): ").strip()
        class_name = input("Enter Class (optional): ").strip()
        
        metadata = {}
        if grade:
            metadata['grade'] = grade
        if class_name:
            metadata['class'] = class_name
        
        # Save to database
        print("\n💾 Saving enrollment...")
        success = self.db.enroll_student(
            student_id=student_id,
            name=student_name,
            embeddings=embeddings,
            metadata=metadata
        )
        
        if success:
            print("\n" + "=" * 60)
            print("✅ ENROLLMENT SUCCESSFUL!")
            print("=" * 60)
            print(f"Student ID: {student_id}")
            print(f"Name: {student_name}")
            print(f"Embeddings captured: {len(embeddings)}")
            print(f"Total enrolled students: {len(self.db)}")
            print("=" * 60)
        else:
            print("\n❌ Failed to save enrollment")
    
    def list_students(self):
        """List all enrolled students."""
        students = self.db.get_all_students()
        
        if not students:
            print("\n📋 No students enrolled yet")
            return
        
        print("\n" + "=" * 60)
        print("ENROLLED STUDENTS")
        print("=" * 60)
        
        for student_id, info in students.items():
            print(f"\nID: {student_id}")
            print(f"  Name: {info['name']}")
            print(f"  Enrolled: {info['enrolled_date']}")
            print(f"  Embeddings: {len(info['embeddings'])}")
            if info.get('metadata'):
                print(f"  Metadata: {info['metadata']}")
        
        print("=" * 60)
    
    def run(self):
        """Run the enrollment demo."""
        while True:
            print("\n" + "=" * 60)
            print("ENROLLMENT MENU")
            print("=" * 60)
            print("1. Enroll new student")
            print("2. List enrolled students")
            print("3. Exit")
            print("=" * 60)
            
            choice = input("\nEnter choice (1-3): ").strip()
            
            if choice == '1':
                self.enroll_student()
            elif choice == '2':
                self.list_students()
            elif choice == '3':
                print("\n👋 Goodbye!")
                break
            else:
                print("❌ Invalid choice")


def main():
    """Main entry point."""
    try:
        demo = EnrollmentDemo()
        demo.run()
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
