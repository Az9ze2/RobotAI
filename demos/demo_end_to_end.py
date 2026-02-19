"""
End-to-End Vision + Voice Interaction Pipeline
================================================

Flow:
1. Vision: Detect if person is looking at camera
2. Vision: Recognize who is looking (after sustained gaze)
3. STT: Open microphone for voice input (with silence detection)
4. LLM: Process input with Qwen 2.5 (student-aware)
5. TTS: Respond with MMS Thai voice

Press 'q' to quit
Press 'r' to reset recognition
"""

import cv2
import numpy as np
import sys
from pathlib import Path
import time
from datetime import datetime
import sounddevice as sd
import soundfile as sf
import requests
import yaml
from loguru import logger

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from vision.detector_factory import create_scrfd_detector
from vision.tracker import ByteTracker
from vision.head_pose import HeadPoseEstimator
from vision.recognition_trigger import RecognitionTrigger
from vision.recognizer import FaceRecognizer
from vision.database import EnrollmentDatabase
from stt.typhoon_asr_client import TyphoonASR
from tts.vachana_client import VachanaTTS
from mcp.context_builder import ContextBuilder
from llm.ollama_client import OllamaClient

class EndToEndPipeline:
    """Complete vision + voice interaction pipeline"""
    
    def __init__(self):
        """Initialize all components"""
        print("\n" + "="*70)
        print("🤖 END-TO-END INTERACTION PIPELINE")
        print("="*70)
        
        # Load config
        config_path = Path(__file__).parent.parent / "config" / "settings.yaml"
        with open(config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)
        
        # === VISION COMPONENTS ===
        print("\n📹 Initializing Vision Pipeline...")
        
        self.detector = create_scrfd_detector(
            model_path="models/buffalo_l/det_10g.onnx",
            confidence_threshold=0.5,
            nms_threshold=0.4,
            input_size=(640, 640),
            device="cpu"
        )
        print("  ✅ Face Detector")
        
        self.tracker = ByteTracker(
            track_thresh=0.5,
            track_buffer=30,
            match_thresh=0.4,
            min_box_area=100
        )
        print("  ✅ Face Tracker")
        
        self.head_pose = HeadPoseEstimator(
            yaw_threshold=25,
            pitch_threshold=15,
            roll_threshold=30
        )
        print("  ✅ Head Pose Estimator")
        
        self.trigger = RecognitionTrigger(
            cooldown_seconds=5.0,
            require_attention=True
        )
        print("  ✅ Recognition Trigger")
        
        self.recognizer = FaceRecognizer(
            model_path="models/arcface_r100_v1_fp16.onnx",
            device="cpu"
        )
        print("  ✅ Face Recognizer")
        
        self.db = EnrollmentDatabase("data/enrollments.json")
        print(f"  ✅ Enrollment DB ({len(self.db)} students)")
        
        # === VOICE COMPONENTS ===
        print("\n🎤 Initializing Voice Pipeline...")
        
        self.stt = TyphoonASR(
            model_name="scb10x/typhoon-asr-realtime",
            language="th",
            device="auto"
        )
        print("  ✅ STT (Typhoon ASR)")
        
        self.tts = VachanaTTS()
        print("  ✅ TTS (MMS)")
        
        self.llm = OllamaClient(
            api_url=self.config['llm']['api_url'],
            model=self.config['llm']['model']
        )
        print("  ✅ LLM (Qwen 2.5)")
        
        self.context_builder = ContextBuilder()
        self.session_id = "end_to_end_session"
        self.context_builder.create_session(self.session_id)
        print("  ✅ Context Builder")
        
        # === STATE ===
        self.current_student = None  # {"id": ..., "name": ..., "year": ...}
        self.interaction_active = False
        self.last_interaction_time = None
        
        # Visual feedback
        self.status_message = "Waiting for someone to look..."
        self.status_color = (200, 200, 200)  # Gray
        
        print("\n" + "="*70)
        print("✅ Pipeline Ready!")
        print("="*70)
    
    def record_with_silence_detection(self, 
                                      silence_threshold: float = 0.02,
                                      silence_duration: float = 3.0,
                                      max_duration: float = 30.0,
                                      sample_rate: int = 16000) -> tuple:
        """
        Record audio with automatic silence detection cutoff
        
        Args:
            silence_threshold: RMS threshold for silence detection
            silence_duration: Seconds of silence before stopping
            max_duration: Maximum recording duration
            sample_rate: Audio sample rate
            
        Returns:
            (audio_data, actual_duration)
        """
        print(f"\n🎤 Recording (will stop after {silence_duration}s of silence)...")
        
        audio_chunks = []
        silence_start = None
        start_time = time.time()
        
        def audio_callback(indata, frames, time_info, status):
            nonlocal silence_start
            
            if status:
                print(f"⚠️  Audio status: {status}")
            
            # Calculate RMS (volume level)
            rms = np.sqrt(np.mean(indata**2))
            
            # Check for silence
            if rms < silence_threshold:
                if silence_start is None:
                    silence_start = time.time()
            else:
                silence_start = None  # Reset on sound
            
            audio_chunks.append(indata.copy())
        
        # Start recording
        with sd.InputStream(callback=audio_callback, 
                           channels=1, 
                           samplerate=sample_rate,
                           dtype='float32'):
            while True:
                elapsed = time.time() - start_time
                
                # Check max duration
                if elapsed >= max_duration:
                    print(f"⏱️  Max duration reached ({max_duration}s)")
                    break
                
                # Check silence duration
                if silence_start and (time.time() - silence_start >= silence_duration):
                    print(f"🔇 Silence detected for {silence_duration}s - stopping")
                    break
                
                time.sleep(0.1)
        
        # Combine chunks
        if audio_chunks:
            audio_data = np.concatenate(audio_chunks, axis=0)
            actual_duration = len(audio_data) / sample_rate
            print(f"✅ Recorded {actual_duration:.1f}s of audio")
            return audio_data, actual_duration
        else:
            return None, 0
    
    def process_voice_interaction(self):
        """Handle complete voice interaction cycle"""
        if not self.current_student:
            print("⚠️  No student recognized, skipping interaction")
            return
        
        student_name = self.current_student['name']
        student_id = self.current_student['id']
        student_year = self.current_student.get('year', 1)
        
        print("\n" + "="*70)
        print(f"💬 Starting interaction with {student_name}")
        print("="*70)
        
        # Update context with student info
        self.context_builder.update_student_identity(
            self.session_id, student_id, student_name
        )
        
        # === 1. RECORD AUDIO ===
        self.status_message = f"🎤 Listening to {student_name}..."
        self.status_color = (0, 255, 0)  # Green
        
        audio_data, duration = self.record_with_silence_detection(
            silence_threshold=0.02,
            silence_duration=3.0,
            max_duration=30.0
        )
        
        if audio_data is None or duration < 0.5:
            print("❌ No audio recorded")
            self.status_message = "No audio detected"
            self.status_color = (0, 0, 255)  # Red
            return
        
        # Save temporary audio file
        temp_audio = "temp_voice_input.wav"
        sf.write(temp_audio, audio_data, 16000)
        
        # === 2. TRANSCRIBE (STT) ===
        self.status_message = "🔄 Transcribing..."
        print("\n🔄 Transcribing speech...")
        
        try:
            result = self.stt.transcribe_audio(temp_audio)
            user_text = result['text']
            confidence = result['confidence']
            
            print(f"📝 You said: {user_text}")
            print(f"   Confidence: {confidence:.1%}")
            
            if not user_text.strip():
                print("❌ Empty transcription")
                self.status_message = "Could not understand"
                self.status_color = (0, 0, 255)
                return
                
        except Exception as e:
            print(f"❌ STT Error: {e}")
            self.status_message = "STT Error"
            self.status_color = (0, 0, 255)
            return
        
        # === 3. PROCESS WITH LLM ===
        self.status_message = "🤖 Thinking..."
        print("\n🤖 Processing with LLM...")
        
        try:
            # Build context-aware prompt
            llm_context = self.context_builder.build_llm_context(self.session_id)
            context_text = self.context_builder.format_context_as_prompt(llm_context)
            
            # Year-based greeting
            year_greeting = {
                1: "น้องปี 1",
                2: "น้องปี 2", 
                3: "น้องปี 3",
                4: "พี่ปี 4"
            }.get(student_year, "คุณ")
            
            system_prompt = f"""คุณคือหุ่นยนต์ผู้ช่วยของสถาบันเทคโนโลยีพระจอมเกล้าเจ้าคุณทหารลาดกระบัง (KMITL)

กฎสำคัญ:
1. ใช้ชื่อนักศึกษา "{student_name}" ในทุกประโยคตอบ
2. เรียกนักศึกษาว่า "{year_greeting}" (ปี {student_year})
3. ห้ามใช้คำนำหน้าเช่น "น้องบอท:" หรือ "ผม:"
4. ตอบสั้นและกระชับ (1-2 ประโยค)
5. ใช้ "ครับ" ท้ายประโยค"""

            user_message = f"{context_text}\n\nนักศึกษา: {user_text}\n\nน้องบอท:"
            
            response = self.llm.chat(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message}
                ],
                temperature=0.7,
                max_tokens=512
            )
            
            if not response:
                response = f"ขอโทษครับ {student_name} ผมไม่เข้าใจคำถามครับ"
            
            print(f"🤖 Response: {response}")
            
        except Exception as e:
            print(f"❌ LLM Error: {e}")
            response = f"ขอโทษครับ {student_name} ระบบมีปัญหาครับ"
        
        # === 4. SYNTHESIZE SPEECH (TTS) ===
        self.status_message = "🔊 Speaking..."
        print("\n🔊 Synthesizing speech...")
        
        try:
            audio_file, metadata = self.tts.synthesize(response)
            print(f"✅ Audio generated: {metadata['duration']:.2f}s")
            
            # Play audio
            data, samplerate = sf.read(audio_file)
            sd.play(data, samplerate)
            sd.wait()
            
            # Cleanup
            Path(audio_file).unlink(missing_ok=True)
            
        except Exception as e:
            print(f"❌ TTS Error: {e}")
        
        # Cleanup temp audio
        Path(temp_audio).unlink(missing_ok=True)
        
        self.status_message = f"✅ Interaction complete with {student_name}"
        self.status_color = (0, 255, 0)
        self.last_interaction_time = time.time()
        
        print("\n" + "="*70)
        print("✅ Interaction Complete!")
        print("="*70)
    
    def run(self):
        """Main loop with camera feed"""
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("❌ Cannot open camera")
            return
        
        print("\n🎥 Camera opened - starting pipeline...")
        print("Press 'q' to quit, 'r' to reset recognition\n")
        
        frame_count = 0
        fps_start = time.time()
        fps = 0
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Calculate FPS
                frame_count += 1
                if frame_count % 30 == 0:
                    fps = 30 / (time.time() - fps_start)
                    fps_start = time.time()
                
                # === VISION PIPELINE ===
                # 1. Detect faces
                detections = self.detector.detect(frame)
                
                # 2. Track faces
                tracks = self.tracker.update(detections)
                
                # 3. Process each track
                for track in tracks:
                    x1, y1, x2, y2 = map(int, track.bbox)
                    track_id = track.track_id
                    
                    # Draw bounding box
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    
                    # 4. Estimate head pose
                    face_roi = frame[y1:y2, x1:x2]
                    if face_roi.size > 0 and track.landmarks is not None:
                        # Use estimate_simple with landmarks
                        result = self.head_pose.estimate_simple(track.landmarks)
                        is_looking = result['is_looking']
                        angles = {
                            'yaw': result['yaw'],
                            'pitch': result['pitch'],
                            'roll': result['roll']
                        }
                        
                        # Display angles
                        angle_text = f"Y:{angles['yaw']:.0f} P:{angles['pitch']:.0f}"
                        cv2.putText(frame, angle_text, (x1, y1-10),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                        
                        # 5. Check recognition trigger
                        if is_looking:
                            decision = self.trigger.should_trigger(
                                track_id, track.confidence, track.age, is_looking
                            )
                            
                            if decision.should_trigger:
                                # 6. Recognize face
                                print(f"\n🔍 Recognizing Track #{track_id}...")
                                embedding = self.recognizer.extract_embedding(
                                    frame, 
                                    track.bbox,
                                    track.landmarks
                                )
                                
                                if embedding is not None:
                                    student_id, similarity, student_name = self.db.recognize(
                                        embedding, threshold=0.4
                                    )
                                    
                                    if student_id is not None:
                                        # Calculate year from student ID
                                        student_year = self.context_builder._calculate_student_year(student_id)
                                        
                                        print(f"✅ Recognized: {student_name} ({similarity:.2%})")
                                        
                                        # Update current student
                                        self.current_student = {
                                            "id": student_id,
                                            "name": student_name,
                                            "year": student_year,
                                            "track_id": track_id
                                        }
                                        
                                        # Display name
                                        cv2.putText(frame, student_name, (x1, y2+20),
                                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                                        
                                        # Start voice interaction
                                        self.process_voice_interaction()
                                    else:
                                        print("❌ No match found")
                                        cv2.putText(frame, "Unknown", (x1, y2+20),
                                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                        else:
                            # Not looking
                            cv2.putText(frame, "Not looking", (x1, y2+20),
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 1)
                
                # === DISPLAY STATUS ===
                # FPS
                cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                # Status message
                cv2.putText(frame, self.status_message, (10, frame.shape[0]-20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.status_color, 2)
                
                # Current student
                if self.current_student:
                    student_info = f"Student: {self.current_student['name']} (Year {self.current_student['year']})"
                    cv2.putText(frame, student_info, (10, 60),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                
                # Show frame
                cv2.imshow("End-to-End Pipeline", frame)
                
                # Handle keys
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("\n👋 Quitting...")
                    break
                elif key == ord('r'):
                    print("\n🔄 Resetting recognition...")
                    self.trigger.reset()
                    self.current_student = None
                    self.status_message = "Recognition reset"
                    self.status_color = (255, 165, 0)
        
        finally:
            cap.release()
            cv2.destroyAllWindows()
            print("\n✅ Pipeline stopped")

if __name__ == "__main__":
    pipeline = EndToEndPipeline()
    pipeline.run()
