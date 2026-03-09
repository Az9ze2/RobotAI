"""
Real-Time Vision Pipeline Demo - RealSense Performance Benchmark

This script bypasses the InsightFace/ONNX pipeline (due to Windows build constraints)
and directly benchmarks the Intel RealSense D415 camera's startup time and rendering FPS.
"""

import cv2
import numpy as np
import sys
import time
from datetime import datetime

class RealSenseBenchmarkDemo:
    def __init__(self):
        print("Initializing RealSense Benchmark Demo...")
        self.frame_count = 0
        self.fps = 0.0
        self.last_time = datetime.now()

    def run(self):
        print("Starting Intel RealSense camera...")
        print("Press 'q' to quit, 's' to save screenshot\n")
        
        import pyrealsense2 as rs
        
        pipeline = rs.pipeline()
        config = rs.config()

        pipeline_wrapper = rs.pipeline_wrapper(pipeline)
        try:
            pipeline_profile = config.resolve(pipeline_wrapper)
            device = pipeline_profile.get_device()
            
            found_rgb = False
            for s in device.sensors:
                if s.get_info(rs.camera_info.name) == 'RGB Camera':
                    found_rgb = True
                    break
            if not found_rgb:
                print("❌ The camera does not have an RGB sensor!")
                return
        except Exception as e:
            print(f"❌ Failed to find RealSense camera: {e}")
            return

        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        
        try:
            t0 = time.time()
            pipeline.start(config)
            t1 = time.time()
            startup_time = t1 - t0
            print(f"✅ Pipeline started successfully in {startup_time:.2f} seconds.")
        except Exception as e:
            print(f"❌ Failed to start RealSense pipeline: {e}")
            return
            
        cv2.namedWindow("Vision Pipeline Demo - RealSense", cv2.WINDOW_NORMAL)
        
        while True:
            try:
                frames = pipeline.wait_for_frames()
                color_frame = frames.get_color_frame()
                if not color_frame:
                    continue
                frame = np.asanyarray(color_frame.get_data())
            except Exception as e:
                print(f"Failed to read frame: {e}")
                break
            
            current_time = datetime.now()
            elapsed = (current_time - self.last_time).total_seconds()
            if elapsed > 0:
                # Exponential moving average for smoother FPS
                current_fps = 1.0 / elapsed
                if self.fps == 0:
                    self.fps = current_fps
                else:
                    self.fps = self.fps * 0.9 + current_fps * 0.1
                    
            self.last_time = current_time
            self.frame_count += 1
            
            # Draw info panel
            h, w = frame.shape[:2]
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (300, 150), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
            
            cv2.putText(frame, "REALSENSE BENCHMARK", (10, 30), cv2.FONT_HERSHEY_DUPLEX, 0.6, (0, 255, 255), 2)
            cv2.putText(frame, f"FPS: {self.fps:.1f}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"Startup Time: {startup_time:.2f}s", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            cv2.putText(frame, f"Resolution: {w}x{h}", (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            
            cv2.imshow("Vision Pipeline Demo - RealSense", frame)
            
            # Auto-benchmark termination
            if self.frame_count >= 50:
                print(f"BENCHMARK COMPLETE:")
                print(f"  Startup Time: {startup_time:.2f} seconds")
                print(f"  Average FPS (last 50 frames): {self.fps:.1f} fps")
                print("  Resolution: 640x480")
                break
                
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("\nQuitting...")
                break
            elif key == ord('s'):
                filename = f"screenshot_realsense_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
                cv2.imwrite(filename, frame)
                print(f"  Screenshot saved: {filename}")
        
        pipeline.stop()
        cv2.destroyAllWindows()
        print("Demo ended.")

def main():
    print("=" * 60)
    print("REAL-TIME VISION PIPELINE BENCHMARK (Intel RealSense)")
    print("=" * 60)
    try:
        demo = RealSenseBenchmarkDemo()
        demo.run()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        return 1
    return 0

if __name__ == "__main__":
    sys.exit(main())
