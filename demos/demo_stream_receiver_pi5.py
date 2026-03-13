"""
Pi5 receiver demo – displays the Jetson stream in an OpenCV window.

Usage:
    python demos/demo_stream_receiver_pi5.py --host 192.168.1.10 --port 5555
"""
import argparse
import sys
import cv2
import numpy as np
from loguru import logger

sys.path.insert(0, "src")
from vision.stream_receiver import StreamReceiver


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--host", default="192.168.1.10", help="Jetson IP")
    p.add_argument("--port", type=int, default=5555)
    p.add_argument("--no-display", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()

    def on_frame(frame, result):
        fps = result.get("fps", 0)
        tracks = result.get("tracks", [])
        logger.info(f"FPS={fps:.1f}  tracks={len(tracks)}")
        for t in tracks:
            name = t.get("name") or "Unknown"
            tid = t["track_id"]
            confirmed = "✓" if t.get("confirmed") else ""
            logger.info(f"  Track {tid}: {name} {confirmed}")

        if not args.no_display:
            # Draw simple overlays
            for t in tracks:
                x1, y1, x2, y2 = t["bbox"]
                label = t.get("name") or f"ID:{t['track_id']}"
                color = (0, 255, 0) if t.get("confirmed") else (0, 200, 255)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            cv2.putText(frame, f"FPS: {fps:.1f}", (10, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            cv2.imshow("Jetson Stream  (Pi5 Receiver)", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                receiver.stop()

    receiver = StreamReceiver(host=args.host, port=args.port)
    try:
        receiver.start(callback=on_frame)
    except KeyboardInterrupt:
        logger.info("Interrupted.")
    finally:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
