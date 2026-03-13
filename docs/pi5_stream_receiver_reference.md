# Pi5 LAN Stream Receiver – Reference Documentation

This document is the **complete reference** for setting up the Raspberry Pi 5 as a downstream consumer of the Jetson's ZMQ vision stream. Implement this when you are ready to work on the Pi5 side.

---

## Overview

```
┌────────────────────────────┐       LAN cable       ┌────────────────────┐
│  Jetson Orin Nano          │ ──────────────────→   │  Network Switch    │
│  RealSense D435i attached  │                        └────────┬───────────┘
│  IP: 192.168.1.10          │                                 │
│  ZMQ PUB :5555             │              ┌──────────────────┴──────────────────┐
└────────────────────────────┘              │ LAN cables                          │
                                   ┌────────┴───────────┐                ┌────────┴───────────┐
                                   │  Raspberry Pi 5 #1 │                │  Raspberry Pi 5 #2 │
                                   │  IP: 192.168.1.20  │                │  IP: 192.168.1.30  │
                                   │  ZMQ SUB :5555     │                │  ZMQ SUB :5555     │
                                   └────────────────────┘                └────────────────────┘
```

The Jetson publishes a **ZMQ PUB** multipart message on every processed frame. The Pi5 subscribes to the `"vision"` topic and decodes each message.

---

## Message Format

Every message is a **3-part multipart ZMQ message**:

| Part | Content | Encoding |
|------|---------|----------|
| 0 | `b"vision"` | Raw bytes (topic filter) |
| 1 | JSON payload | UTF-8 |
| 2 | JPEG frame | Binary (OpenCV imencode) |

### JSON Payload Schema

```json
{
  "timestamp": 1741964000.123,
  "fps": 14.2,
  "frame_id": 1337,
  "tracks": [
    {
      "track_id": 1,
      "name": "Somchai",
      "bbox": [120, 80, 300, 340],
      "is_looking": true,
      "confirmed": true
    }
  ]
}
```

- `name` is `null` if the person has not yet been recognised.
- `confirmed` is `true` once recognition has succeeded (name is locked in).
- JPEG frame is resized to **320×240** by default (configurable via `--stream-width / --stream-height`).

---

## Network Setup (Static IPs)

> [!IMPORTANT]
> Both devices need static IPs in the same subnet. No router required – the switch alone is sufficient.

### On the Jetson

```bash
# Find the ethernet connection name
nmcli con show

# Set static IP (replace "Wired connection 1" with your connection name)
sudo nmcli con mod "Wired connection 1" \
    ipv4.addresses 192.168.1.10/24 \
    ipv4.method manual
sudo nmcli con up "Wired connection 1"

# Verify
ip addr show
```

### On the Pi5 (#1 - Device 20)

```bash
# NMCLI (Standard method)
sudo nmcli con mod "Wired connection 1" \
    ipv4.addresses 192.168.1.20/24 \
    ipv4.method manual
sudo nmcli con up "Wired connection 1"

# Manual Workaround (If NMCLI fails/times out)
sudo ip addr add 192.168.1.20/24 dev eth0
```

### On the Pi5 (#2 - Device 30)

```bash
# NMCLI (Standard method)
sudo nmcli con mod "Wired connection 1" \
    ipv4.addresses 192.168.1.30/24 \
    ipv4.method manual
sudo nmcli con up "Wired connection 1"

# Manual Workaround (If NMCLI fails/times out)
sudo ip addr add 192.168.1.30/24 dev eth0
```

Ping test:
```bash
# From Pi5 #1
ping 192.168.1.10
ping 192.168.1.30

# From Pi5 #2
ping 192.168.1.10
ping 192.168.1.20

# From Jetson
ping 192.168.1.20
ping 192.168.1.30
```

---

## Pi5 Installation

```bash
# Python deps (Raspberry Pi OS Bookworm)
pip install pyzmq>=25.0 opencv-python numpy loguru

# Optional: headless OpenCV (lighter)
pip install opencv-python-headless numpy loguru pyzmq
```

---

## stream_receiver.py (Pi5 module)

Create this at `src/vision/stream_receiver.py` **on the Pi5**:

```python
"""
stream_receiver.py  –  Pi5 side
================================
Subscribes to the Jetson's ZMQ PUB socket and decodes each frame + result.
"""
import json
import time
from typing import Callable
from loguru import logger

import zmq
import cv2
import numpy as np


class StreamReceiver:
    """
    ZMQ SUB socket that receives vision data from the Jetson.

    Usage::

        def on_frame(frame, result):
            print(result["tracks"])
            cv2.imshow("Stream", frame)
            cv2.waitKey(1)

        receiver = StreamReceiver(host="192.168.1.10", port=5555)
        receiver.start(callback=on_frame)   # blocks until stop() called
    """

    TOPIC = b"vision"

    def __init__(self, host: str = "192.168.1.10", port: int = 5555):
        self.host = host
        self.port = port
        self._ctx = None
        self._sock = None
        self._running = False
        self._frames_received = 0

    def start(self, callback: Callable | None = None):
        """Connect and enter receive loop. `callback(frame, result)` is called per frame."""
        self._ctx = zmq.Context()
        self._sock = self._ctx.socket(zmq.SUB)
        self._sock.setsockopt(zmq.SUBSCRIBE, self.TOPIC)
        addr = f"tcp://{self.host}:{self.port}"
        self._sock.connect(addr)
        self._running = True
        logger.info(f"StreamReceiver connected to {addr}")

        try:
            while self._running:
                if self._sock.poll(10):   # 10 ms timeout
                    parts = self._sock.recv_multipart()
                    if len(parts) != 3:
                        continue
                    _, json_bytes, jpeg_bytes = parts
                    try:
                        result = json.loads(json_bytes.decode("utf-8"))
                        buf = np.frombuffer(jpeg_bytes, dtype=np.uint8)
                        frame = cv2.imdecode(buf, cv2.IMREAD_COLOR)
                        self._frames_received += 1
                        if callback and frame is not None:
                            callback(frame, result)
                    except Exception as e:
                        logger.error(f"Decode error: {e}")
        finally:
            self.stop()

    def stop(self):
        self._running = False
        if self._sock:
            self._sock.close()
            self._sock = None
        if self._ctx:
            self._ctx.term()
            self._ctx = None
        logger.info(f"StreamReceiver stopped. Frames received: {self._frames_received}")

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.stop()
```

---

## demo_stream_receiver_pi5.py (standalone Pi5 demo)

Create this at `demos/demo_stream_receiver_pi5.py` **on the Pi5**:

```python
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
```

---

## Firewall / Port

On the Jetson, ensure port `5555/tcp` is open:

```bash
# Check UFW status
sudo ufw status

# Allow if needed
sudo ufw allow 5555/tcp
```

---

## Troubleshooting

| Symptom | Likely cause | Fix |
|---------|-------------|-----|
| `Cannot connect` | Wrong IP or port | `ping` both directions; check `--stream-host` on Jetson |
| Frame is blank / black | JPEG decode error | Check `--jpeg-quality` is ≥ 10 |
| No messages arriving | Jetson `--stream-enabled` not set | Re-launch Jetson with the flag |
| High latency | Large frame size | Reduce `--stream-width/height` or `--jpeg-quality` |
| `ImportError: zmq` | pyzmq not installed | `pip install pyzmq>=25.0` on both devices |

---

*Last updated: 2026-03-12. Refer to `src/vision/stream_sender.py` for the Jetson-side implementation.*
