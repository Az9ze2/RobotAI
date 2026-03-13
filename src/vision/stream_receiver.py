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
