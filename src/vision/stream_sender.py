"""
stream_sender.py  –  Jetson side
================================
Publishes vision results (JPEG frame + JSON metadata) over a ZMQ PUB socket
so that any device on the same LAN (e.g. a Raspberry Pi 5 connected via a
network switch) can subscribe and consume the stream.

Protocol (multipart ZMQ message)
---------------------------------
  Frame 0 : b"vision"          – topic string for SUB filter
  Frame 1 : JSON bytes         – structured recognition results
  Frame 2 : JPEG bytes         – compressed frame (resized for bandwidth)

Install dependency (both Jetson JetPack 6 and Pi OS bookworm):
  pip install pyzmq>=25.0
"""

import json
import time
from loguru import logger

try:
    import zmq
    _ZMQ_AVAILABLE = True
except ImportError:
    _ZMQ_AVAILABLE = False
    logger.warning("pyzmq not installed – StreamSender will be a no-op. "
                   "Install with: pip install pyzmq")

try:
    import cv2
    import numpy as np
    _CV2_AVAILABLE = True
except ImportError:
    _CV2_AVAILABLE = False


class StreamSender:
    """
    ZMQ PUB socket that streams vision results from the Jetson to subscribers.

    Usage (Jetson side)::

        sender = StreamSender(host="0.0.0.0", port=5555,
                              jpeg_quality=60,
                              resize=(320, 240))
        sender.start()

        # inside frame loop …
        result = build_result_dict(tracks, fps, frame_id)
        sender.send(frame_bgr, result)

        sender.stop()
    """

    TOPIC = b"vision"

    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int = 5555,
        jpeg_quality: int = 60,
        resize: tuple[int, int] | None = (320, 240),
    ):
        """
        Parameters
        ----------
        host : str
            Interface to bind on. "0.0.0.0" binds all interfaces (default).
        port : int
            TCP port for the ZMQ PUB socket (default 5555).
        jpeg_quality : int
            JPEG compression quality 0-100. Lower = smaller packets (default 60).
        resize : (width, height) | None
            Resize frame before encoding. None = send at original resolution.
        """
        self.host = host
        self.port = port
        self.jpeg_quality = jpeg_quality
        self.resize = resize

        self._ctx = None
        self._sock = None
        self._running = False
        self._frame_count_sent = 0

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> bool:
        """Bind the PUB socket. Returns True on success, False if pyzmq unavailable."""
        if not _ZMQ_AVAILABLE:
            logger.warning("StreamSender.start() skipped – pyzmq not installed.")
            return False

        self._ctx = zmq.Context()
        self._sock = self._ctx.socket(zmq.PUB)
        addr = f"tcp://{self.host}:{self.port}"
        self._sock.bind(addr)
        self._running = True
        logger.info(f"StreamSender bound on {addr} (JPEG quality={self.jpeg_quality}, resize={self.resize})")
        return True

    def stop(self):
        """Close the PUB socket and ZMQ context gracefully."""
        self._running = False
        if self._sock is not None:
            self._sock.close()
            self._sock = None
        if self._ctx is not None:
            self._ctx.term()
            self._ctx = None
        logger.info(f"StreamSender stopped. Total frames sent: {self._frame_count_sent}")

    # ------------------------------------------------------------------
    # Send
    # ------------------------------------------------------------------

    def send(self, frame, result: dict) -> bool:
        """
        Encode *frame* as JPEG and publish it together with *result*.

        Parameters
        ----------
        frame : np.ndarray (BGR)
            Raw camera frame from OpenCV / RealSense.
        result : dict
            Must follow the schema::

                {
                    "timestamp": float,   # time.time()
                    "fps": float,
                    "frame_id": int,
                    "tracks": [
                        {
                            "track_id": int,
                            "name": str | None,
                            "bbox": [x1, y1, x2, y2],
                            "is_looking": bool,
                            "confirmed": bool,
                        },
                        ...
                    ]
                }

        Returns
        -------
        bool
            True if message was sent, False otherwise.
        """
        if not self._running or self._sock is None:
            return False

        if not _CV2_AVAILABLE:
            logger.warning("cv2 not available – cannot encode frame.")
            return False

        try:
            # --- Resize for bandwidth efficiency ---
            if self.resize is not None:
                send_frame = cv2.resize(frame, self.resize,
                                        interpolation=cv2.INTER_LINEAR)
            else:
                send_frame = frame

            # --- JPEG encode ---
            encode_params = [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality]
            ok, jpeg_buf = cv2.imencode(".jpg", send_frame, encode_params)
            if not ok:
                logger.warning("JPEG encoding failed – skipping frame.")
                return False
            jpeg_bytes = jpeg_buf.tobytes()

            # --- JSON payload ---
            result.setdefault("timestamp", time.time())
            json_bytes = json.dumps(result, ensure_ascii=False).encode("utf-8")

            # --- Publish multipart (non-blocking) ---
            self._sock.send_multipart(
                [self.TOPIC, json_bytes, jpeg_bytes],
                flags=zmq.NOBLOCK,
            )
            self._frame_count_sent += 1
            return True

        except zmq.Again:
            # No subscribers yet – silently discard
            return False
        except Exception as e:
            logger.error(f"StreamSender.send() error: {e}")
            return False

    # ------------------------------------------------------------------
    # Context manager support
    # ------------------------------------------------------------------

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *_):
        self.stop()


# ---------------------------------------------------------------------------
# Helper – build a result dict from pipeline data
# ---------------------------------------------------------------------------

def build_result(
    tracks: list,
    fps: float,
    frame_id: int,
    confirmed_tracks: set,
    track_names: dict,
) -> dict:
    """Build the JSON-serialisable result dict from pipeline state.

    Parameters
    ----------
    tracks : list
        ByteTracker track objects from the current frame.
    fps : float
        Current frames-per-second reading.
    frame_id : int
        Monotonically increasing frame counter.
    confirmed_tracks : set[int]
        Set of track IDs that have been recognised.
    track_names : dict[int, str]
        Mapping of track_id → student name.
    """
    track_list = []
    for t in tracks:
        tid = getattr(t, "track_id", -1)
        bbox = list(getattr(t, "bbox", [0, 0, 0, 0]))
        head_pose = getattr(t, "head_pose", None) or {}
        track_list.append(
            {
                "track_id": tid,
                "name": track_names.get(tid),          # None if not yet recognised
                "bbox": [int(v) for v in bbox],
                "is_looking": bool(head_pose.get("is_looking", False)),
                "confirmed": tid in confirmed_tracks,
            }
        )
    return {
        "timestamp": time.time(),
        "fps": round(fps, 2),
        "frame_id": frame_id,
        "tracks": track_list,
    }
