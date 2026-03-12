"""
Loopback test for StreamSender / StreamReceiver.
Verifies the full encode → ZMQ PUB/SUB → decode round-trip on localhost.

Run:
    python -m pytest tests/tests_Vision/test_stream_loopback.py -v
    # or directly:
    python tests/tests_Vision/test_stream_loopback.py
"""

import json
import sys
import threading
import time
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

try:
    import zmq
    _ZMQ_AVAILABLE = True
except ImportError:
    _ZMQ_AVAILABLE = False

try:
    import cv2
    _CV2_AVAILABLE = True
except ImportError:
    _CV2_AVAILABLE = False


# ── Skip entire module if deps missing ────────────────────────────────────────
pytestmark = pytest.mark.skipif(
    not (_ZMQ_AVAILABLE and _CV2_AVAILABLE),
    reason="pyzmq and/or opencv-python not installed",
)

# ── Helpers ───────────────────────────────────────────────────────────────────
TEST_PORT = 15555  # use non-standard port to avoid conflicts
TEST_HOST = "127.0.0.1"

_SYNTHETIC_RESULT = {
    "timestamp": 1741964000.0,
    "fps": 14.2,
    "frame_id": 1,
    "tracks": [
        {
            "track_id": 1,
            "name": "Test Student",
            "bbox": [10, 20, 100, 180],
            "is_looking": True,
            "confirmed": True,
        }
    ],
}


def _make_test_frame(height=240, width=320):
    """Create a synthetic BGR frame (gradient + some shapes)."""
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    frame[:, :, 1] = np.linspace(0, 255, width, dtype=np.uint8)  # green gradient
    cv2.rectangle(frame, (10, 20), (100, 180), (0, 255, 0), 2)
    return frame


# ── Tests ─────────────────────────────────────────────────────────────────────

class TestStreamSenderInit:
    def test_import(self):
        from vision.stream_sender import StreamSender, build_result
        assert StreamSender is not None
        assert callable(build_result)

    def test_instantiate(self):
        from vision.stream_sender import StreamSender
        s = StreamSender(host="0.0.0.0", port=TEST_PORT, jpeg_quality=60, resize=(80, 60))
        assert s.host == "0.0.0.0"
        assert s.port == TEST_PORT
        assert s.jpeg_quality == 60
        assert s.resize == (80, 60)

    def test_send_before_start_returns_false(self):
        from vision.stream_sender import StreamSender
        s = StreamSender(host="0.0.0.0", port=TEST_PORT + 1)
        frame = _make_test_frame()
        assert s.send(frame, _SYNTHETIC_RESULT.copy()) is False


class TestBuildResult:
    def test_empty_tracks(self):
        from vision.stream_sender import build_result
        r = build_result(
            tracks=[],
            fps=12.5,
            frame_id=42,
            confirmed_tracks=set(),
            track_names={},
        )
        assert r["fps"] == 12.5
        assert r["frame_id"] == 42
        assert r["tracks"] == []
        assert "timestamp" in r

    def test_tracks_with_names(self):
        from vision.stream_sender import build_result

        # Create a minimal mock track object
        class MockTrack:
            track_id = 7
            bbox = [10, 20, 100, 180]
            head_pose = {"is_looking": True}

        r = build_result(
            tracks=[MockTrack()],
            fps=10.0,
            frame_id=5,
            confirmed_tracks={7},
            track_names={7: "Somchai"},
        )
        t = r["tracks"][0]
        assert t["track_id"] == 7
        assert t["name"] == "Somchai"
        assert t["confirmed"] is True
        assert t["is_looking"] is True
        assert t["bbox"] == [10, 20, 100, 180]


class TestLoopback:
    """Full round-trip: sender binds, receiver connects, message exchanged."""

    def test_send_receive_roundtrip(self):
        from vision.stream_sender import StreamSender

        port = TEST_PORT + 2
        received = []

        def receiver_thread():
            ctx = zmq.Context()
            sock = ctx.socket(zmq.SUB)
            sock.setsockopt(zmq.SUBSCRIBE, b"vision")
            sock.connect(f"tcp://{TEST_HOST}:{port}")
            # Wait up to 2 seconds for a message
            if sock.poll(2000):
                parts = sock.recv_multipart()
                received.append(parts)
            sock.close()
            ctx.term()

        sender = StreamSender(host=TEST_HOST, port=port, jpeg_quality=50, resize=(80, 60))
        ok = sender.start()
        assert ok, "sender.start() failed – pyzmq missing?"

        thread = threading.Thread(target=receiver_thread, daemon=True)
        thread.start()
        time.sleep(0.3)  # allow SUB to connect before first publish

        frame = _make_test_frame()
        result = _SYNTHETIC_RESULT.copy()
        assert sender.send(frame, result) is True

        thread.join(timeout=3.0)
        sender.stop()

        assert len(received) == 1, "No message received within timeout"
        parts = received[0]
        assert len(parts) == 3
        assert parts[0] == b"vision"

        # Verify JSON
        decoded_result = json.loads(parts[1].decode("utf-8"))
        assert decoded_result["fps"] == result["fps"]
        assert decoded_result["frame_id"] == result["frame_id"]

        # Verify JPEG decodes back to a valid image
        buf = np.frombuffer(parts[2], dtype=np.uint8)
        decoded_frame = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        assert decoded_frame is not None
        assert decoded_frame.shape[0] == 60   # resize height
        assert decoded_frame.shape[1] == 80   # resize width


# ── Standalone runner ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    if not _ZMQ_AVAILABLE:
        print("SKIP: pyzmq not installed")
        sys.exit(0)
    if not _CV2_AVAILABLE:
        print("SKIP: opencv-python not installed")
        sys.exit(0)

    print("Running stream loopback tests…")
    suite = [
        TestStreamSenderInit().test_import,
        TestStreamSenderInit().test_instantiate,
        TestStreamSenderInit().test_send_before_start_returns_false,
        TestBuildResult().test_empty_tracks,
        TestBuildResult().test_tracks_with_names,
        TestLoopback().test_send_receive_roundtrip,
    ]
    failed = 0
    for fn in suite:
        try:
            fn()
            print(f"  PASS  {fn.__name__}")
        except Exception as exc:
            print(f"  FAIL  {fn.__name__}: {exc}")
            failed += 1
    if failed:
        print(f"\n{failed} test(s) FAILED")
        sys.exit(1)
    else:
        print("\nAll tests PASSED")
