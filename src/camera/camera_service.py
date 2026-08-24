from __future__ import annotations

import contextlib
import os
import threading
import time
from pathlib import Path

import cv2

from src.camera.frame_selector import compute_sharpness
from src.common.schemas import FramePacket


class CameraService:
    def __init__(
        self,
        camera_id: int = 0,
        mirror_correction: bool = False,
        stream_fps: float = 20.0,
    ) -> None:
        self.camera_id = camera_id
        self.mirror_correction = mirror_correction
        self.stream_fps = max(stream_fps, 1.0)
        self.frame_id = 0
        self._capture = None
        self._stream_thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._latest_frame_bgr = None
        self._latest_timestamp_ms = 0
        self._frame_lock = threading.Lock()

    @contextlib.contextmanager
    def _suppress_native_stderr(self):
        saved_stderr = os.dup(2)
        try:
            with open(os.devnull, "w", encoding="utf-8") as devnull:
                os.dup2(devnull.fileno(), 2)
                yield
        finally:
            os.dup2(saved_stderr, 2)
            os.close(saved_stderr)

    def _open_capture(self):
        with self._suppress_native_stderr():
            capture = cv2.VideoCapture(self.camera_id)
            capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return capture

    def _read_stable_frame(self, capture, warmup_frames: int = 8):
        frame = None
        ok = False
        for _ in range(warmup_frames):
            with self._suppress_native_stderr():
                ok, frame = capture.read()
            time.sleep(0.03)
        return ok, frame

    def _prepare_frame(self, frame):
        if self.mirror_correction:
            return cv2.flip(frame, 1)
        return frame

    def _make_packet_from_bgr(
        self,
        frame,
        timestamp_ms: int | None = None,
        already_prepared: bool = False,
    ) -> FramePacket:
        prepared = frame if already_prepared else self._prepare_frame(frame)
        packet = FramePacket(
            frame_id=self.frame_id,
            timestamp_ms=timestamp_ms if timestamp_ms is not None else int(time.time() * 1000),
            rgb_image=cv2.cvtColor(prepared, cv2.COLOR_BGR2RGB),
            sharpness_score=compute_sharpness(prepared),
        )
        self.frame_id += 1
        return packet

    def start_stream(self) -> None:
        if self._stream_thread is not None and self._stream_thread.is_alive():
            return
        capture = self._open_capture()
        if not capture.isOpened():
            capture.release()
            raise RuntimeError(
                "Camera access failed. Grant camera permission to the current terminal or Python app in macOS System Settings."
            )
        ok, frame = self._read_stable_frame(capture)
        if not ok or frame is None:
            capture.release()
            raise RuntimeError(
                f"Failed to start camera stream from camera {self.camera_id}. Check camera permission and whether another app is using the camera."
            )
        with self._frame_lock:
            self._latest_frame_bgr = self._prepare_frame(frame)
            self._latest_timestamp_ms = int(time.time() * 1000)
        self._capture = capture
        self._stop_event.clear()
        self._stream_thread = threading.Thread(
            target=self._stream_loop,
            name="camera-stream",
            daemon=True,
        )
        self._stream_thread.start()

    def _stream_loop(self) -> None:
        interval_s = 1.0 / self.stream_fps
        while not self._stop_event.is_set():
            loop_start = time.time()
            capture = self._capture
            if capture is None:
                break
            with self._suppress_native_stderr():
                ok, frame = capture.read()
            if ok and frame is not None:
                with self._frame_lock:
                    self._latest_frame_bgr = self._prepare_frame(frame)
                    self._latest_timestamp_ms = int(time.time() * 1000)
            elapsed = time.time() - loop_start
            remaining = interval_s - elapsed
            if remaining > 0:
                time.sleep(remaining)

    def stop_stream(self) -> None:
        self._stop_event.set()
        if self._stream_thread is not None:
            self._stream_thread.join(timeout=1.0)
            self._stream_thread = None
        if self._capture is not None:
            self._capture.release()
            self._capture = None

    def get_latest_rgb_image(self):
        with self._frame_lock:
            if self._latest_frame_bgr is None:
                return None
            frame = self._latest_frame_bgr.copy()
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    def _snapshot_stream_packet(self) -> FramePacket | None:
        with self._frame_lock:
            if self._latest_frame_bgr is None:
                return None
            frame = self._latest_frame_bgr.copy()
            timestamp_ms = self._latest_timestamp_ms
        return self._make_packet_from_bgr(
            frame,
            timestamp_ms=timestamp_ms,
            already_prepared=True,
        )

    def capture_once(self) -> FramePacket:
        if self._stream_thread is not None and self._stream_thread.is_alive():
            packet = self._snapshot_stream_packet()
            if packet is not None:
                return packet
            self.start_stream()
            time.sleep(0.05)
            packet = self._snapshot_stream_packet()
            if packet is not None:
                return packet
        capture = self._open_capture()
        if not capture.isOpened():
            capture.release()
            raise RuntimeError(
                "Camera access failed. Grant camera permission to the current terminal or Python app in macOS System Settings."
            )
        ok, frame = self._read_stable_frame(capture)
        capture.release()
        if not ok or frame is None:
            raise RuntimeError(
                f"Failed to capture a stable frame from camera {self.camera_id}. Check camera permission and whether another app is using the camera."
            )
        return self._make_packet_from_bgr(frame)

    def capture_burst(self, n: int = 3) -> list[FramePacket]:
        if self._stream_thread is not None and self._stream_thread.is_alive():
            frames: list[FramePacket] = []
            for _ in range(n):
                packet = self._snapshot_stream_packet()
                if packet is None:
                    time.sleep(0.05)
                    packet = self._snapshot_stream_packet()
                if packet is None:
                    raise RuntimeError(
                        f"Failed to capture a stable burst frame from camera {self.camera_id}. Check camera permission and whether another app is using the camera."
                    )
                frames.append(packet)
                time.sleep(max(0.03, 1.0 / self.stream_fps))
            return frames
        capture = self._open_capture()
        if not capture.isOpened():
            capture.release()
            raise RuntimeError(
                "Camera access failed. Grant camera permission to the current terminal or Python app in macOS System Settings."
            )
        frames: list[FramePacket] = []
        try:
            for _ in range(n):
                ok, frame = self._read_stable_frame(capture, warmup_frames=3)
                if not ok or frame is None:
                    raise RuntimeError(
                        f"Failed to capture a stable burst frame from camera {self.camera_id}. Check camera permission and whether another app is using the camera."
                    )
                frames.append(self._make_packet_from_bgr(frame))
        finally:
            capture.release()
        return frames

    def read_image(self, image_path: str | Path) -> FramePacket:
        path = Path(image_path)
        frame = cv2.imread(str(path))
        if frame is None:
            raise FileNotFoundError(f"Could not read image: {path}")
        return self._make_packet_from_bgr(frame)
