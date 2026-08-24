from __future__ import annotations

from src.camera.camera_service import CameraService
from src.camera.frame_selector import select_sharpest
from src.common.schemas import FramePacket


class CaptureController:
    def __init__(self, camera_service: CameraService, burst_count: int = 3) -> None:
        self.camera_service = camera_service
        self.burst_count = burst_count

    def get_initial_frame(self, image_path: str | None = None) -> FramePacket:
        if image_path:
            return self.camera_service.read_image(image_path)
        frames = self.camera_service.capture_burst(self.burst_count)
        return select_sharpest(frames)

    def get_session_frame(self, image_path: str | None = None) -> FramePacket:
        if image_path:
            return self.camera_service.read_image(image_path)
        return self.camera_service.capture_once()
