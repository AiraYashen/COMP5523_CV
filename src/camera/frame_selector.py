from __future__ import annotations

import cv2

from src.common.schemas import FramePacket


def compute_sharpness(image) -> float:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def select_sharpest(frames: list[FramePacket]) -> FramePacket:
    if not frames:
        raise ValueError("No frames provided for selection.")
    return max(frames, key=lambda frame: frame.sharpness_score)
