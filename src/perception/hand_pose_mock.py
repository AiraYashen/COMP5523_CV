from __future__ import annotations

import numpy as np

from src.common.schemas import HandPoseResult


class MockHandPoseEstimator:
    def estimate_hand(self, rgb_image: np.ndarray) -> HandPoseResult:
        _ = rgb_image
        return HandPoseResult(hand_present=False)
