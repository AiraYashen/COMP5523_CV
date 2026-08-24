from __future__ import annotations

import numpy as np

from src.common.schemas import DepthResult


class MockDepthEstimator:
    def __init__(self) -> None:
        pass

    def predict_depth(self, rgb_image: np.ndarray) -> DepthResult:
        image_h, image_w = rgb_image.shape[:2]
        y_coords = np.linspace(0.0, 1.0, image_h, dtype=np.float32).reshape(image_h, 1)
        x_coords = np.linspace(0.0, 1.0, image_w, dtype=np.float32).reshape(1, image_w)

        vertical = np.repeat(y_coords, image_w, axis=1)
        center_bias = 1.0 - np.abs(x_coords - 0.5) * 0.3
        center_bias = np.repeat(center_bias, image_h, axis=0)

        gray = rgb_image.astype(np.float32).mean(axis=2) / 255.0
        depth_map = 0.65 * vertical + 0.2 * center_bias + 0.15 * (1.0 - gray)
        depth_map = depth_map.astype(np.float32)
        return DepthResult(
            depth_map=depth_map,
            min_depth=float(np.nanmin(depth_map)),
            max_depth=float(np.nanmax(depth_map)),
        )
