from __future__ import annotations

import math

import numpy as np


def _median_valid(values: np.ndarray) -> float:
    valid = values[np.isfinite(values)]
    if valid.size == 0:
        return math.nan
    return float(np.median(valid))


def sample_box_center_depth(
    depth_map: np.ndarray, bbox_xyxy: tuple[float, float, float, float]
) -> float:
    x1, y1, x2, y2 = bbox_xyxy
    cx = int((x1 + x2) / 2)
    cy = int((y1 + y2) / 2)
    half_w = max(2, int((x2 - x1) * 0.15))
    half_h = max(2, int((y2 - y1) * 0.15))
    y_start = max(0, cy - half_h)
    y_end = min(depth_map.shape[0], cy + half_h + 1)
    x_start = max(0, cx - half_w)
    x_end = min(depth_map.shape[1], cx + half_w + 1)
    return _median_valid(depth_map[y_start:y_end, x_start:x_end])


def sample_point_depth(depth_map: np.ndarray, point_xy: tuple[float, float]) -> float:
    x, y = int(point_xy[0]), int(point_xy[1])
    y_start = max(0, y - 2)
    y_end = min(depth_map.shape[0], y + 3)
    x_start = max(0, x - 2)
    x_end = min(depth_map.shape[1], x + 3)
    return _median_valid(depth_map[y_start:y_end, x_start:x_end])
