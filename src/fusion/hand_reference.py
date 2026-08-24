from __future__ import annotations


PALM_INDEXES = [0, 5, 9, 13, 17]


def compute_palm_center(landmarks_xy: list[tuple[float, float]]) -> tuple[float, float]:
    if len(landmarks_xy) < max(PALM_INDEXES) + 1:
        raise ValueError("Not enough landmarks to compute palm center.")
    xs = [landmarks_xy[idx][0] for idx in PALM_INDEXES]
    ys = [landmarks_xy[idx][1] for idx in PALM_INDEXES]
    return (sum(xs) / len(xs), sum(ys) / len(ys))
