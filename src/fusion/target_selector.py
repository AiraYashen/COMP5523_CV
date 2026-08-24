from __future__ import annotations

from src.common.schemas import Detection


def _spatial_hint_score(
    bbox_xyxy: tuple[float, float, float, float],
    spatial_hint: str,
    image_size: tuple[int, int],
) -> float:
    width, _ = image_size
    center_x = (bbox_xyxy[0] + bbox_xyxy[2]) / 2
    relative_x = center_x / width
    if spatial_hint == "left":
        return max(0.0, 1.0 - relative_x * 1.5)
    if spatial_hint == "right":
        return max(0.0, relative_x * 1.5)
    if spatial_hint in {"center", "front-center"}:
        return max(0.0, 1.0 - abs(relative_x - 0.5) * 2)
    return 0.5


def _stability_score(
    bbox_xyxy: tuple[float, float, float, float],
    previous_box: tuple[float, float, float, float] | None,
) -> float:
    if previous_box is None:
        return 0.5
    cx = (bbox_xyxy[0] + bbox_xyxy[2]) / 2
    cy = (bbox_xyxy[1] + bbox_xyxy[3]) / 2
    pcx = (previous_box[0] + previous_box[2]) / 2
    pcy = (previous_box[1] + previous_box[3]) / 2
    distance = abs(cx - pcx) + abs(cy - pcy)
    return max(0.0, 1.0 - distance / 200.0)


def select_target(
    detections: list[Detection],
    spatial_hint: str,
    image_size: tuple[int, int],
    previous_box: tuple[float, float, float, float] | None,
    score_weights: dict[str, float] | None = None,
) -> Detection | None:
    if not detections:
        return None
    weights = score_weights or {
        "target_score_weight": 0.5,
        "spatial_score_weight": 0.3,
        "stability_score_weight": 0.2,
    }
    scored: list[tuple[float, Detection]] = []
    for detection in detections:
        total = (
            weights["target_score_weight"] * detection.score
            + weights["spatial_score_weight"]
            * _spatial_hint_score(detection.bbox_xyxy, spatial_hint, image_size)
            + weights["stability_score_weight"]
            * _stability_score(detection.bbox_xyxy, previous_box)
        )
        scored.append((total, detection))
    scored.sort(key=lambda item: item[0], reverse=True)
    return scored[0][1]
