from src.common.schemas import Detection
from src.fusion.target_selector import select_target


def test_select_target_prefers_left_for_left_hint() -> None:
    detections = [
        Detection(bbox_xyxy=(10, 10, 50, 50), label="left object", score=0.8),
        Detection(bbox_xyxy=(300, 10, 340, 50), label="right object", score=0.9),
    ]
    selected = select_target(detections, "left", (400, 300), None)
    assert selected is not None
    assert selected.label == "left object"


def test_select_target_uses_stability_when_available() -> None:
    detections = [
        Detection(bbox_xyxy=(90, 90, 140, 140), label="stable object", score=0.6),
        Detection(bbox_xyxy=(250, 90, 300, 140), label="new object", score=0.8),
    ]
    selected = select_target(detections, "none", (400, 300), (100, 100, 150, 150))
    assert selected is not None
    assert selected.label == "stable object"
