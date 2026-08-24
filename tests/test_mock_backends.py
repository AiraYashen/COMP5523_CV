import numpy as np

from src.perception.depth_mock import MockDepthEstimator
from src.perception.detector_mock import MockDetector


def test_mock_detector_finds_red_region_for_coke_prompt() -> None:
    image = np.ones((240, 320, 3), dtype=np.uint8) * 255
    image[100:200, 130:180] = [255, 0, 0]
    detector = MockDetector()
    detections = detector.detect(image, "red soda can")
    assert detections
    best = detections[0]
    assert best.score >= 0.3
    assert best.bbox_xyxy[0] < 180
    assert best.bbox_xyxy[2] > 130


def test_mock_depth_returns_same_size_map() -> None:
    image = np.zeros((120, 160, 3), dtype=np.uint8)
    estimator = MockDepthEstimator()
    result = estimator.predict_depth(image)
    assert result.depth_map.shape == (120, 160)
    assert result.max_depth > result.min_depth
