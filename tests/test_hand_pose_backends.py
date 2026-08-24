import numpy as np

from src.perception.hand_pose_mock import MockHandPoseEstimator


def test_mock_hand_pose_returns_no_hand() -> None:
    image = np.zeros((120, 160, 3), dtype=np.uint8)
    estimator = MockHandPoseEstimator()
    result = estimator.estimate_hand(image)
    assert result.hand_present is False
    assert result.landmarks_xy == []
