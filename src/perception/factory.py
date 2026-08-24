from __future__ import annotations

from src.perception.depth_anything import DepthAnythingEstimator
from src.perception.depth_mock import MockDepthEstimator
from src.perception.detector_grounding_dino import GroundingDinoDetector
from src.perception.detector_mock import MockDetector
from src.perception.hand_pose_mock import MockHandPoseEstimator
from src.perception.hand_pose_mediapipe import MediaPipeHandPoseEstimator


def build_detector(backend: str, model_id: str, device: str):
    if backend == "grounding_dino":
        return GroundingDinoDetector(model_id, device=device)
    if backend == "mock":
        return MockDetector()
    raise ValueError(f"Unsupported detector backend: {backend}")


def build_depth_estimator(backend: str, model_id: str, device: str):
    if backend == "depth_anything":
        return DepthAnythingEstimator(model_id, device=device)
    if backend == "mock":
        return MockDepthEstimator()
    raise ValueError(f"Unsupported depth backend: {backend}")


def build_hand_estimator(backend: str):
    if backend == "mediapipe":
        return MediaPipeHandPoseEstimator(max_num_hands=1)
    if backend == "mock":
        return MockHandPoseEstimator()
    raise ValueError(f"Unsupported hand backend: {backend}")
