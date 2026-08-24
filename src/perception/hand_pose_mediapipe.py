from __future__ import annotations

import contextlib
import os
from pathlib import Path
from urllib.request import urlretrieve

import numpy as np

from src.common.schemas import HandPoseResult


@contextlib.contextmanager
def _suppress_native_stderr():
    saved_stderr = os.dup(2)
    try:
        with open(os.devnull, "w", encoding="utf-8") as devnull:
            os.dup2(devnull.fileno(), 2)
            yield
    finally:
        os.dup2(saved_stderr, 2)
        os.close(saved_stderr)


class MediaPipeHandPoseEstimator:
    def __init__(self, max_num_hands: int = 1) -> None:
        self.max_num_hands = max_num_hands
        self.hands = None
        self.available = True
        self.model_path = (
            Path(__file__).resolve().parents[2]
            / "models"
            / "mediapipe"
            / "hand_landmarker.task"
        )

    def _ensure_model(self) -> None:
        if self.model_path.exists():
            return
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        urlretrieve(
            "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
            self.model_path,
        )

    def _load(self) -> None:
        if self.hands is not None or not self.available:
            return
        try:
            import importlib
            import mediapipe as mp

            self._ensure_model()
            with _suppress_native_stderr():
                mp_tasks_python = importlib.import_module("mediapipe.tasks.python")
                mp_vision = importlib.import_module("mediapipe.tasks.python.vision")
                base_options = mp_tasks_python.BaseOptions(
                    model_asset_path=str(self.model_path)
                )
                options = mp_vision.HandLandmarkerOptions(
                    base_options=base_options,
                    running_mode=mp_vision.RunningMode.IMAGE,
                    num_hands=self.max_num_hands,
                    min_hand_detection_confidence=0.5,
                    min_hand_presence_confidence=0.5,
                    min_tracking_confidence=0.5,
                )
                self.image_cls = getattr(mp, "Image")
                self.image_format_cls = getattr(mp, "ImageFormat")
                self.hands = mp_vision.HandLandmarker.create_from_options(options)
        except Exception:
            self.available = False

    def estimate_hand(self, rgb_image: np.ndarray) -> HandPoseResult:
        self._load()
        if not self.available:
            return HandPoseResult(hand_present=False)
        assert self.hands is not None
        mp_image = self.image_cls(
            image_format=self.image_format_cls.SRGB, data=rgb_image
        )
        with _suppress_native_stderr():
            result = self.hands.detect(mp_image)
        hand_landmarks = getattr(result, "hand_landmarks", [])
        if not hand_landmarks:
            return HandPoseResult(hand_present=False)
        hand_landmark_set = hand_landmarks[0]
        h, w = rgb_image.shape[:2]
        landmarks_xy = [(lm.x * w, lm.y * h) for lm in hand_landmark_set]
        handedness = "unknown"
        result_handedness = getattr(result, "handedness", [])
        if result_handedness:
            handedness = result_handedness[0][0].category_name.lower()
        return HandPoseResult(
            hand_present=True,
            score=1.0,
            handedness=handedness,
            landmarks_xy=landmarks_xy,
        )
