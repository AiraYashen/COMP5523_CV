import numpy as np

from src.common.schemas import Detection, FramePacket, HandPoseResult
from src.fusion.fusion_state import build_fusion_state


def test_build_fusion_state_computes_offsets() -> None:
    frame = FramePacket(
        frame_id=1, timestamp_ms=1, rgb_image=np.zeros((100, 200, 3), dtype=np.uint8)
    )
    detection = Detection(bbox_xyxy=(100, 40, 140, 80), label="can", score=0.9)
    depth = np.ones((100, 200), dtype=np.float32)
    depth[50:60, 115:125] = 0.8
    hand_landmarks = [(80.0, 50.0)] * 21
    hand_pose = HandPoseResult(
        hand_present=True, score=1.0, handedness="right", landmarks_xy=hand_landmarks
    )

    state = build_fusion_state(frame, detection, depth, hand_pose, target_locked=True)
    assert state.target_visible is True
    assert state.hand_visible is True
    assert state.dx_norm is not None
    assert state.dx_norm > 0
