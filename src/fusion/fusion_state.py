from __future__ import annotations

from src.common.schemas import Detection, FramePacket, FusionState, HandPoseResult
from src.fusion.depth_sampler import sample_box_center_depth, sample_point_depth
from src.fusion.hand_reference import compute_palm_center


def build_fusion_state(
    frame: FramePacket,
    detection: Detection | None,
    depth_map,
    hand_pose: HandPoseResult,
    target_locked: bool,
) -> FusionState:
    target_center_xy = None
    hand_center_xy = None
    target_depth = None
    hand_depth = None
    dx_norm = None
    dy_norm = None
    dz_rel = None

    image_h, image_w = frame.rgb_image.shape[:2]
    if detection is not None:
        x1, y1, x2, y2 = detection.bbox_xyxy
        target_center_xy = ((x1 + x2) / 2, (y1 + y2) / 2)
        target_depth = sample_box_center_depth(depth_map, detection.bbox_xyxy)
    if hand_pose.hand_present and hand_pose.landmarks_xy:
        hand_center_xy = compute_palm_center(hand_pose.landmarks_xy)
        hand_depth = sample_point_depth(depth_map, hand_center_xy)

    if target_center_xy is not None and hand_center_xy is not None:
        dx_norm = (target_center_xy[0] - hand_center_xy[0]) / image_w
        dy_norm = (target_center_xy[1] - hand_center_xy[1]) / image_h
    if target_depth is not None and hand_depth is not None:
        dz_rel = target_depth - hand_depth

    return FusionState(
        target_visible=detection is not None,
        hand_visible=hand_pose.hand_present,
        target_locked=target_locked,
        target_center_xy=target_center_xy,
        hand_center_xy=hand_center_xy,
        target_depth=target_depth,
        hand_depth=hand_depth,
        dx_norm=dx_norm,
        dy_norm=dy_norm,
        dz_rel=dz_rel,
        target_confidence=detection.score if detection is not None else 0.0,
        hand_confidence=hand_pose.score,
        frame_id=frame.frame_id,
        timestamp_ms=frame.timestamp_ms,
        frame_size_xy=(image_w, image_h),
    )
