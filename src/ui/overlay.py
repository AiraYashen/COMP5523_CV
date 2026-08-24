from __future__ import annotations

import cv2
import numpy as np

from src.common.schemas import Detection, FusionState, HandPoseResult

TITLE_BAR_HEIGHT = 88
TITLE_FONT_SCALE = 1.2
TITLE_FONT_THICKNESS = 3
SUBTITLE_FONT_SCALE = 0.78
SUBTITLE_FONT_THICKNESS = 2
STATUS_FONT_SCALE = 0.92
STATUS_FONT_THICKNESS = 2
DETECTION_FONT_SCALE = 0.95
DETECTION_FONT_THICKNESS = 2
PANEL_HEIGHT = 420
PANEL_WIDTH = 640


def _resize_panel(image: np.ndarray, width: int, height: int) -> np.ndarray:
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_LINEAR)


def _draw_title(frame: np.ndarray, title: str, subtitle: str = "") -> np.ndarray:
    cv2.rectangle(frame, (0, 0), (frame.shape[1], TITLE_BAR_HEIGHT), (20, 20, 20), -1)
    cv2.putText(
        frame,
        title,
        (16, 38),
        cv2.FONT_HERSHEY_SIMPLEX,
        TITLE_FONT_SCALE,
        (255, 255, 255),
        TITLE_FONT_THICKNESS,
    )
    if subtitle:
        cv2.putText(
            frame,
            subtitle,
            (18, 72),
            cv2.FONT_HERSHEY_SIMPLEX,
            SUBTITLE_FONT_SCALE,
            (210, 210, 210),
            SUBTITLE_FONT_THICKNESS,
        )
    cv2.rectangle(frame, (0, 0), (frame.shape[1] - 1, frame.shape[0] - 1), (60, 60, 60), 2)
    return frame


def _normalize_detections(
    detections: Detection | list[Detection] | None,
) -> list[Detection]:
    if detections is None:
        return []
    if isinstance(detections, list):
        return detections
    return [detections]


def _draw_detection_panel(
    rgb_image, detections: Detection | list[Detection] | None, fusion_state: FusionState
):
    frame = cv2.cvtColor(rgb_image.copy(), cv2.COLOR_RGB2BGR)
    for detection in _normalize_detections(detections):
        x1, y1, x2, y2 = map(int, detection.bbox_xyxy)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
        cv2.putText(
            frame,
            f"{detection.label} {detection.score:.2f}",
            (x1, max(TITLE_BAR_HEIGHT + 24, y1 - 14)),
            cv2.FONT_HERSHEY_SIMPLEX,
            DETECTION_FONT_SCALE,
            (0, 255, 0),
            DETECTION_FONT_THICKNESS,
        )
    if fusion_state.target_center_xy is not None:
        cv2.circle(
            frame,
            (
                int(fusion_state.target_center_xy[0]),
                int(fusion_state.target_center_xy[1]),
            ),
            6,
            (0, 255, 255),
            -1,
        )
    return _draw_title(frame, "Grounding DINO", "Object Detection")


def _draw_hand_panel(rgb_image, hand_pose: HandPoseResult, fusion_state: FusionState):
    frame = cv2.cvtColor(rgb_image.copy(), cv2.COLOR_RGB2BGR)
    for idx, point in enumerate(hand_pose.landmarks_xy):
        cv2.circle(frame, (int(point[0]), int(point[1])), 4, (255, 0, 0), -1)
        if idx in {0, 5, 9, 13, 17}:
            cv2.circle(frame, (int(point[0]), int(point[1])), 6, (0, 255, 255), 2)
    if fusion_state.hand_center_xy is not None:
        cv2.circle(
            frame,
            (int(fusion_state.hand_center_xy[0]), int(fusion_state.hand_center_xy[1])),
            7,
            (0, 255, 0),
            -1,
        )
    return _draw_title(frame, "MediaPipe Hand", "Hand Pose")


def _draw_depth_panel(depth_map: np.ndarray, fusion_state: FusionState):
    valid = depth_map[np.isfinite(depth_map)]
    if valid.size == 0:
        normalized = np.zeros_like(depth_map, dtype=np.uint8)
    else:
        min_val = float(valid.min())
        max_val = float(valid.max())
        scale = max(max_val - min_val, 1e-6)
        normalized = np.clip((depth_map - min_val) / scale * 255.0, 0, 255).astype(
            np.uint8
        )
    heatmap = cv2.applyColorMap(normalized, cv2.COLORMAP_TURBO)
    if fusion_state.target_center_xy is not None:
        cv2.circle(
            heatmap,
            (
                int(fusion_state.target_center_xy[0]),
                int(fusion_state.target_center_xy[1]),
            ),
            6,
            (255, 255, 255),
            -1,
        )
    if fusion_state.hand_center_xy is not None:
        cv2.circle(
            heatmap,
            (int(fusion_state.hand_center_xy[0]), int(fusion_state.hand_center_xy[1])),
            6,
            (0, 0, 0),
            -1,
        )
    return _draw_title(heatmap, "Depth Anything", "Monocular Depth")


def _draw_camera_panel(
    rgb_image, fusion_state: FusionState, command: str, state_name: str
):
    frame = cv2.cvtColor(rgb_image.copy(), cv2.COLOR_RGB2BGR)
    if fusion_state.target_center_xy and fusion_state.hand_center_xy:
        cv2.line(
            frame,
            (int(fusion_state.hand_center_xy[0]), int(fusion_state.hand_center_xy[1])),
            (
                int(fusion_state.target_center_xy[0]),
                int(fusion_state.target_center_xy[1]),
            ),
            (255, 255, 0),
            2,
        )
    status_lines = [
        f"State: {state_name}",
        f"Command: {command}",
        f"Target visible: {fusion_state.target_visible}",
        f"Hand visible: {fusion_state.hand_visible}",
    ]
    y = TITLE_BAR_HEIGHT + 34
    for line in status_lines:
        cv2.putText(
            frame,
            line,
            (12, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            STATUS_FONT_SCALE,
            (255, 255, 255),
            STATUS_FONT_THICKNESS,
        )
        y += 38
    return _draw_title(frame, "Fusion Runtime", "RGB Camera + Runtime State")


def render_overlay(
    rgb_image,
    detections: Detection | list[Detection] | None,
    depth_map: np.ndarray,
    hand_pose: HandPoseResult,
    fusion_state: FusionState,
    command: str,
    state_name: str,
):
    panel_h, panel_w = PANEL_HEIGHT, PANEL_WIDTH
    camera_panel = _resize_panel(
        _draw_camera_panel(rgb_image, fusion_state, command, state_name),
        panel_w,
        panel_h,
    )
    detection_panel = _resize_panel(
        _draw_detection_panel(rgb_image, detections, fusion_state), panel_w, panel_h
    )
    depth_panel = _resize_panel(
        _draw_depth_panel(depth_map, fusion_state), panel_w, panel_h
    )
    hand_panel = _resize_panel(
        _draw_hand_panel(rgb_image, hand_pose, fusion_state), panel_w, panel_h
    )

    top = cv2.hconcat([camera_panel, detection_panel])
    bottom = cv2.hconcat([depth_panel, hand_panel])
    dashboard = cv2.vconcat([top, bottom])
    return dashboard
