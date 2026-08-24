from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


ActionType = Literal["grasp", "scene_query", "stop", "restart", "unknown"]
SpatialHint = Literal[
    "left",
    "right",
    "front",
    "center",
    "front-left",
    "front-right",
    "front-center",
    "none",
]


@dataclass
class CommandSpec:
    action: ActionType
    target_name_raw: str
    target_prompt_en: str
    query_text: str = ""
    spatial_hint: SpatialHint = "none"
    session_mode: str = "capture_and_guide"
    need_confirmation: bool = False
    confirmation_question: str = ""


@dataclass
class FramePacket:
    frame_id: int
    timestamp_ms: int
    rgb_image: Any
    sharpness_score: float = 0.0


@dataclass
class Detection:
    bbox_xyxy: tuple[float, float, float, float]
    label: str
    score: float


@dataclass
class DepthResult:
    depth_map: Any
    min_depth: float
    max_depth: float


@dataclass
class HandPoseResult:
    hand_present: bool
    score: float = 0.0
    handedness: str = "unknown"
    landmarks_xy: list[tuple[float, float]] = field(default_factory=list)


@dataclass
class FusionState:
    target_visible: bool
    hand_visible: bool
    target_locked: bool
    target_center_xy: tuple[float, float] | None
    hand_center_xy: tuple[float, float] | None
    target_depth: float | None
    hand_depth: float | None
    dx_norm: float | None
    dy_norm: float | None
    dz_rel: float | None
    target_confidence: float
    hand_confidence: float
    frame_id: int
    timestamp_ms: int
    frame_size_xy: tuple[int, int] | None = None


@dataclass
class ControlState:
    name: str = "SEARCH_TARGET"
    stable_frames: int = 0
    target_lost_frames: int = 0
    hand_lost_frames: int = 0
    target_visible_frames: int = 0
    hand_visible_frames: int = 0


@dataclass
class GuidanceCommand:
    command: str
    reason: str
    should_play: bool = True


@dataclass
class AsrResult:
    text: str
    confidence: float
    latency_ms: int


@dataclass
class VLMNarration:
    scene_status: str
    scene_description: str
    uncertainty: str = "unknown"
    speak: bool = True
