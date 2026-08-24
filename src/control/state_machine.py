from __future__ import annotations

from collections.abc import Mapping

from src.common.schemas import ControlState, FusionState


def next_state(
    prev_state: ControlState | None,
    fusion_state: FusionState,
    cfg: Mapping[str, float | int],
) -> ControlState:
    state = prev_state or ControlState()
    target_found_frames = int(cfg.get("target_found_frames", 1))
    hand_found_frames = int(cfg.get("hand_found_frames", 1))
    target_lost_limit = int(cfg.get("target_lost_frames", 1))
    hand_lost_limit = int(cfg.get("hand_lost_frames", 1))
    target_visible_frames = (
        state.target_visible_frames + 1 if fusion_state.target_visible else 0
    )
    hand_visible_frames = state.hand_visible_frames + (
        1 if fusion_state.hand_visible else 0
    )
    target_lost_frames = state.target_lost_frames + (
        0 if fusion_state.target_visible else 1
    )
    hand_lost_frames = state.hand_lost_frames + (0 if fusion_state.hand_visible else 1)
    stable_frames = state.stable_frames + 1

    if not fusion_state.target_visible:
        if state.name != "SEARCH_TARGET" and target_lost_frames < target_lost_limit:
            return ControlState(
                name=state.name,
                stable_frames=0,
                target_lost_frames=target_lost_frames,
                hand_lost_frames=hand_lost_frames,
                target_visible_frames=0,
                hand_visible_frames=hand_visible_frames,
            )
        return ControlState(
            name="SEARCH_TARGET",
            stable_frames=0,
            target_lost_frames=target_lost_frames,
            hand_lost_frames=hand_lost_frames,
            target_visible_frames=0,
            hand_visible_frames=hand_visible_frames,
        )
    if (
        prev_state is not None
        and state.name == "SEARCH_TARGET"
        and target_visible_frames < target_found_frames
    ):
        return ControlState(
            name="SEARCH_TARGET",
            stable_frames=0,
            target_visible_frames=target_visible_frames,
            hand_visible_frames=hand_visible_frames,
        )
    if fusion_state.target_visible and not fusion_state.hand_visible:
        if (
            state.name not in {"SEARCH_TARGET", "WAIT_HAND"}
            and hand_lost_frames < hand_lost_limit
        ):
            return ControlState(
                name=state.name,
                stable_frames=0,
                target_lost_frames=target_lost_frames,
                hand_lost_frames=hand_lost_frames,
                target_visible_frames=target_visible_frames,
                hand_visible_frames=0,
            )
        return ControlState(
            name="WAIT_HAND",
            stable_frames=0,
            target_lost_frames=target_lost_frames,
            hand_lost_frames=hand_lost_frames,
            target_visible_frames=target_visible_frames,
            hand_visible_frames=0,
        )
    if (
        prev_state is not None
        and state.name == "WAIT_HAND"
        and hand_visible_frames < hand_found_frames
    ):
        return ControlState(
            name="WAIT_HAND",
            stable_frames=0,
            target_visible_frames=target_visible_frames,
            hand_visible_frames=hand_visible_frames,
        )

    dx = abs(fusion_state.dx_norm or 0.0)
    dy = abs(fusion_state.dy_norm or 0.0)
    dz = abs(fusion_state.dz_rel or 0.0)

    if dx < cfg["grasp_x"] and dy < cfg["grasp_y"] and dz < cfg["grasp_z"]:
        stable_frames = state.stable_frames + 1 if state.name == "GRASP_READY" else 1
        return ControlState(
            name="GRASP_READY",
            stable_frames=stable_frames,
            target_visible_frames=target_visible_frames,
            hand_visible_frames=hand_visible_frames,
        )
    if dx < cfg["align_x"] and dy < cfg["align_y"]:
        return ControlState(
            name="ALIGN",
            stable_frames=stable_frames,
            target_visible_frames=target_visible_frames,
            hand_visible_frames=hand_visible_frames,
        )
    return ControlState(
        name="APPROACH",
        stable_frames=stable_frames,
        target_visible_frames=target_visible_frames,
        hand_visible_frames=hand_visible_frames,
    )
