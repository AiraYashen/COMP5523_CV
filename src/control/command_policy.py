from __future__ import annotations

from collections.abc import Mapping

from src.common.schemas import ControlState, FusionState, GuidanceCommand


def decide_command(
    control_state: ControlState,
    fusion_state: FusionState,
    cfg: Mapping[str, float | int],
) -> GuidanceCommand:
    if control_state.name == "SEARCH_TARGET":
        return GuidanceCommand(command="target lost", reason="target not visible")
    if control_state.name == "WAIT_HAND":
        return GuidanceCommand(command="place hand in view", reason="hand not visible")
    if (
        control_state.name == "GRASP_READY"
        and control_state.stable_frames >= cfg["grasp_stable_frames"]
    ):
        return GuidanceCommand(command="grasp", reason="stable close-range alignment")

    dx = fusion_state.dx_norm or 0.0
    if bool(cfg.get("invert_x_guidance", False)):
        dx = -dx
    dy = fusion_state.dy_norm or 0.0
    dz = fusion_state.dz_rel or 0.0
    magnitudes = {
        "x": abs(dx),
        "y": abs(dy),
        "z": abs(dz),
    }
    dominant = max(magnitudes.items(), key=lambda item: item[1])[0]

    if dominant == "x" and abs(dx) > cfg["deadband_x"]:
        return GuidanceCommand(
            command="right" if dx > 0 else "left", reason="largest lateral error"
        )
    if dominant == "y" and abs(dy) > cfg["deadband_y"]:
        return GuidanceCommand(
            command="down" if dy > 0 else "up", reason="largest vertical error"
        )
    # Only allow depth-only commands after lateral alignment is reasonably close,
    # otherwise noisy monocular depth tends to over-trigger `back`.
    lateral_ready = abs(dx) < cfg["align_x"] * 1.5 and abs(dy) < cfg["align_y"] * 1.5
    if dominant == "z" and lateral_ready and abs(dz) > cfg["grasp_z"] * 1.5:
        return GuidanceCommand(
            command="forward" if dz > 0 else "back", reason="largest depth error"
        )
    return GuidanceCommand(command="hold", reason="within deadband")
