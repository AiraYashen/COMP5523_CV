from __future__ import annotations

from collections.abc import Mapping

from src.common.schemas import FusionState, GuidanceCommand


def enforce_safety(
    command: GuidanceCommand,
    fusion_state: FusionState,
    cfg: Mapping[str, float | int],
) -> GuidanceCommand:
    _ = cfg
    if not fusion_state.target_visible:
        return GuidanceCommand(command="target lost", reason="target missing")
    if not fusion_state.hand_visible and command.command not in {
        "place hand in view",
        "target lost",
    }:
        return GuidanceCommand(command="hand lost", reason="hand missing")
    return command
