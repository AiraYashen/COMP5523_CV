from __future__ import annotations

from dataclasses import replace

from src.common.schemas import FusionState


def _ema(
    current: float | None, previous: float | None, alpha: float = 0.6
) -> float | None:
    if current is None:
        return previous
    if previous is None:
        return current
    return alpha * current + (1 - alpha) * previous


def smooth_state(current: FusionState, previous: FusionState | None) -> FusionState:
    if previous is None:
        return current
    return replace(
        current,
        dx_norm=_ema(current.dx_norm, previous.dx_norm),
        dy_norm=_ema(current.dy_norm, previous.dy_norm),
        dz_rel=_ema(current.dz_rel, previous.dz_rel),
    )
