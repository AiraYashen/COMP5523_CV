from __future__ import annotations

from statistics import mean


def target_acquisition_rate(found_flags: list[bool]) -> float:
    return sum(found_flags) / len(found_flags) if found_flags else 0.0


def guidance_success_rate(success_flags: list[bool]) -> float:
    return sum(success_flags) / len(success_flags) if success_flags else 0.0


def average_latency(latencies_ms: list[int]) -> float:
    return mean(latencies_ms) if latencies_ms else 0.0
