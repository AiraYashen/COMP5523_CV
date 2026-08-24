from src.audio.command_player import CommandPlayer
from src.audio.narration_player import NarrationPlayer
from src.audio.tts_service import TtsService


def test_narration_player_blocks_during_high_priority_guidance() -> None:
    tts = TtsService(backend="print")
    command_player = CommandPlayer(tts, debounce_ms=600)
    narration_player = NarrationPlayer(tts, cooldown_ms=5000, priority_window_ms=1500)
    command_player.last_command = "right"
    command_player.last_played_ms = 1000
    assert (
        narration_player.can_play(
            "The can is on the table.",
            timestamp_ms=2000,
            rule_command="right",
            command_player=command_player,
        )
        is False
    )


def test_narration_player_allows_non_urgent_status_narration() -> None:
    tts = TtsService(backend="print")
    command_player = CommandPlayer(tts, debounce_ms=600)
    narration_player = NarrationPlayer(tts, cooldown_ms=5000, priority_window_ms=1500)
    command_player.last_command = "target found"
    command_player.last_played_ms = 1000
    assert (
        narration_player.can_play(
            "The target is in front of you.",
            timestamp_ms=2000,
            rule_command="place hand in view",
            command_player=command_player,
        )
        is True
    )


def test_narration_player_allows_high_priority_guidance_when_raw_commands_muted() -> (
    None
):
    tts = TtsService(backend="print")
    command_player = CommandPlayer(tts, debounce_ms=600, enabled=False)
    narration_player = NarrationPlayer(tts, cooldown_ms=5000, priority_window_ms=1500)
    assert (
        narration_player.can_play(
            "请把手向右移动一点。",
            timestamp_ms=2000,
            rule_command="right",
            command_player=command_player,
        )
        is True
    )
