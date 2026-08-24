from __future__ import annotations

from src.audio.command_player import CommandPlayer
from src.audio.tts_service import TtsService


HIGH_PRIORITY_COMMANDS = {
    "left",
    "right",
    "up",
    "down",
    "forward",
    "back",
    "grasp",
}


class NarrationPlayer:
    def __init__(
        self,
        tts_service: TtsService,
        cooldown_ms: int = 5000,
        priority_window_ms: int = 1500,
    ) -> None:
        self.tts_service = tts_service
        self.cooldown_ms = cooldown_ms
        self.priority_window_ms = priority_window_ms
        self.last_spoken_text = ""
        self.last_spoken_ms = -(10**9)

    def can_play(
        self,
        text: str,
        timestamp_ms: int,
        rule_command: str,
        command_player: CommandPlayer,
    ) -> bool:
        if not text.strip():
            return False
        if timestamp_ms - self.last_spoken_ms < self.cooldown_ms:
            return False
        if command_player.enabled and rule_command in HIGH_PRIORITY_COMMANDS:
            return False
        if (
            command_player.enabled
            and command_player.last_command in HIGH_PRIORITY_COMMANDS
            and timestamp_ms - command_player.last_played_ms < self.priority_window_ms
        ):
            return False
        return True

    def play(self, text: str, timestamp_ms: int) -> bool:
        if not text.strip():
            return False
        self.tts_service.speak(text)
        self.last_spoken_text = text
        self.last_spoken_ms = timestamp_ms
        return True
