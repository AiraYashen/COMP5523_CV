from __future__ import annotations

from src.audio.tts_service import TtsService


COMMAND_TEXT = {
    "left": "left",
    "right": "right",
    "up": "up",
    "down": "down",
    "forward": "forward",
    "back": "back",
    "hold": "hold",
    "grasp": "grasp",
    "target lost": "target lost",
    "hand lost": "hand lost",
    "place hand in view": "place hand in view",
    "target found": "target found",
    "scan slowly": "scan slowly",
}


class CommandPlayer:
    def __init__(
        self, tts_service: TtsService, debounce_ms: int = 600, enabled: bool = True
    ) -> None:
        self.tts_service = tts_service
        self.debounce_ms = debounce_ms
        self.enabled = enabled
        self.last_command = ""
        self.last_played_ms = -(10**9)

    def play_if_changed(self, command: str, timestamp_ms: int) -> bool:
        if not command:
            return False
        if (
            command == self.last_command
            and timestamp_ms - self.last_played_ms < self.debounce_ms
        ):
            return False
        text = COMMAND_TEXT.get(command, command)
        self.last_command = command
        self.last_played_ms = timestamp_ms
        if not self.enabled:
            return False
        self.tts_service.speak(text)
        return True
