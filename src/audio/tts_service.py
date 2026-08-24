from __future__ import annotations

import shutil
import subprocess
import time


class TtsService:
    def __init__(self, backend: str = "print") -> None:
        self.backend = backend
        self.last_spoken_finished_at = -(10**9)

    def speak(self, text: str) -> None:
        if self.backend == "print":
            print(f"[AUDIO] {text}")
            self.last_spoken_finished_at = time.monotonic()
            return
        if self.backend == "mac_say":
            if shutil.which("say") is None:
                raise RuntimeError("macOS 'say' command is not available.")
            print(f"[AUDIO] {text}")
            subprocess.run(["say", text], check=False)
            self.last_spoken_finished_at = time.monotonic()
            return
        raise RuntimeError(f"Unsupported TTS backend: {self.backend}")

    def time_since_last_speak_ms(self) -> float:
        return max(0.0, (time.monotonic() - self.last_spoken_finished_at) * 1000.0)
