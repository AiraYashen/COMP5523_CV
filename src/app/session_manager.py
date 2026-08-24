from __future__ import annotations

from dataclasses import dataclass

from src.common.schemas import CommandSpec


@dataclass
class Session:
    command: CommandSpec
    active: bool = True
    success: bool = False
    steps_run: int = 0


@dataclass
class DialogueTurn:
    role: str
    text: str


class SessionManager:
    def __init__(self) -> None:
        self.current: Session | None = None
        self.dialogue_history: list[DialogueTurn] = []

    def start(self, command: CommandSpec) -> Session:
        self.current = Session(command=command)
        return self.current

    def stop(self, success: bool = False) -> None:
        if self.current is None:
            return
        self.current.active = False
        self.current.success = success

    def append_turn(self, role: str, text: str) -> None:
        cleaned = text.strip()
        if not cleaned:
            return
        self.dialogue_history.append(DialogueTurn(role=role, text=cleaned))

    def render_history(self, max_turns: int = 6) -> str:
        turns = self.dialogue_history[-max_turns:]
        if not turns:
            return "无"
        rendered: list[str] = []
        for turn in turns:
            speaker = "用户" if turn.role == "user" else "助手"
            rendered.append(f"{speaker}: {turn.text}")
        return "\n".join(rendered)

    def clear_history(self) -> None:
        self.dialogue_history.clear()
