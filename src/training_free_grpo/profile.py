from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from typing import Any


@dataclass
class ExperienceEntry:
    experience_id: str
    mode: str
    text: str
    score: float = 0.0
    trigger: dict[str, Any] = field(default_factory=dict)


@dataclass
class PromptProfile:
    version: int = 1
    shared_principles: list[str] = field(default_factory=list)
    mode_principles: dict[str, list[str]] = field(default_factory=dict)
    experiences: list[ExperienceEntry] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "PromptProfile":
        experiences = [
            ExperienceEntry(
                experience_id=str(item.get("experience_id", "")),
                mode=str(item.get("mode", "dialogue_turn")),
                text=str(item.get("text", "")).strip(),
                score=float(item.get("score", 0.0)),
                trigger=dict(item.get("trigger", {})),
            )
            for item in payload.get("experiences", [])
            if str(item.get("text", "")).strip()
        ]
        return cls(
            version=int(payload.get("version", 1)),
            shared_principles=[
                str(item).strip()
                for item in payload.get("shared_principles", [])
                if str(item).strip()
            ],
            mode_principles={
                str(key): [str(item).strip() for item in values if str(item).strip()]
                for key, values in dict(payload.get("mode_principles", {})).items()
            },
            experiences=experiences,
            metadata=dict(payload.get("metadata", {})),
        )

    @classmethod
    def from_path(cls, path: str | Path) -> "PromptProfile":
        file_path = Path(path)
        with file_path.open("r", encoding="utf-8") as handle:
            return cls.from_dict(json.load(handle))

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "shared_principles": self.shared_principles,
            "mode_principles": self.mode_principles,
            "experiences": [asdict(item) for item in self.experiences],
            "metadata": self.metadata,
        }

    def save(self, path: str | Path) -> None:
        file_path = Path(path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(
            json.dumps(self.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def merge_experiences(
        self,
        candidates: list[ExperienceEntry],
        max_total: int,
    ) -> "PromptProfile":
        merged: dict[tuple[str, str], ExperienceEntry] = {}
        for item in self.experiences + candidates:
            key = (item.mode, item.text)
            existing = merged.get(key)
            if existing is None or item.score > existing.score:
                merged[key] = item
        ranked = sorted(merged.values(), key=lambda item: item.score, reverse=True)
        self.experiences = ranked[:max_total]
        return self
