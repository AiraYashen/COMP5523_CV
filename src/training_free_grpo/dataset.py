from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from typing import Any

from src.training_free_grpo.runtime import strip_training_free_prefix


@dataclass
class TrainingFreeSample:
    sample_id: str
    mode: str
    prompt: str
    rgb_image_path: str
    depth_image_path: str
    prompt_payload: dict[str, Any] = field(default_factory=dict)
    reference_answer: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_training_samples(path: str | Path) -> list[TrainingFreeSample]:
    file_path = Path(path)
    raw_text = file_path.read_text(encoding="utf-8").strip()
    if not raw_text:
        return []
    if file_path.suffix.lower() == ".jsonl":
        payloads = [json.loads(line) for line in raw_text.splitlines() if line.strip()]
    else:
        parsed = json.loads(raw_text)
        payloads = parsed if isinstance(parsed, list) else [parsed]
    samples: list[TrainingFreeSample] = []
    for payload in payloads:
        samples.append(
            TrainingFreeSample(
                sample_id=str(payload.get("sample_id", "")),
                mode=str(payload.get("mode", "dialogue_turn")),
                prompt=strip_training_free_prefix(str(payload.get("prompt", ""))),
                rgb_image_path=str(payload.get("rgb_image_path", "")),
                depth_image_path=str(payload.get("depth_image_path", "")),
                prompt_payload=dict(payload.get("prompt_payload", {})),
                reference_answer=str(payload.get("reference_answer", "")),
                metadata=dict(payload.get("metadata", {})),
            )
        )
    return samples


def export_seed_dataset_from_logs(
    log_dir: str | Path,
    output_path: str | Path,
) -> list[TrainingFreeSample]:
    source_dir = Path(log_dir)
    output_file = Path(output_path)
    samples: list[TrainingFreeSample] = []
    for metadata_path in sorted(source_dir.glob("*.json")):
        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
        sample = TrainingFreeSample(
            sample_id=metadata_path.stem,
            mode=str(payload.get("mode", "dialogue_turn")),
            prompt=strip_training_free_prefix(str(payload.get("prompt", ""))),
            rgb_image_path=str(payload.get("rgb_image", "")),
            depth_image_path=str(payload.get("depth_image", "")),
            prompt_payload=dict(payload.get("prompt_payload", {})),
            reference_answer=str(payload.get("reference_answer", "")),
            metadata={
                "event": payload.get("event", ""),
                "timestamp_ms": payload.get("timestamp_ms", 0),
                "model_id": payload.get("model_id", ""),
                "backend": payload.get("backend", ""),
            },
        )
        samples.append(sample)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w", encoding="utf-8") as handle:
        for sample in samples:
            handle.write(json.dumps(sample.to_dict(), ensure_ascii=False) + "\n")
    return samples
