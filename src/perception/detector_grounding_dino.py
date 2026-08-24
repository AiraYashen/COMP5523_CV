from __future__ import annotations

from typing import Any

import numpy as np
from PIL import Image

from src.common.schemas import Detection


class GroundingDinoDetector:
    def __init__(self, model_id: str, device: str = "cpu") -> None:
        self.model_id = model_id
        self.device = device
        self.processor = None
        self.model = None

    def _load(self) -> None:
        if self.model is not None:
            return
        try:
            from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

            self.processor = AutoProcessor.from_pretrained(self.model_id)
            self.model = AutoModelForZeroShotObjectDetection.from_pretrained(
                self.model_id
            )
            self.model.to(self.device)
            self.model.eval()
        except Exception as exc:  # pragma: no cover - network/runtime dependent
            raise RuntimeError(
                f"Failed to load Grounding DINO model '{self.model_id}'. "
                "Pre-download the model from Hugging Face or check local network connectivity."
            ) from exc

    def detect(
        self, rgb_image: np.ndarray, prompt: str, threshold: float = 0.15
    ) -> list[Detection]:
        self._load()
        assert self.processor is not None
        assert self.model is not None
        import torch

        image = Image.fromarray(rgb_image)
        normalized_prompt = self._normalize_prompt(prompt)
        inputs = self.processor(
            images=image, text=normalized_prompt, return_tensors="pt"
        )
        inputs = {
            key: value.to(self.device) if hasattr(value, "to") else value
            for key, value in inputs.items()
        }
        with torch.no_grad():
            outputs = self.model(**inputs)
        results = self.processor.post_process_grounded_object_detection(
            outputs,
            inputs["input_ids"],
            threshold=threshold,
            text_threshold=threshold,
            target_sizes=[image.size[::-1]],
        )
        packed: list[Detection] = []
        if not results:
            return packed
        result = results[0]
        boxes = result.get("boxes", [])
        scores = result.get("scores", [])
        labels = result.get("labels", [])
        for box, score, label in zip(boxes, scores, labels):
            x1, y1, x2, y2 = [float(v) for v in box.tolist()]
            if hasattr(score, "detach"):
                score = score.detach().cpu()
            packed.append(
                Detection(
                    bbox_xyxy=(x1, y1, x2, y2), label=str(label), score=float(score)
                )
            )
        return packed

    @staticmethod
    def _normalize_prompt(prompt: str) -> str:
        normalized = prompt.strip().lower()
        if not normalized:
            return normalized
        if not normalized.endswith("."):
            normalized = f"{normalized}."
        return normalized
