from __future__ import annotations

import numpy as np
from PIL import Image

from src.common.schemas import DepthResult


class DepthAnythingEstimator:
    def __init__(self, model_id: str, device: str = "cpu") -> None:
        self.model_id = model_id
        self.device = device
        self.processor = None
        self.model = None

    def _load(self) -> None:
        if self.model is not None:
            return
        try:
            from transformers import AutoImageProcessor, AutoModelForDepthEstimation

            self.processor = AutoImageProcessor.from_pretrained(self.model_id)
            self.model = AutoModelForDepthEstimation.from_pretrained(self.model_id)
            self.model.to(self.device)
            self.model.eval()
        except Exception as exc:  # pragma: no cover - network/runtime dependent
            raise RuntimeError(
                f"Failed to load Depth Anything model '{self.model_id}'. "
                "Pre-download the model from Hugging Face or check local network connectivity."
            ) from exc

    def predict_depth(self, rgb_image: np.ndarray) -> DepthResult:
        self._load()
        assert self.processor is not None
        assert self.model is not None
        import torch

        image = Image.fromarray(rgb_image)
        inputs = self.processor(images=image, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(self.device)
        with torch.no_grad():
            outputs = self.model(pixel_values=pixel_values)
            predicted_depth = outputs.predicted_depth
        prediction = torch.nn.functional.interpolate(
            predicted_depth.unsqueeze(1),
            size=image.size[::-1],
            mode="bicubic",
            align_corners=False,
        )
        depth_map = prediction.squeeze().cpu().numpy()
        return DepthResult(
            depth_map=depth_map,
            min_depth=float(np.nanmin(depth_map)),
            max_depth=float(np.nanmax(depth_map)),
        )
