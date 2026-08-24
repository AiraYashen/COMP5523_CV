from __future__ import annotations

import base64
import io
import json
import re
from urllib import error as urllib_error
from urllib import request as urllib_request
from typing import Any, cast

import numpy as np
from PIL import Image

from src.common.schemas import VLMNarration


class VLMService:
    def __init__(
        self,
        enabled: bool,
        model_id: str,
        backend: str = "transformers",
        device: str = "cpu",
        max_new_tokens: int = 96,
        api_key: str = "",
        api_base_url: str = "",
        timeout_s: int = 45,
        thinking_enabled: bool = False,
    ) -> None:
        self.enabled = enabled
        self.model_id = model_id
        self.backend = backend
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.api_key = api_key
        self.api_base_url = api_base_url
        self.timeout_s = timeout_s
        self.thinking_enabled = thinking_enabled
        self.processor = None
        self.model = None

    def _default_api_headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "Accept": "*/*",
            "Connection": "keep-alive",
            "User-Agent": (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/123.0.0.0 Safari/537.36"
            ),
        }

    def _load(self) -> None:
        if not self.enabled or self.model is not None or self.backend != "transformers":
            return
        from transformers import AutoModelForImageTextToText, AutoProcessor

        self.processor = cast(Any, AutoProcessor.from_pretrained(self.model_id))
        self.model = cast(
            Any, AutoModelForImageTextToText.from_pretrained(self.model_id)
        )
        self.model.to(self.device)
        self.model.eval()

    @staticmethod
    def _extract_message_content(message: dict[str, Any]) -> str:
        content = message.get("content", "")
        if isinstance(content, str):
            return content.strip()
        if isinstance(content, list):
            text_parts: list[str] = []
            for item in content:
                if isinstance(item, str):
                    text_parts.append(item)
                    continue
                if isinstance(item, dict):
                    text_value = item.get("text") or item.get("content") or ""
                    if text_value:
                        text_parts.append(str(text_value))
            return "\n".join(
                part.strip() for part in text_parts if part.strip()
            ).strip()
        if isinstance(content, dict):
            text_value = content.get("text") or content.get("content") or ""
            return str(text_value).strip()
        return str(content).strip()

    @staticmethod
    def _clean_plain_text(decoded: str) -> str:
        return decoded.replace("Assistant:", "").strip()

    @staticmethod
    def _parse_response(decoded: str) -> VLMNarration:
        decoded = decoded.replace("Assistant:", "").strip()
        match = re.search(r"\{.*\}", decoded, re.DOTALL)
        if match is None:
            cleaned = VLMService._clean_plain_text(decoded)
            return VLMNarration(
                scene_status="unknown",
                scene_description=cleaned,
                uncertainty="high",
                speak=bool(cleaned),
            )
        try:
            payload = json.loads(match.group(0))
        except json.JSONDecodeError:
            cleaned = VLMService._clean_plain_text(decoded)
            return VLMNarration(
                scene_status="unknown",
                scene_description=cleaned,
                uncertainty="high",
                speak=bool(cleaned),
            )
        return VLMNarration(
            scene_status=str(payload.get("scene_status", "unknown")),
            scene_description=str(payload.get("scene_description", "")).strip(),
            uncertainty=str(payload.get("uncertainty", "unknown")),
            speak=bool(payload.get("speak", True)),
        )

    @staticmethod
    def _to_png_data_url(image: np.ndarray) -> str:
        pil_image = Image.fromarray(image)
        buffer = io.BytesIO()
        pil_image.save(buffer, format="PNG")
        encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
        return f"data:image/png;base64,{encoded}"

    @staticmethod
    def _depth_map_to_rgb(depth_map: np.ndarray) -> np.ndarray:
        valid = depth_map[np.isfinite(depth_map)]
        if valid.size == 0:
            normalized = np.zeros_like(depth_map, dtype=np.uint8)
        else:
            min_val = float(valid.min())
            max_val = float(valid.max())
            scale = max(max_val - min_val, 1e-6)
            normalized = np.clip((depth_map - min_val) / scale * 255.0, 0, 255).astype(
                np.uint8
            )
        return np.stack([normalized, normalized, normalized], axis=-1)

    def _build_glm_payload(
        self,
        rgb_image: np.ndarray,
        depth_map: np.ndarray,
        prompt: str,
        temperature: float | None = None,
    ) -> dict[str, Any]:
        content: list[dict[str, Any]] = [
            {
                "type": "image_url",
                "image_url": {"url": self._to_png_data_url(rgb_image)},
            },
            {
                "type": "image_url",
                "image_url": {
                    "url": self._to_png_data_url(self._depth_map_to_rgb(depth_map))
                },
            },
            {"type": "text", "text": prompt},
        ]
        payload: dict[str, Any] = {
            "model": self.model_id,
            "messages": [{"role": "user", "content": content}],
            "max_tokens": self.max_new_tokens,
        }
        if temperature is not None:
            payload["temperature"] = temperature
        if self.thinking_enabled:
            payload["thinking"] = {"type": "enabled"}
        else:
            payload["thinking"] = {"type": "disabled"}
        return payload

    def _build_responses_payload(
        self,
        rgb_image: np.ndarray,
        depth_map: np.ndarray,
        prompt: str,
        temperature: float | None = None,
    ) -> dict[str, Any]:
        content: list[dict[str, Any]] = [
            {
                "type": "input_image",
                "image_url": self._to_png_data_url(rgb_image),
            },
            {
                "type": "input_image",
                "image_url": self._to_png_data_url(self._depth_map_to_rgb(depth_map)),
            },
            {
                "type": "input_text",
                "text": prompt,
            },
        ]
        payload: dict[str, Any] = {
            "model": self.model_id,
            "input": [
                {
                    "type": "message",
                    "role": "user",
                    "content": content,
                }
            ],
            "max_output_tokens": self.max_new_tokens,
        }
        if temperature is not None:
            payload["temperature"] = temperature
        return payload

    def _analyze_with_glm(
        self,
        rgb_image: np.ndarray,
        depth_map: np.ndarray,
        prompt: str,
        temperature: float | None = None,
    ) -> VLMNarration:
        if not self.api_key:
            return VLMNarration(
                scene_status="missing_api_key",
                scene_description="",
                uncertainty="high",
                speak=False,
            )
        payload = self._build_glm_payload(rgb_image, depth_map, prompt, temperature)
        request = urllib_request.Request(
            self.api_base_url,
            data=json.dumps(payload).encode("utf-8"),
            headers=self._default_api_headers(),
            method="POST",
        )
        try:
            with urllib_request.urlopen(request, timeout=self.timeout_s) as response:
                raw_payload = json.loads(response.read().decode("utf-8"))
        except urllib_error.URLError as exc:
            raise RuntimeError(f"GLM request failed: {exc}") from exc
        choices = raw_payload.get("choices", [])
        if not choices:
            raise RuntimeError("GLM response did not contain choices")
        message = choices[0].get("message", {})
        decoded = self._extract_message_content(message)
        return self._parse_response(decoded)

    @staticmethod
    def _extract_responses_output_text(raw_payload: dict[str, Any]) -> str:
        output_text = raw_payload.get("output_text")
        if isinstance(output_text, str) and output_text.strip():
            return output_text.strip()

        output = raw_payload.get("output", [])
        if not isinstance(output, list):
            return ""
        text_parts: list[str] = []
        for item in output:
            if not isinstance(item, dict):
                continue
            content = item.get("content", [])
            if not isinstance(content, list):
                continue
            for block in content:
                if not isinstance(block, dict):
                    continue
                if block.get("type") == "output_text":
                    text_value = block.get("text", "")
                    if text_value:
                        text_parts.append(str(text_value).strip())
        return "\n".join(part for part in text_parts if part).strip()

    def _analyze_with_responses_api(
        self,
        rgb_image: np.ndarray,
        depth_map: np.ndarray,
        prompt: str,
        temperature: float | None = None,
    ) -> VLMNarration:
        if not self.api_key:
            return VLMNarration(
                scene_status="missing_api_key",
                scene_description="",
                uncertainty="high",
                speak=False,
            )
        payload = self._build_responses_payload(
            rgb_image, depth_map, prompt, temperature
        )
        request = urllib_request.Request(
            self.api_base_url,
            data=json.dumps(payload).encode("utf-8"),
            headers=self._default_api_headers(),
            method="POST",
        )
        try:
            with urllib_request.urlopen(request, timeout=self.timeout_s) as response:
                raw_payload = json.loads(response.read().decode("utf-8"))
        except urllib_error.HTTPError as exc:
            details = exc.read().decode("utf-8", errors="ignore")
            raise RuntimeError(f"Responses API request failed: {exc.code} {details}") from exc
        except urllib_error.URLError as exc:
            raise RuntimeError(f"Responses API request failed: {exc}") from exc
        decoded = self._extract_responses_output_text(raw_payload)
        if not decoded:
            raise RuntimeError("Responses API response did not contain output text")
        return self._parse_response(decoded)

    def analyze_multimodal(
        self,
        rgb_image: np.ndarray,
        depth_map: np.ndarray,
        prompt: str,
        temperature: float | None = None,
    ) -> VLMNarration:
        if not self.enabled:
            return VLMNarration(
                scene_status="disabled",
                scene_description="",
                uncertainty="unknown",
                speak=False,
            )
        if self.backend == "glm_api":
            return self._analyze_with_glm(rgb_image, depth_map, prompt, temperature)
        if self.backend == "responses_api":
            return self._analyze_with_responses_api(
                rgb_image, depth_map, prompt, temperature
            )
        dashboard = rgb_image
        return self.analyze_dashboard(dashboard, prompt, temperature)

    def analyze_dashboard(
        self,
        dashboard_bgr: np.ndarray,
        prompt: str,
        temperature: float | None = None,
    ) -> VLMNarration:
        if not self.enabled:
            return VLMNarration(
                scene_status="disabled",
                scene_description="",
                uncertainty="unknown",
                speak=False,
            )
        self._load()
        assert self.processor is not None
        assert self.model is not None
        import torch

        image = Image.fromarray(dashboard_bgr[:, :, ::-1])
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        chat_prompt = self.processor.apply_chat_template(
            messages, add_generation_prompt=True
        )
        inputs = self.processor(text=chat_prompt, images=[image], return_tensors="pt")
        inputs = {
            key: value.to(self.device) if hasattr(value, "to") else value
            for key, value in inputs.items()
        }
        generation_kwargs: dict[str, Any] = {"max_new_tokens": self.max_new_tokens}
        if temperature is not None and temperature > 0:
            generation_kwargs["do_sample"] = True
            generation_kwargs["temperature"] = temperature
        else:
            generation_kwargs["do_sample"] = False
        with torch.no_grad():
            outputs = self.model.generate(**inputs, **generation_kwargs)
        generated = outputs[0][inputs["input_ids"].shape[1] :]
        decoded = self.processor.decode(generated, skip_special_tokens=True)
        return self._parse_response(decoded)
