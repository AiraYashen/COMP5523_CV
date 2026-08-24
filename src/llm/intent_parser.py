from __future__ import annotations

from dataclasses import asdict
import json
import re
from typing import Any, cast

from src.common.schemas import CommandSpec, SpatialHint


SPATIAL_KEYWORDS = {
    "left": ["left", "左边", "左"],
    "right": ["right", "右边", "右"],
    "front": ["front", "前面", "前方"],
    "center": ["center", "中间", "正中"],
}

SCENE_QUERY_KEYWORDS = [
    "你看到了什么",
    "看到什么",
    "看见什么",
    "能看到什么",
    "前面有什么",
    "有什么东西",
    "有什么内容",
    "什么内容",
    "桌面上有什么",
    "桌上有什么",
    "桌子上有什么",
    "有哪些东西",
    "现在有什么",
    "帮我看看",
    "场景",
    "描述一下",
    "我手在哪里",
    "手在哪里",
    "手在哪",
    "多远",
    "距离",
    "what do you see",
    "what is in front of me",
    "describe the scene",
    "look at",
    "where is my hand",
    "where is the hand",
    "how far",
    "distance",
]

GRASP_KEYWORDS = ["grab", "grasp", "pick", "拿", "抓", "取", "帮我拿"]
GRASP_GUIDANCE_KEYWORDS = [
    "怎么移动",
    "怎么动",
    "如何移动",
    "如何动",
    "怎样移动",
    "怎样动",
    "应该怎么移动",
    "应该怎么动",
    "怎么拿",
    "如何拿",
    "怎样拿",
    "怎么抓",
    "如何抓",
    "怎样抓",
    "怎么够到",
    "如何够到",
    "怎样够到",
    "才能拿到",
    "能拿到",
    "能够拿到",
    "how should i move",
    "how do i move",
    "how should i grab",
]
QUESTION_HINTS = [
    "吗",
    "呢",
    "什么",
    "内容",
    "东西",
    "哪里",
    "哪儿",
    "在哪",
    "几",
    "怎么",
    "如何",
    "怎样",
    "how",
    "what",
    "where",
    "which",
    "distance",
    "far",
]


class IntentParser:
    def __init__(
        self,
        prompt_config: dict[str, Any],
        backend: str = "rule",
        model_id: str | None = None,
        device: str = "cpu",
    ) -> None:
        self.prompt_config = prompt_config
        self.backend = backend
        self.model_id = model_id
        self.device = device
        self.tokenizer = None
        self.model = None

    def _load_qwen(self) -> None:
        if self.model is not None and self.tokenizer is not None:
            return
        from transformers import AutoModelForCausalLM, AutoTokenizer

        model_id = self.model_id or "models/qwen2.5-0.5b-instruct"
        tokenizer = cast(Any, AutoTokenizer.from_pretrained(model_id))
        model = cast(Any, AutoModelForCausalLM.from_pretrained(model_id))
        model.to(self.device)
        self.tokenizer = tokenizer
        self.model = model

    def _extract_target_prompt(self, target_text: str) -> tuple[str, str]:
        lowered = target_text.lower().strip()
        object_map = self.prompt_config.get("object_map", {})
        for _, payload in object_map.items():
            keywords = payload.get("keywords", [])
            if any(
                keyword.lower() in lowered or lowered in keyword.lower()
                for keyword in keywords
            ):
                prompts = payload.get("prompts", [])
                prompt = prompts[0] if prompts else target_text
                return target_text, prompt
        return target_text, ""

    @staticmethod
    def _normalize_text(text: str) -> str:
        cleaned = text.strip()
        cleaned = cleaned.replace("’", "").replace("‘", "")
        cleaned = re.sub(r"\s+", " ", cleaned)
        return cleaned

    @staticmethod
    def _is_low_information_text(text: str) -> bool:
        compact = re.sub(r"[\s\W_]+", "", text, flags=re.UNICODE)
        if len(compact) < 2:
            return True
        if len(compact) >= 8 and len(set(compact)) <= max(2, len(compact) // 6):
            return True
        tokens = [token for token in text.split(" ") if token]
        if len(tokens) >= 4 and len(set(tokens)) <= max(1, len(tokens) // 4):
            return True
        return False

    def _has_target_keyword(self, cleaned: str) -> bool:
        object_map = self.prompt_config.get("object_map", {})
        for payload in object_map.values():
            for keyword in payload.get("keywords", []):
                if keyword.lower() in cleaned:
                    return True
        return False

    def _is_scene_query(self, cleaned: str) -> bool:
        if self._is_low_information_text(cleaned):
            return False
        if any(keyword in cleaned for keyword in SCENE_QUERY_KEYWORDS):
            return True
        if any(keyword in cleaned for keyword in ["看到", "看见", "内容", "东西"]):
            if any(keyword in cleaned for keyword in QUESTION_HINTS):
                return True
        if any(keyword in cleaned for keyword in ["手", "hand", "距离", "多远"]):
            if any(keyword in cleaned for keyword in QUESTION_HINTS):
                return True
        if cleaned.endswith("?") or cleaned.endswith("？"):
            if self._has_target_keyword(cleaned) and any(
                keyword in cleaned for keyword in GRASP_KEYWORDS
            ):
                return False
            return True
        return False

    @staticmethod
    def _is_grasp_guidance_request(cleaned: str, target_prompt_en: str) -> bool:
        if not target_prompt_en:
            return False
        if any(keyword in cleaned for keyword in GRASP_GUIDANCE_KEYWORDS):
            return True
        if any(keyword in cleaned for keyword in ["手", "hand"]) and any(
            keyword in cleaned for keyword in GRASP_KEYWORDS
        ):
            return True
        return False

    def _parse_with_qwen(self, text: str) -> CommandSpec:
        self._load_qwen()
        assert self.tokenizer is not None
        assert self.model is not None

        object_map = self.prompt_config.get("object_map", {})
        object_choices: list[dict[str, Any]] = []
        for object_key, payload in object_map.items():
            object_choices.append(
                {
                    "key": object_key,
                    "keywords": payload.get("keywords", []),
                    "prompt": (payload.get("prompts", [""]) or [""])[0],
                }
            )

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a command parser for a tabletop assistive grasping system. "
                    "Choose the target only from the provided object list. "
                    f"Object list: {json.dumps(object_choices, ensure_ascii=False)}. "
                    "Return compact JSON with keys: action, target_key, spatial_hint. "
                    "Questions asking how to move the hand to reach a specific listed object should use action=grasp. "
                    "Questions asking what is visible, where the hand/object is, or how far something is should use action=scene_query. "
                    "If the user asks an informative scene question but no listed object fits, prefer scene_query over unknown. "
                    "Use action in {grasp, scene_query, stop, restart, unknown}. Use spatial_hint in "
                    "{left, right, front, center, front-left, front-right, front-center, none}. "
                    "Return JSON only."
                ),
            },
            {"role": "user", "content": text},
        ]
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer([prompt], return_tensors="pt")
        inputs = {key: value.to(self.model.device) for key, value in inputs.items()}
        outputs = self.model.generate(**inputs, max_new_tokens=96, do_sample=False)
        generated = outputs[0][inputs["input_ids"].shape[1] :]
        decoded = self.tokenizer.decode(generated, skip_special_tokens=True)
        match = re.search(r"\{.*\}", decoded, re.DOTALL)
        if match is None:
            return self._parse_rule_based(text)
        try:
            payload = json.loads(match.group(0))
        except json.JSONDecodeError:
            return self._parse_rule_based(text)

        action = str(payload.get("action", "unknown")).strip().lower()
        target_key = str(payload.get("target_key", "")).strip()
        spatial_hint_raw = str(payload.get("spatial_hint", "none")).strip().lower()
        spatial_hint = cast(
            SpatialHint, spatial_hint_raw if spatial_hint_raw else "none"
        )
        payload = object_map.get(target_key, {})
        target_keywords = payload.get("keywords", [])
        target_name_raw = target_keywords[0] if target_keywords else target_key
        target_prompts = payload.get("prompts", [])
        target_prompt_en = target_prompts[0] if target_prompts else ""

        if action == "stop":
            return CommandSpec(action="stop", target_name_raw="", target_prompt_en="")
        if action == "restart":
            return CommandSpec(
                action="restart", target_name_raw="", target_prompt_en=""
            )
        if action == "scene_query":
            return CommandSpec(
                action="scene_query",
                target_name_raw="",
                target_prompt_en="",
                query_text=text.strip(),
            )
        if action != "grasp" or not target_prompt_en:
            return CommandSpec(
                action="unknown",
                target_name_raw=target_name_raw or text,
                target_prompt_en="",
                spatial_hint="none",
                need_confirmation=True,
                confirmation_question="请再说一遍。",
            )
        return CommandSpec(
            action="grasp",
            target_name_raw=target_name_raw or target_key,
            target_prompt_en=target_prompt_en,
            spatial_hint=spatial_hint,
        )

    def _parse_rule_based(self, text: str) -> CommandSpec:
        normalized = self._normalize_text(text)
        cleaned = normalized.lower()
        if not cleaned:
            return CommandSpec(
                action="unknown", target_name_raw="", target_prompt_en=""
            )
        if self._is_low_information_text(cleaned):
            return CommandSpec(
                action="unknown",
                target_name_raw=normalized,
                target_prompt_en="",
                spatial_hint="none",
                need_confirmation=True,
                confirmation_question="请再说一遍。",
            )
        if any(keyword in cleaned for keyword in ["stop", "停止"]):
            return CommandSpec(action="stop", target_name_raw="", target_prompt_en="")
        if any(keyword in cleaned for keyword in ["restart", "重新开始", "重来"]):
            return CommandSpec(
                action="restart", target_name_raw="", target_prompt_en=""
            )

        target_name_raw = ""
        target_prompt_en = ""
        spatial_hint: SpatialHint = "none"
        object_map = self.prompt_config.get("object_map", {})
        for _, payload in object_map.items():
            keywords = payload.get("keywords", [])
            for keyword in keywords:
                if keyword.lower() in cleaned:
                    target_name_raw = keyword
                    prompts = payload.get("prompts", [])
                    target_prompt_en = prompts[0] if prompts else keyword
                    break
            if target_prompt_en:
                break

        for hint, keywords in SPATIAL_KEYWORDS.items():
            if any(keyword in cleaned for keyword in keywords):
                spatial_hint = cast(SpatialHint, hint)
                break

        if spatial_hint in {"front", "center"}:
            spatial_hint = cast(
                SpatialHint,
                f"front-{spatial_hint}" if spatial_hint == "center" else spatial_hint,
            )

        if self._is_grasp_guidance_request(cleaned, target_prompt_en):
            return CommandSpec(
                action="grasp",
                target_name_raw=target_name_raw or normalized,
                target_prompt_en=target_prompt_en,
                spatial_hint=spatial_hint,
            )

        if any(keyword in cleaned for keyword in GRASP_GUIDANCE_KEYWORDS):
            return CommandSpec(
                action="scene_query",
                target_name_raw="",
                target_prompt_en="",
                query_text=normalized,
            )

        if self._is_scene_query(cleaned):
            return CommandSpec(
                action="scene_query",
                target_name_raw="",
                target_prompt_en="",
                query_text=normalized,
            )

        if not target_prompt_en:
            return CommandSpec(
                action="unknown",
                target_name_raw=normalized,
                target_prompt_en="",
                query_text="",
                spatial_hint="none",
                need_confirmation=True,
                confirmation_question="请再说一遍。",
            )

        return CommandSpec(
            action="grasp",
            target_name_raw=target_name_raw or normalized,
            target_prompt_en=target_prompt_en,
            spatial_hint=spatial_hint,
        )

    def parse_intent(self, text: str) -> CommandSpec:
        normalized = self._normalize_text(text)
        rule_result = self._parse_rule_based(normalized)
        if rule_result.action != "unknown" or self.backend != "qwen":
            return rule_result
        if self._is_low_information_text(normalized.lower()):
            return rule_result
        qwen_result = self._parse_with_qwen(normalized)
        return qwen_result if qwen_result.action != "unknown" else rule_result

    def translate_text(self, text: str, target_language: str) -> str:
        cleaned_text = text.strip()
        if not cleaned_text or not self.model_id:
            return cleaned_text
        try:
            self._load_qwen()
            assert self.tokenizer is not None
            assert self.model is not None
            messages = [
                {
                    "role": "system",
                    "content": (
                        "You are a precise translator. "
                        f"Translate the user's sentence into {target_language}. "
                        "Return the translated sentence only."
                    ),
                },
                {"role": "user", "content": cleaned_text},
            ]
            prompt = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = self.tokenizer([prompt], return_tensors="pt")
            inputs = {key: value.to(self.model.device) for key, value in inputs.items()}
            outputs = self.model.generate(**inputs, max_new_tokens=96, do_sample=False)
            generated = outputs[0][inputs["input_ids"].shape[1] :]
            decoded = self.tokenizer.decode(generated, skip_special_tokens=True)
        except Exception:
            return cleaned_text
        translated = decoded.replace("Assistant:", "").strip()
        translated = translated.splitlines()[0].strip().strip('"')
        return translated or cleaned_text

    def parse_intent_dict(self, text: str) -> dict[str, Any]:
        return asdict(self.parse_intent(text))
