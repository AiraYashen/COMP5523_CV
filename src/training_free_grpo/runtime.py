from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

from src.common.config import load_training_free_grpo_config
from src.training_free_grpo.profile import PromptProfile


TRAINING_FREE_PREFIX_START = "[Training-Free Prompt Prior]"
TRAINING_FREE_PREFIX_END = "[/Training-Free Prompt Prior]"

_SCENE_OVERVIEW_KEYWORDS = [
    "前面有什么",
    "看到了什么",
    "看到什么",
    "看见什么",
    "桌面上有什么",
    "桌上有什么",
    "桌子上有什么",
    "有什么东西",
    "what do you see",
    "describe the scene",
]
_MOTION_KEYWORDS = [
    "怎么移动",
    "怎么动",
    "如何移动",
    "怎样移动",
    "怎么拿",
    "如何拿",
    "怎样拿",
    "怎么抓",
    "如何抓",
    "怎样抓",
    "how should i move",
    "how do i move",
    "how to grab",
]
_DISTANCE_KEYWORDS = ["多远", "距离", "多近", "how far", "distance", "how close"]
_CONFIRMATION_KEYWORDS = [
    "是不是",
    "是吗",
    "对吗",
    "抓住的是",
    "拿着的是",
    "抓到的是",
    "am i holding",
    "did i grab",
    "is this",
]


def prefers_chinese(text: str) -> bool:
    return any("\u4e00" <= char <= "\u9fff" for char in text)


def infer_question_type(text: str) -> str:
    lowered = text.lower().strip()
    if any(keyword in text for keyword in _CONFIRMATION_KEYWORDS) or any(
        keyword in lowered for keyword in _CONFIRMATION_KEYWORDS
    ):
        return "confirmation"
    if any(keyword in text for keyword in _DISTANCE_KEYWORDS) or any(
        keyword in lowered for keyword in _DISTANCE_KEYWORDS
    ):
        return "distance"
    if any(keyword in text for keyword in _MOTION_KEYWORDS) or any(
        keyword in lowered for keyword in _MOTION_KEYWORDS
    ):
        return "motion"
    if any(keyword in text for keyword in _SCENE_OVERVIEW_KEYWORDS) or any(
        keyword in lowered for keyword in _SCENE_OVERVIEW_KEYWORDS
    ):
        return "scene_overview"
    return "dialogue"


def strip_training_free_prefix(prompt: str) -> str:
    if TRAINING_FREE_PREFIX_START not in prompt or TRAINING_FREE_PREFIX_END not in prompt:
        return prompt
    start = prompt.find(TRAINING_FREE_PREFIX_START)
    end = prompt.find(TRAINING_FREE_PREFIX_END)
    if start < 0 or end < 0 or end < start:
        return prompt
    return (prompt[:start] + prompt[end + len(TRAINING_FREE_PREFIX_END) :]).strip()


def enrich_prompt_context(context: dict[str, Any] | None) -> dict[str, Any]:
    payload = deepcopy(context or {})
    text = str(
        payload.get("question")
        or payload.get("user_text")
        or payload.get("target_name")
        or ""
    )
    fusion = dict(payload.get("fusion_payload") or payload.get("fusion") or {})
    hand_pose = dict(payload.get("hand_pose_payload") or payload.get("hand_pose") or {})
    payload["question_text"] = text
    payload["question_type"] = infer_question_type(text)
    payload["requires_chinese"] = bool(
        payload.get("requires_chinese", prefers_chinese(text))
    )
    payload["hand_present"] = bool(
        hand_pose.get("hand_present", payload.get("hand_present", False))
    )
    payload["target_visible"] = bool(
        fusion.get("target_visible", payload.get("target_visible", False))
    )
    dx = fusion.get("dx_norm")
    dy = fusion.get("dy_norm")
    dz = fusion.get("dz_rel")
    payload["dx_sign"] = (
        "left" if isinstance(dx, (int, float)) and dx < -0.05 else
        "right" if isinstance(dx, (int, float)) and dx > 0.05 else
        "center"
    )
    payload["dy_sign"] = (
        "up" if isinstance(dy, (int, float)) and dy < -0.05 else
        "down" if isinstance(dy, (int, float)) and dy > 0.05 else
        "center"
    )
    payload["dz_sign"] = (
        "far" if isinstance(dz, (int, float)) and dz > 0.08 else
        "near" if isinstance(dz, (int, float)) and dz < -0.08 else
        "aligned"
    )
    payload["guidance_command"] = str(
        payload.get("guidance_command") or payload.get("rule_command") or ""
    )
    return payload


class TrainingFreePromptAdapter:
    def __init__(
        self,
        config: dict[str, Any] | None = None,
        profile: PromptProfile | None = None,
    ) -> None:
        root = config or load_training_free_grpo_config()
        self.config = dict(root.get("training_free_grpo", root))
        self._profile = profile

    @property
    def enabled(self) -> bool:
        return bool(self.config.get("enabled", True))

    def load_profile(self) -> PromptProfile:
        if self._profile is not None:
            return self._profile
        profile_path = Path(
            str(self.config.get("profile_path", "outputs/training_free_grpo/latest_profile.json"))
        )
        if profile_path.exists():
            self._profile = PromptProfile.from_path(profile_path)
            return self._profile
        self._profile = PromptProfile(
            shared_principles=[
                str(item).strip()
                for item in self.config.get("shared_principles", [])
                if str(item).strip()
            ],
            mode_principles={
                str(key): [str(item).strip() for item in values if str(item).strip()]
                for key, values in dict(self.config.get("mode_principles", {})).items()
            },
            metadata={"source": "config_defaults"},
        )
        return self._profile

    def set_profile(self, profile: PromptProfile) -> None:
        self._profile = profile

    def wrap_prompt(
        self,
        base_prompt: str,
        mode: str,
        context: dict[str, Any] | None = None,
        profile: PromptProfile | None = None,
    ) -> str:
        prefix = self.render_prefix(mode=mode, context=context, profile=profile)
        if not prefix:
            return strip_training_free_prefix(base_prompt)
        cleaned_prompt = strip_training_free_prefix(base_prompt)
        return f"{prefix}\n\n{cleaned_prompt}"

    def render_prefix(
        self,
        mode: str,
        context: dict[str, Any] | None = None,
        profile: PromptProfile | None = None,
    ) -> str:
        if not self.enabled:
            return ""
        payload = enrich_prompt_context(context)
        active_profile = profile or self.load_profile()
        sections: list[str] = [TRAINING_FREE_PREFIX_START]
        shared = active_profile.shared_principles[: int(self.config.get("max_shared_principles", 5))]
        if shared:
            sections.append("共享先验：")
            for index, item in enumerate(shared, start=1):
                sections.append(f"{index}. {item}")
        mode_rules = active_profile.mode_principles.get(mode, [])
        if mode_rules:
            sections.append("模式经验：")
            for index, item in enumerate(mode_rules, start=1):
                sections.append(f"{index}. {item}")
        dynamic_cues = self._build_dynamic_cues(mode=mode, context=payload)
        if dynamic_cues:
            sections.append("当前轮关注点：")
            for index, item in enumerate(dynamic_cues, start=1):
                sections.append(f"{index}. {item}")
        experiences = self._select_experiences(mode=mode, context=payload, profile=active_profile)
        if experiences:
            sections.append("检索经验：")
            for index, item in enumerate(experiences, start=1):
                sections.append(f"{index}. {item}")
        sections.append(TRAINING_FREE_PREFIX_END)
        return "\n".join(sections)

    def _select_experiences(
        self,
        mode: str,
        context: dict[str, Any],
        profile: PromptProfile,
    ) -> list[str]:
        selected: list[str] = []
        max_active = int(self.config.get("max_active_experiences", 4))
        ranked = sorted(profile.experiences, key=lambda item: item.score, reverse=True)
        for item in ranked:
            if item.mode not in {mode, "shared"}:
                continue
            if not self._matches_trigger(item.trigger, context):
                continue
            selected.append(item.text)
            if len(selected) >= max_active:
                break
        return selected

    @staticmethod
    def _matches_trigger(trigger: dict[str, Any], context: dict[str, Any]) -> bool:
        for key, expected in trigger.items():
            actual = context.get(key)
            if isinstance(expected, list):
                if actual not in expected:
                    return False
                continue
            if actual != expected:
                return False
        return True

    @staticmethod
    def _build_dynamic_cues(mode: str, context: dict[str, Any]) -> list[str]:
        cues: list[str] = []
        question_type = str(context.get("question_type", "dialogue"))
        if context.get("requires_chinese"):
            cues.append("最终回答只用简体中文，适合直接语音播报，不夹杂英文控制词。")
        cues.append("左右和几点钟方向都以用户自身身体朝向为参考，不以屏幕画面或镜像预览为参考。")
        if not context.get("hand_present"):
            cues.append("如果当前没有看到手，先说明手未进入画面，不要继续做精细对齐判断。")
        if context.get("dx_sign") == "left":
            cues.append("如果需要描述水平方向，要明确说目标在用户或手的左侧。")
        if context.get("dx_sign") == "right":
            cues.append("如果需要描述水平方向，要明确说目标在用户或手的右侧。")
        if context.get("dy_sign") == "up":
            cues.append("如果需要补充竖直关系，明确说目标比手更高。")
        if context.get("dy_sign") == "down":
            cues.append("如果需要补充竖直关系，明确说目标比手更低。")
        if context.get("dz_sign") == "far":
            cues.append("如果深度显示目标更远，应提示用户向前靠近，而不是只重复左右修正。")
        if context.get("dz_sign") == "near":
            cues.append("如果深度显示手已经略微过前，应提示用户小幅回手。")
        if question_type == "scene_overview":
            cues.append("如果用户问当前场景，优先概括2到4个稳定可见且与操作最相关的物体。")
        if question_type == "motion":
            cues.append("如果用户问怎么拿或怎么移动，优先回答方向、距离和当前唯一最关键的一步动作。")
        if question_type == "distance":
            cues.append("如果用户问距离或远近，先直接回答相对距离，再补充必要说明。")
        if question_type == "confirmation":
            cues.append("如果用户问是否已经抓住，先判断当前画面里的实际接触关系；手已明显包住、压住或捏住目标时可以直接肯定。")
        if mode == "grasp_guidance":
            cues.append("把系统状态翻译成专业、自然、短句的辅助指令，不要暴露底层控制标签。")
        return cues[:5]
