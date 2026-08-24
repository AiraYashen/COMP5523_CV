from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Any

from src.training_free_grpo.dataset import TrainingFreeSample
from src.training_free_grpo.runtime import enrich_prompt_context, prefers_chinese


_ENGLISH_RE = re.compile(r"[A-Za-z]{2,}")


@dataclass
class RewardResult:
    score: float
    positive_tags: list[str] = field(default_factory=list)
    negative_tags: list[str] = field(default_factory=list)
    breakdown: dict[str, float] = field(default_factory=dict)


def _contains_any(text: str, keywords: list[str]) -> bool:
    return any(keyword in text for keyword in keywords)


def _localize_label(label: str) -> str:
    zh_map = {
        "cup": "杯子",
        "bottle": "瓶子",
        "plastic bottle": "塑料瓶",
        "water bottle": "水瓶",
        "can": "易拉罐",
        "coke": "可乐罐",
        "coke can": "可乐罐",
        "cola can": "可乐罐",
        "red soda can": "可乐罐",
        "milk carton": "牛奶盒",
        "green bottle": "绿色瓶子",
        "tissue": "纸巾",
        "tissue box": "纸巾盒",
        "tissue pack": "纸巾包",
        "paper towel": "纸巾",
        "mouse": "鼠标",
        "computer mouse": "鼠标",
        "table": "桌面",
    }
    lowered = label.strip().lower()
    return zh_map.get(lowered, label)


def _character_f1(prediction: str, reference: str) -> float:
    pred_tokens = [char for char in prediction if not char.isspace()]
    ref_tokens = [char for char in reference if not char.isspace()]
    if not pred_tokens or not ref_tokens:
        return 0.0
    common = 0
    ref_pool = ref_tokens.copy()
    for token in pred_tokens:
        if token in ref_pool:
            common += 1
            ref_pool.remove(token)
    if common == 0:
        return 0.0
    precision = common / len(pred_tokens)
    recall = common / len(ref_tokens)
    return 2 * precision * recall / max(precision + recall, 1e-6)


class PromptRewardModel:
    def score(self, sample: TrainingFreeSample, response: str) -> RewardResult:
        answer = response.strip()
        context = enrich_prompt_context(sample.prompt_payload)
        detection_payload = dict(
            context.get("detection_payload")
            or context.get("focus_detection")
            or context.get("detection")
            or {}
        )
        scene_detections = list(
            context.get("scene_detections_payload")
            or context.get("scene_detections")
            or context.get("scene_candidates")
            or []
        )
        question_text = str(context.get("question_text", ""))
        breakdown: dict[str, float] = {}
        positive_tags: list[str] = []
        negative_tags: list[str] = []
        score = 0.0

        if not answer:
            return RewardResult(score=-2.0, negative_tags=["empty_response"], breakdown={"empty_response": -2.0})

        if context.get("requires_chinese") or prefers_chinese(question_text):
            if _ENGLISH_RE.search(answer):
                breakdown["chinese_only"] = -0.8
                negative_tags.append("used_english")
                score -= 0.8
            else:
                breakdown["chinese_only"] = 0.4
                positive_tags.append("chinese_only")
                score += 0.4

        max_len = 80 if sample.mode == "grasp_guidance" else 120
        if len(answer) <= max_len:
            breakdown["concise"] = 0.3
            positive_tags.append("concise")
            score += 0.3
        else:
            breakdown["concise"] = -0.2
            negative_tags.append("too_long")
            score -= 0.2

        if not context.get("hand_present"):
            if _contains_any(answer, ["手", "镜头", "画面", "视野"]):
                breakdown["hand_in_view"] = 0.8
                positive_tags.append("hand_in_view_first")
                score += 0.8
            else:
                breakdown["hand_in_view"] = -0.4
                negative_tags.append("missing_hand_in_view")
                score -= 0.4

        dx_sign = context.get("dx_sign")
        if dx_sign == "left":
            if "左" in answer:
                breakdown["horizontal_relation"] = 0.6
                positive_tags.append("explicit_horizontal_relation")
                score += 0.6
            else:
                negative_tags.append("missing_horizontal_relation")
                score -= 0.4
        elif dx_sign == "right":
            if "右" in answer:
                breakdown["horizontal_relation"] = 0.6
                positive_tags.append("explicit_horizontal_relation")
                score += 0.6
            else:
                negative_tags.append("missing_horizontal_relation")
                score -= 0.4

        dz_sign = context.get("dz_sign")
        if dz_sign == "far":
            if _contains_any(answer, ["前", "远", "靠近", "伸"]):
                breakdown["depth_relation"] = 0.5
                positive_tags.append("explicit_depth_relation")
                score += 0.5
            else:
                negative_tags.append("missing_depth_relation")
                score -= 0.3
        elif dz_sign == "near":
            if _contains_any(answer, ["后", "回", "近"]):
                breakdown["depth_relation"] = 0.5
                positive_tags.append("explicit_depth_relation")
                score += 0.5
            else:
                negative_tags.append("missing_depth_relation")
                score -= 0.3

        question_type = str(context.get("question_type", "dialogue"))
        if question_type == "scene_overview" and scene_detections:
            mentioned = 0
            for item in scene_detections[:4]:
                label = _localize_label(str(item.get("label", "")))
                if label and label not in {"桌面", "table"} and label in answer:
                    mentioned += 1
            if mentioned >= min(2, len(scene_detections)):
                breakdown["scene_summary"] = 0.8
                positive_tags.append("summarize_key_objects")
                score += 0.8
            else:
                negative_tags.append("missing_scene_summary")
                score -= 0.5

        if question_type == "motion":
            has_relation = _contains_any(answer, ["左", "右", "前", "后", "上", "下", "正前方"])
            has_action = _contains_any(answer, ["移动", "靠近", "伸", "回", "抓", "放"])
            if has_relation and has_action:
                breakdown["relation_then_action"] = 0.8
                positive_tags.append("relation_then_action")
                score += 0.8
            else:
                negative_tags.append("missing_motion_guidance")
                score -= 0.5

        if not context.get("target_visible", bool(detection_payload.get("visible", False))):
            if _contains_any(answer, ["不确定", "看不清", "暂时", "无法判断"]):
                breakdown["uncertainty"] = 0.4
                positive_tags.append("explicit_uncertainty")
                score += 0.4
            else:
                negative_tags.append("missing_uncertainty")
                score -= 0.2

        reference_answer = sample.reference_answer.strip()
        if reference_answer:
            alignment = _character_f1(answer, reference_answer)
            reference_reward = alignment * 2.0
            breakdown["reference_alignment"] = reference_reward
            if alignment >= 0.35:
                positive_tags.append("reference_alignment")
            else:
                negative_tags.append("low_reference_alignment")
            score += reference_reward

        return RewardResult(
            score=score,
            positive_tags=positive_tags,
            negative_tags=negative_tags,
            breakdown=breakdown,
        )
