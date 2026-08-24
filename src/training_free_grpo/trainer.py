from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
import json
import random
from pathlib import Path
from typing import Any

from src.common.config import load_app_config, load_training_free_grpo_config
from src.training_free_grpo.dataset import TrainingFreeSample, load_training_samples
from src.training_free_grpo.profile import ExperienceEntry, PromptProfile
from src.training_free_grpo.reward import PromptRewardModel, RewardResult
from src.training_free_grpo.runtime import (
    TrainingFreePromptAdapter,
    enrich_prompt_context,
)
from src.vlm.vlm_service import VLMService


@dataclass
class RolloutRecord:
    sample_id: str
    mode: str
    group_index: int
    prompt: str
    response: str
    reward: float
    positive_tags: list[str]
    negative_tags: list[str]
    breakdown: dict[str, float]
    prompt_payload: dict[str, Any]
    reference_answer: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class TrainingFreeGRPOTrainer:
    def __init__(
        self,
        app_config: dict[str, Any] | None = None,
        tfgrpo_config: dict[str, Any] | None = None,
    ) -> None:
        self.app_config = app_config or load_app_config()
        root = tfgrpo_config or load_training_free_grpo_config()
        self.tfgrpo_config = dict(root.get("training_free_grpo", root))
        vlm_cfg = self.app_config.get("vlm", {})
        self.vlm_service = VLMService(
            enabled=True,
            model_id=vlm_cfg.get("model_id", ""),
            backend=vlm_cfg.get("backend", "transformers"),
            device=vlm_cfg.get("device_preference", "cpu"),
            max_new_tokens=vlm_cfg.get("max_new_tokens", 256),
            api_key=vlm_cfg.get("api_key", ""),
            api_base_url=vlm_cfg.get("api_base_url", ""),
            timeout_s=vlm_cfg.get("timeout_s", 45),
            thinking_enabled=vlm_cfg.get("thinking_enabled", False),
        )
        self.prompt_adapter = TrainingFreePromptAdapter(config=root)
        self.reward_model = PromptRewardModel()

    def train(
        self,
        dataset_path: str | Path,
        output_dir: str | Path,
        epochs: int | None = None,
        batch_size: int | None = None,
        group_size: int | None = None,
        rollout_temperature: float | None = None,
        rollout_concurrency: int | None = None,
    ) -> PromptProfile:
        samples = load_training_samples(dataset_path)
        if not samples:
            raise RuntimeError(f"No training samples found in {dataset_path}")
        output_root = Path(output_dir)
        output_root.mkdir(parents=True, exist_ok=True)
        epochs = int(epochs or self.tfgrpo_config.get("epochs", 2))
        batch_size = int(batch_size or self.tfgrpo_config.get("batch_size", 4))
        group_size = int(group_size or self.tfgrpo_config.get("group_size", 4))
        rollout_temperature = float(
            rollout_temperature
            if rollout_temperature is not None
            else self.tfgrpo_config.get("rollout_temperature", 0.7)
        )
        rollout_concurrency = int(
            rollout_concurrency or self.tfgrpo_config.get("rollout_concurrency", 2)
        )
        max_experiences = int(self.tfgrpo_config.get("max_total_experiences", 12))
        current_profile = self.prompt_adapter.load_profile()
        random.seed(int(self.tfgrpo_config.get("seed", 42)))

        step = 0
        for epoch in range(epochs):
            shuffled = samples[:]
            random.shuffle(shuffled)
            epoch_dir = output_root / f"epoch_{epoch}"
            epoch_dir.mkdir(parents=True, exist_ok=True)
            for offset in range(0, len(shuffled), batch_size):
                batch = shuffled[offset : offset + batch_size]
                step_dir = output_root / f"step_{step}"
                step_dir.mkdir(parents=True, exist_ok=True)
                rollouts = self._rollout_batch(
                    batch=batch,
                    profile=current_profile,
                    group_size=group_size,
                    rollout_temperature=rollout_temperature,
                    rollout_concurrency=rollout_concurrency,
                )
                self._write_rollouts(rollouts, step_dir / "rollouts.jsonl")
                current_profile = self._update_profile(
                    profile=current_profile,
                    batch=batch,
                    rollouts=rollouts,
                    max_experiences=max_experiences,
                )
                current_profile.metadata.update(
                    {
                        "last_epoch": epoch,
                        "last_step": step,
                        "dataset_path": str(dataset_path),
                    }
                )
                current_profile.save(step_dir / "profile.json")
                step += 1

        latest_profile_path = Path(
            str(
                self.tfgrpo_config.get(
                    "profile_path", "outputs/training_free_grpo/latest_profile.json"
                )
            )
        )
        current_profile.save(latest_profile_path)
        current_profile.save(output_root / "latest_profile.json")
        return current_profile

    def _rollout_batch(
        self,
        batch: list[TrainingFreeSample],
        profile: PromptProfile,
        group_size: int,
        rollout_temperature: float,
        rollout_concurrency: int,
    ) -> list[RolloutRecord]:
        tasks: list[tuple[TrainingFreeSample, int]] = []
        for sample in batch:
            for group_index in range(group_size):
                tasks.append((sample, group_index))
        rollouts: list[RolloutRecord] = []
        with ThreadPoolExecutor(max_workers=max(1, rollout_concurrency)) as executor:
            futures = {
                executor.submit(
                    self._single_rollout,
                    sample,
                    group_index,
                    profile,
                    rollout_temperature,
                ): (sample.sample_id, group_index)
                for sample, group_index in tasks
            }
            for future in as_completed(futures):
                rollouts.append(future.result())
        rollouts.sort(key=lambda item: (item.sample_id, item.group_index))
        return rollouts

    def _single_rollout(
        self,
        sample: TrainingFreeSample,
        group_index: int,
        profile: PromptProfile,
        rollout_temperature: float,
    ) -> RolloutRecord:
        context = enrich_prompt_context(sample.prompt_payload)
        prompt = self.prompt_adapter.wrap_prompt(
            base_prompt=sample.prompt,
            mode=sample.mode,
            context=context,
            profile=profile,
        )
        rgb_image = self._load_rgb_image(sample.rgb_image_path)
        depth_map = self._load_depth_map(sample.depth_image_path)
        narration = self.vlm_service.analyze_multimodal(
            rgb_image=rgb_image,
            depth_map=depth_map,
            prompt=prompt,
            temperature=rollout_temperature,
        )
        reward = self.reward_model.score(sample, narration.scene_description)
        return RolloutRecord(
            sample_id=sample.sample_id,
            mode=sample.mode,
            group_index=group_index,
            prompt=prompt,
            response=narration.scene_description,
            reward=reward.score,
            positive_tags=reward.positive_tags,
            negative_tags=reward.negative_tags,
            breakdown=reward.breakdown,
            prompt_payload=sample.prompt_payload,
            reference_answer=sample.reference_answer,
        )

    def _update_profile(
        self,
        profile: PromptProfile,
        batch: list[TrainingFreeSample],
        rollouts: list[RolloutRecord],
        max_experiences: int,
    ) -> PromptProfile:
        sample_lookup = {sample.sample_id: sample for sample in batch}
        grouped: dict[str, list[RolloutRecord]] = {}
        for rollout in rollouts:
            grouped.setdefault(rollout.sample_id, []).append(rollout)
        candidates: list[ExperienceEntry] = []
        for sample_id, sample_rollouts in grouped.items():
            baseline = sum(item.reward for item in sample_rollouts) / max(
                len(sample_rollouts), 1
            )
            for rollout in sample_rollouts:
                advantage = rollout.reward - baseline
                reward_result = RewardResult(
                    score=rollout.reward,
                    positive_tags=rollout.positive_tags,
                    negative_tags=rollout.negative_tags,
                    breakdown=rollout.breakdown,
                )
                candidates.extend(
                    self._experience_candidates(
                        sample=sample_lookup[sample_id],
                        reward=reward_result,
                        advantage=advantage,
                    )
                )
        updated = PromptProfile.from_dict(profile.to_dict())
        updated.merge_experiences(candidates, max_total=max_experiences)
        return updated

    def _experience_candidates(
        self,
        sample: TrainingFreeSample,
        reward: RewardResult,
        advantage: float,
    ) -> list[ExperienceEntry]:
        context = enrich_prompt_context(sample.prompt_payload)
        weight = abs(advantage) + max(reward.score, 0.0) * 0.1
        if weight <= 0:
            return []
        candidates: list[ExperienceEntry] = []
        mode = sample.mode

        def add(text: str, trigger: dict[str, Any]) -> None:
            candidates.append(
                ExperienceEntry(
                    experience_id="",
                    mode=mode,
                    text=text,
                    score=weight,
                    trigger=trigger,
                )
            )

        if (
            "hand_in_view_first" in reward.positive_tags
            or "missing_hand_in_view" in reward.negative_tags
        ):
            add(
                "When the hand is absent, first ask the user to move the hand into the camera view, then continue with spatial guidance.",
                {"hand_present": False},
            )
        if (
            "explicit_horizontal_relation" in reward.positive_tags
            or "missing_horizontal_relation" in reward.negative_tags
        ):
            dx_sign = str(context.get("dx_sign", "center"))
            if dx_sign in {"left", "right"}:
                add(
                    f"When lateral offset is clear, explicitly describe the target as {dx_sign} of the hand instead of using vague location words.",
                    {"dx_sign": dx_sign},
                )
        if (
            "explicit_depth_relation" in reward.positive_tags
            or "missing_depth_relation" in reward.negative_tags
        ):
            dz_sign = str(context.get("dz_sign", "aligned"))
            if dz_sign == "far":
                add(
                    "When depth says the target is farther, explicitly mention that the hand should move forward or closer.",
                    {"dz_sign": "far"},
                )
            if dz_sign == "near":
                add(
                    "When the hand is already too close, say it should move slightly back instead of only repeating horizontal guidance.",
                    {"dz_sign": "near"},
                )
        if (
            "summarize_key_objects" in reward.positive_tags
            or "missing_scene_summary" in reward.negative_tags
        ):
            add(
                "For broad scene questions, summarize several salient tabletop objects instead of only mentioning the table background.",
                {"question_type": "scene_overview"},
            )
        if (
            "relation_then_action" in reward.positive_tags
            or "missing_motion_guidance" in reward.negative_tags
        ):
            add(
                "For motion questions, answer with target-hand relation first and the next executable hand action second.",
                {"question_type": "motion"},
            )
        if (
            "used_english" in reward.negative_tags
            or "chinese_only" in reward.positive_tags
        ):
            add(
                "When the user speaks Chinese, keep the final answer in concise Simplified Chinese and hide raw controller labels.",
                {"requires_chinese": True},
            )
        if (
            "explicit_uncertainty" in reward.positive_tags
            or "missing_uncertainty" in reward.negative_tags
        ):
            add(
                "When the target evidence is weak, state the uncertainty briefly before giving any tentative suggestion.",
                {"target_visible": False},
            )
        if "reference_alignment" in reward.positive_tags:
            add(
                "Prefer evidence-backed assistive wording that matches the expected task answer without adding unrelated scene details.",
                {},
            )
        for index, item in enumerate(candidates):
            item.experience_id = f"{mode}_{index}"
        return candidates

    @staticmethod
    def _write_rollouts(rollouts: list[RolloutRecord], path: Path) -> None:
        with path.open("w", encoding="utf-8") as handle:
            for rollout in rollouts:
                handle.write(json.dumps(rollout.to_dict(), ensure_ascii=False) + "\n")

    @staticmethod
    def _load_rgb_image(path: str):
        import cv2

        image = cv2.imread(path, cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Could not load RGB image: {path}")
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    @staticmethod
    def _load_depth_map(path: str):
        import cv2

        depth = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if depth is None:
            raise FileNotFoundError(f"Could not load depth image: {path}")
        return depth.astype("float32")
