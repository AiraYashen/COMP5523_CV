from src.training_free_grpo.profile import ExperienceEntry, PromptProfile
from src.training_free_grpo.runtime import (
    TRAINING_FREE_PREFIX_END,
    TRAINING_FREE_PREFIX_START,
    TrainingFreePromptAdapter,
    strip_training_free_prefix,
)


def test_training_free_prompt_adapter_wraps_prompt_with_matching_experience() -> None:
    adapter = TrainingFreePromptAdapter(
        config={
            "training_free_grpo": {
                "enabled": True,
                "max_shared_principles": 3,
                "max_active_experiences": 3,
                "shared_principles": ["Trust structured evidence first."],
                "mode_principles": {
                    "grasp_guidance": ["Translate controller intent to natural language."]
                },
            }
        },
        profile=PromptProfile(
            shared_principles=["Trust structured evidence first."],
            mode_principles={
                "grasp_guidance": ["Translate controller intent to natural language."]
            },
            experiences=[
                ExperienceEntry(
                    experience_id="g0",
                    mode="grasp_guidance",
                    text="When the hand is absent, ask the user to move the hand into view first.",
                    score=2.0,
                    trigger={"hand_present": False},
                )
            ],
        ),
    )
    wrapped = adapter.wrap_prompt(
        base_prompt="base prompt",
        mode="grasp_guidance",
        context={"hand_pose_payload": {"hand_present": False}},
    )
    assert TRAINING_FREE_PREFIX_START in wrapped
    assert TRAINING_FREE_PREFIX_END in wrapped
    assert "move the hand into view" in wrapped
    assert wrapped.endswith("base prompt")


def test_strip_training_free_prefix_removes_injected_prior() -> None:
    prompt = (
        f"{TRAINING_FREE_PREFIX_START}\nShared principles:\n1. cue\n"
        f"{TRAINING_FREE_PREFIX_END}\n\nactual prompt"
    )
    assert strip_training_free_prefix(prompt) == "actual prompt"
