from src.training_free_grpo.dataset import TrainingFreeSample
from src.training_free_grpo.reward import PromptRewardModel


def test_prompt_reward_model_rewards_motion_relation_and_action() -> None:
    sample = TrainingFreeSample(
        sample_id="sample-1",
        mode="scene_query",
        prompt="base",
        rgb_image_path="rgb.png",
        depth_image_path="depth.png",
        prompt_payload={
            "question": "我的手应该怎么移动能拿到纸巾",
            "fusion_payload": {
                "target_visible": True,
                "hand_visible": True,
                "dx_norm": 0.2,
                "dy_norm": 0.0,
                "dz_rel": 0.2,
            },
            "hand_pose_payload": {"hand_present": True},
        },
        reference_answer="纸巾盒在右前方，请把手向右前方移动一点。",
    )
    reward = PromptRewardModel().score(
        sample,
        "纸巾盒在你手的右前方，稍远。请把手向右前方移动一点。",
    )
    assert reward.score > 1.0
    assert "explicit_horizontal_relation" in reward.positive_tags
    assert "relation_then_action" in reward.positive_tags
