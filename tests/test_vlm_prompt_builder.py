from src.common.schemas import CommandSpec, ControlState, FusionState, GuidanceCommand
from src.common.schemas import Detection, HandPoseResult
from src.vlm.prompt_builder import (
    build_detection_context_payload,
    build_detection_payload,
    build_detection_payloads,
    build_fusion_payload,
    build_glm_dialogue_text,
    build_glm_guidance_text,
    build_glm_scene_query_text,
    build_hand_pose_payload,
    build_scene_query_prompt,
    build_spatial_hint_text,
    build_vlm_prompt,
)


def test_build_vlm_prompt_contains_authoritative_state() -> None:
    prompt = build_vlm_prompt(
        event="wait_hand",
        command=CommandSpec(
            action="grasp",
            target_name_raw="coke",
            target_prompt_en="red soda can",
            spatial_hint="front",
        ),
        control_state=ControlState(name="WAIT_HAND"),
        fusion_state=FusionState(
            target_visible=True,
            hand_visible=False,
            target_locked=True,
            target_center_xy=(100.0, 100.0),
            hand_center_xy=None,
            target_depth=0.5,
            hand_depth=None,
            dx_norm=None,
            dy_norm=None,
            dz_rel=None,
            target_confidence=0.9,
            hand_confidence=0.0,
            frame_id=1,
            timestamp_ms=1,
        ),
        guidance=GuidanceCommand(command="place hand in view", reason="hand missing"),
    )
    assert "Target object: red soda can" in prompt
    assert "Control state: WAIT_HAND" in prompt
    assert "red soda can" in prompt
    assert "wait_hand" in prompt


def test_build_scene_query_prompt_contains_question_and_hints() -> None:
    prompt = build_scene_query_prompt(
        question="前面有什么？",
        detection=Detection(
            bbox_xyxy=(10.0, 10.0, 40.0, 60.0), label="cup", score=0.91
        ),
        hand_pose=HandPoseResult(hand_present=True, score=0.8),
        fusion_state=FusionState(
            target_visible=True,
            hand_visible=True,
            target_locked=True,
            target_center_xy=(25.0, 35.0),
            hand_center_xy=(12.0, 18.0),
            target_depth=0.6,
            hand_depth=0.4,
            dx_norm=0.1,
            dy_norm=0.2,
            dz_rel=0.2,
            target_confidence=0.91,
            hand_confidence=0.8,
            frame_id=1,
            timestamp_ms=1,
        ),
    )
    assert "User question: 前面有什么？" in prompt
    assert "Top detection: cup" in prompt
    assert "Panel legend" in prompt


def test_build_glm_scene_query_text_contains_json_sections() -> None:
    fusion_state = FusionState(
        target_visible=True,
        hand_visible=True,
        target_locked=True,
        target_center_xy=(25.0, 35.0),
        hand_center_xy=(12.0, 18.0),
        target_depth=0.6,
        hand_depth=0.4,
        dx_norm=0.1,
        dy_norm=0.2,
        dz_rel=0.2,
        target_confidence=0.91,
        hand_confidence=0.8,
        frame_id=1,
        timestamp_ms=1,
    )
    prompt = build_glm_scene_query_text(
        question="前面有什么？",
        conversation_history="无",
        detection_payload=build_detection_payload(
            Detection(bbox_xyxy=(10.0, 10.0, 40.0, 60.0), label="cup", score=0.91)
        ),
        scene_detections_payload=build_detection_payloads(
            [
                Detection(bbox_xyxy=(10.0, 10.0, 40.0, 60.0), label="cup", score=0.91),
                Detection(
                    bbox_xyxy=(45.0, 10.0, 80.0, 60.0),
                    label="computer mouse",
                    score=0.78,
                ),
            ]
        ),
        hand_pose_payload=build_hand_pose_payload(
            HandPoseResult(hand_present=True, score=0.8), fusion_state
        ),
        fusion_payload=build_fusion_payload(fusion_state),
        training_free_prefix="[Training-Free Prompt Prior]\ncue\n[/Training-Free Prompt Prior]",
    )
    assert "[Training-Free Prompt Prior]" in prompt
    assert "焦点目标JSON：" in prompt
    assert "场景候选物体JSON：" in prompt
    assert '"label": "cup"' in prompt
    assert "手部姿态JSON：" in prompt
    assert "融合状态JSON：" in prompt
    assert "只输出最终要播报的一段自然中文" in prompt


def test_build_glm_guidance_text_requires_chinese_assistive_guidance() -> None:
    prompt = build_glm_guidance_text(
        event="state_change",
        conversation_history="无",
        command=CommandSpec(
            action="grasp",
            target_name_raw="瓶子",
            target_prompt_en="green bottle",
            spatial_hint="none",
        ),
        control_state=ControlState(name="APPROACH"),
        guidance=GuidanceCommand(command="right", reason="target right of hand"),
        detection_payload={"visible": True, "label": "green bottle"},
        hand_pose_payload={"hand_present": True},
        fusion_payload={"dx_norm": 0.1},
        training_free_prefix="[Training-Free Prompt Prior]\ncue\n[/Training-Free Prompt Prior]",
    )
    assert "[Training-Free Prompt Prior]" in prompt
    assert "你必须始终只用简体中文回答" in prompt
    assert "最关键的一步怎么做" in prompt
    assert "规则命令解释" in prompt


def test_build_glm_dialogue_text_contains_professional_broadcast_constraints() -> (
    None
):
    prompt = build_glm_dialogue_text(
        user_text="我还需要怎么做能抓到目标",
        conversation_history="无",
        detection_payload=build_detection_context_payload(
            Detection(
                bbox_xyxy=(10.0, 10.0, 40.0, 60.0), label="tissue box", score=0.91
            ),
            [
                Detection(
                    bbox_xyxy=(10.0, 10.0, 40.0, 60.0), label="tissue box", score=0.91
                ),
                Detection(
                    bbox_xyxy=(60.0, 10.0, 90.0, 60.0), label="computer mouse mouse", score=0.84
                ),
            ],
        ),
        hand_pose_payload={"hand_present": True, "palm_center_xy": [20.0, 20.0]},
        fusion_payload={"dx_norm": 0.2, "dy_norm": 0.1, "dz_rel": 0.15},
        training_free_prefix="[Training-Free Prompt Prior]\ncue\n[/Training-Free Prompt Prior]",
    )
    assert "[Training-Free Prompt Prior]" in prompt
    assert "我还需要怎么做能抓到目标" in prompt
    assert "不要输出JSON" in prompt
    assert "绝不能把屏幕左右或镜像预览直接当成用户方向" in prompt
    assert "物体检测JSON" in prompt
    assert "融合状态JSON" in prompt
    assert "为用户提供可靠、克制、可执行的空间理解与抓取帮助" in prompt
    assert "表达风格参考" in prompt


def test_build_spatial_hint_text_uses_clock_direction_and_distance() -> None:
    hint = build_spatial_hint_text(
        FusionState(
            target_visible=True,
            hand_visible=True,
            target_locked=True,
            target_center_xy=(25.0, 20.0),
            hand_center_xy=(10.0, 25.0),
            target_depth=0.8,
            hand_depth=0.5,
            dx_norm=0.2,
            dy_norm=-0.1,
            dz_rel=0.18,
            target_confidence=0.9,
            hand_confidence=0.8,
            frame_id=1,
            timestamp_ms=1,
            frame_size_xy=(100, 100),
        )
    )
    assert "当前同时检测到目标和手" in hint


def test_build_glm_dialogue_text_confirmation_allows_visible_grasp_judgment() -> None:
    prompt = build_glm_dialogue_text(
        user_text="现在我的手抓住的是水瓶吗",
        conversation_history="无",
        detection_payload={
            "focus_detection": {
                "visible": True,
                "label": "water bottle",
                "score": 0.9,
                "bbox_xyxy": [10.0, 10.0, 110.0, 210.0],
            },
            "scene_candidates": [
                {
                    "visible": True,
                    "label": "water bottle",
                    "score": 0.9,
                    "bbox_xyxy": [10.0, 10.0, 110.0, 210.0],
                }
            ],
        },
        hand_pose_payload={"hand_present": True, "palm_center_xy": [240.0, 240.0]},
        fusion_payload={"dz_rel": 0.25},
    )
    assert "当前RGB画面中只要手指或掌心已经明显包住、压住、捏住或稳定覆盖目标主体" in prompt
    assert "才回答“暂时不能确认”" in prompt
