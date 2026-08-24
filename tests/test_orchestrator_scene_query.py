from __future__ import annotations

import time
from typing import Any

import numpy as np

from src.app.orchestrator import Orchestrator
from src.app.session_manager import SessionManager
from src.common.schemas import (
    AsrResult,
    CommandSpec,
    Detection,
    FramePacket,
    HandPoseResult,
    VLMNarration,
)


class RecordingTTS:
    def __init__(self) -> None:
        self.messages: list[str] = []
        self.last_spoken_finished_at = -(10**9)

    def speak(self, text: str) -> None:
        self.messages.append(text)
        self.last_spoken_finished_at = time.monotonic()

    def time_since_last_speak_ms(self) -> float:
        return max(0.0, (time.monotonic() - self.last_spoken_finished_at) * 1000.0)


class FakeCaptureController:
    def __init__(self, frame: FramePacket) -> None:
        self.frame = frame

    def get_initial_frame(self, image_path: str | None = None) -> FramePacket:
        return self.frame


class FakeDetector:
    def detect(self, rgb_image, prompt: str) -> list[Detection]:
        return [Detection(bbox_xyxy=(5.0, 5.0, 20.0, 20.0), label="cup", score=0.9)]


class EmptyDetector:
    def detect(self, rgb_image, prompt: str) -> list[Detection]:
        return []


class FakeDepthEstimator:
    def predict_depth(self, rgb_image):
        return type(
            "DepthResultStub", (), {"depth_map": np.ones((32, 32), dtype=np.float32)}
        )()


class FakeHandEstimator:
    def estimate_hand(self, rgb_image) -> HandPoseResult:
        return HandPoseResult(hand_present=False, score=0.0, landmarks_xy=[])


class SlowVLMService:
    def __init__(self) -> None:
        self.prompts: list[str] = []

    def analyze_multimodal(self, rgb_image, depth_map, prompt: str) -> VLMNarration:
        self.prompts.append(prompt)
        time.sleep(0.05)
        return VLMNarration(
            scene_status="ok",
            scene_description="前方有一个杯子。",
            uncertainty="low",
            speak=True,
        )


def build_test_orchestrator() -> tuple[Any, RecordingTTS]:
    orchestrator: Any = Orchestrator.__new__(Orchestrator)
    rgb_image = np.zeros((32, 32, 3), dtype=np.uint8)
    frame = FramePacket(frame_id=1, timestamp_ms=1000, rgb_image=rgb_image)
    tts = RecordingTTS()
    orchestrator.capture_controller = FakeCaptureController(frame)
    orchestrator.detector = FakeDetector()
    orchestrator.fallback_detector = EmptyDetector()
    orchestrator.depth_estimator = FakeDepthEstimator()
    orchestrator.hand_estimator = FakeHandEstimator()
    orchestrator.threshold_cfg = {
        "target_score_weight": 0.5,
        "spatial_score_weight": 0.3,
        "stability_score_weight": 0.2,
    }
    orchestrator.prompt_cfg = {
        "object_map": {
            "cup": {"keywords": ["cup", "杯子"], "prompts": ["cup"]},
            "tissue_box": {
                "keywords": ["tissue", "纸巾"],
                "prompts": ["tissue box"],
            },
        }
    }
    orchestrator.scene_query_detection_prompt = "cup . bottle"
    orchestrator.scene_wait_interval_ms = 10
    orchestrator.history_max_turns = 6
    orchestrator.log_vlm_inputs = False
    orchestrator.tts_service = tts
    orchestrator.vlm_service = SlowVLMService()
    orchestrator.session_manager = SessionManager()
    orchestrator.overlay_enabled = False
    orchestrator.save_overlay = False
    return orchestrator, tts


def test_await_scene_query_answer_emits_wait_messages() -> None:
    orchestrator, tts = build_test_orchestrator()
    rgb_image = np.zeros((32, 32, 3), dtype=np.uint8)
    depth_map = np.ones((32, 32), dtype=np.float32)
    answer = orchestrator._await_scene_query_answer(
        rgb_image, depth_map, "前面有什么？"
    )
    assert answer == "前方有一个杯子。"
    assert tts.messages.count("请等待") >= 1


def test_run_scene_query_speaks_final_answer() -> None:
    orchestrator, tts = build_test_orchestrator()
    orchestrator._run_scene_query(
        CommandSpec(
            action="scene_query",
            target_name_raw="",
            target_prompt_en="",
            query_text="前面有什么？",
        )
    )
    assert tts.messages[-1] == "前方有一个杯子。"


def test_run_dialogue_turn_speaks_raw_vlm_answer_without_storing_history() -> None:
    orchestrator, tts = build_test_orchestrator()
    orchestrator._run_dialogue_turn(
        user_text="桌面上有什么东西",
    )
    assert tts.messages[-1] == "前方有一个杯子。"
    assert orchestrator.session_manager.render_history() == "无"


def test_collect_detection_context_uses_generic_scene_candidates_only() -> None:
    orchestrator, _ = build_test_orchestrator()
    focus_detection, scene_detections = orchestrator._collect_detection_context(
        np.zeros((32, 32, 3), dtype=np.uint8)
    )
    assert focus_detection is not None
    assert scene_detections


def test_match_scene_query_language_uses_translator_for_chinese_question() -> None:
    orchestrator, _ = build_test_orchestrator()
    prompt = "User question: 前面有什么？\nTop detection: cup"
    answer = orchestrator._match_scene_query_language(
        "The object is a red soda coke.", prompt
    )
    assert answer == "前方有一个杯子。"


def test_refine_scene_query_answer_prefers_detection_for_generic_question() -> None:
    orchestrator, _ = build_test_orchestrator()
    answer = orchestrator._refine_scene_query_answer(
        answer="前方有一个桌面。",
        question="前面有什么？",
        detection=Detection(bbox_xyxy=(0.0, 0.0, 10.0, 10.0), label="cup", score=0.9),
        detections=[
            Detection(bbox_xyxy=(0.0, 0.0, 10.0, 10.0), label="table", score=0.95),
            Detection(bbox_xyxy=(10.0, 0.0, 20.0, 10.0), label="cup", score=0.9),
            Detection(
                bbox_xyxy=(20.0, 0.0, 30.0, 10.0), label="computer mouse", score=0.8
            ),
        ],
        hand_pose=HandPoseResult(hand_present=False, score=0.0, landmarks_xy=[]),
        fusion_state=type(
            "FusionStub",
            (),
            {
                "target_visible": True,
                "hand_visible": False,
                "target_center_xy": (5.0, 5.0),
                "frame_size_xy": (32, 32),
                "dx_norm": None,
                "dy_norm": None,
                "dz_rel": None,
            },
        )(),
    )
    assert answer == "桌面上有杯子、鼠标。"


def test_refine_scene_query_answer_guides_motion_question() -> None:
    orchestrator, _ = build_test_orchestrator()
    answer = orchestrator._refine_scene_query_answer(
        answer="",
        question="我的手应该怎么移动能拿到纸巾",
        detection=Detection(
            bbox_xyxy=(0.0, 0.0, 10.0, 10.0), label="tissue box", score=0.9
        ),
        detections=[
            Detection(bbox_xyxy=(0.0, 0.0, 10.0, 10.0), label="tissue box", score=0.9)
        ],
        hand_pose=HandPoseResult(hand_present=True, score=0.9, landmarks_xy=[]),
        fusion_state=type(
            "FusionStub",
            (),
            {
                "target_visible": True,
                "hand_visible": True,
                "target_center_xy": (20.0, 8.0),
                "frame_size_xy": (32, 32),
                "dx_norm": 0.2,
                "dy_norm": 0.0,
                "dz_rel": 0.2,
            },
        )(),
    )
    assert answer == "纸巾盒在你手的右侧，稍远。请把手向右移动一点。"


def test_wait_for_post_tts_cooldown_sleeps_when_needed(monkeypatch) -> None:
    orchestrator, _ = build_test_orchestrator()
    orchestrator.post_tts_cooldown_ms = 200
    orchestrator.tts_service.speak("刚刚播报")
    captured: list[float] = []

    def fake_sleep(seconds: float) -> None:
        captured.append(seconds)

    monkeypatch.setattr("src.app.orchestrator.time.sleep", fake_sleep)
    orchestrator._wait_for_post_tts_cooldown(interactive_mode=True)
    assert captured
    assert captured[0] > 0.0


def test_should_request_repeat_for_low_confidence() -> None:
    orchestrator, _ = build_test_orchestrator()
    orchestrator.asr_min_confidence = 0.45
    result = AsrResult(text="这个品质", confidence=0.2, latency_ms=100)
    assert orchestrator._should_request_repeat(result, interactive_mode=True) is True


def test_should_not_request_repeat_for_confident_one_shot_audio() -> None:
    orchestrator, _ = build_test_orchestrator()
    orchestrator.asr_min_confidence = 0.45
    result = AsrResult(text="前面有什么？", confidence=0.8, latency_ms=100)
    assert orchestrator._should_request_repeat(result, interactive_mode=False) is False


def test_force_chinese_output_uses_translator_for_english_text() -> None:
    orchestrator, _ = build_test_orchestrator()
    answer = orchestrator._force_chinese_output("Move your hand right.")
    assert answer == "前方有一个杯子。"


def test_should_timeout_grasp_session_only_after_tracking_loss() -> None:
    orchestrator, _ = build_test_orchestrator()
    orchestrator.session_timeout_s = 90.0
    orchestrator.session_max_runtime_s = 300.0
    message = orchestrator._should_timeout_grasp_session(
        now_s=120.0,
        start_time_s=0.0,
        last_tracking_s=119.0,
        fusion_state=type(
            "FusionStub",
            (),
            {"target_visible": True, "hand_visible": False},
        )(),
    )
    assert message is None


def test_should_timeout_grasp_session_after_long_tracking_loss() -> None:
    orchestrator, _ = build_test_orchestrator()
    orchestrator.session_timeout_s = 90.0
    orchestrator.session_max_runtime_s = 300.0
    message = orchestrator._should_timeout_grasp_session(
        now_s=200.0,
        start_time_s=0.0,
        last_tracking_s=100.0,
        fusion_state=type(
            "FusionStub",
            (),
            {"target_visible": False, "hand_visible": False},
        )(),
    )
    assert message == "长时间没有有效跟踪，请重新调整手和目标后再试一次。"


def test_refine_grasp_guidance_falls_back_when_missing_action() -> None:
    orchestrator, _ = build_test_orchestrator()
    fallback = "我能看到瓶子。请先把手移到镜头里。"
    answer = orchestrator._refine_grasp_guidance(
        text="目标物体是瓶子，当前手不在视野中。",
        command=CommandSpec(
            action="grasp",
            target_name_raw="瓶子",
            target_prompt_en="green bottle",
        ),
        guidance=type("GuidanceStub", (), {"command": "place hand in view"})(),
        fusion_state=type("FusionStub", (), {"hand_visible": False})(),
        fallback_text=fallback,
    )
    assert answer == fallback


def test_describe_target_relation_reports_frame_position_without_hand() -> None:
    orchestrator, _ = build_test_orchestrator()
    relation = orchestrator._describe_target_relation(
        "瓶子",
        type(
            "FusionStub",
            (),
            {
                "target_visible": True,
                "hand_visible": False,
                "target_center_xy": (24.0, 6.0),
                "frame_size_xy": (32, 32),
            },
        )(),
    )
    assert relation == "我能看到瓶子，它在画面右侧上方。"


def test_templated_narration_guides_hand_into_frame_toward_target() -> None:
    orchestrator, _ = build_test_orchestrator()
    message = orchestrator._templated_narration(
        event="frame_guidance",
        command=CommandSpec(
            action="grasp",
            target_name_raw="瓶子",
            target_prompt_en="green bottle",
        ),
        guidance=type("GuidanceStub", (), {"command": "place hand in view"})(),
        fusion_state=type(
            "FusionStub",
            (),
            {
                "target_visible": True,
                "hand_visible": False,
                "target_center_xy": (24.0, 6.0),
                "frame_size_xy": (32, 32),
            },
        )(),
    )
    assert (
        message
        == "我能看到瓶子，它在画面右侧上方。请先把手移到镜头里，再朝它的方向靠近。"
    )
