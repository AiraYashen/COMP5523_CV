from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, TimeoutError
import json
import time
from pathlib import Path

import cv2
import numpy as np

from src.app.capture_controller import CaptureController
from src.app.session_manager import SessionManager
from src.audio.asr_service import AsrService
from src.audio.command_player import CommandPlayer
from src.audio.narration_player import NarrationPlayer
from src.audio.tts_service import TtsService
from src.common.config import load_app_config, load_prompt_config, load_threshold_config
from src.common.schemas import (
    CommandSpec,
    ControlState,
    Detection,
    FusionState,
    GuidanceCommand,
)
from src.training_free_grpo.runtime import TrainingFreePromptAdapter
from src.ui.camera_preview import render_camera_preview
from src.control.command_policy import decide_command
from src.control.safety_guard import enforce_safety
from src.control.state_machine import next_state
from src.fusion.fusion_state import build_fusion_state
from src.fusion.target_selector import select_target
from src.fusion.temporal_filter import smooth_state
from src.perception.factory import (
    build_depth_estimator,
    build_detector,
    build_hand_estimator,
)
from src.perception.detector_mock import MockDetector
from src.ui.overlay import render_overlay
from src.vlm.prompt_builder import (
    build_detection_payload,
    build_detection_context_payload,
    build_detection_payloads,
    build_fusion_payload,
    build_glm_dialogue_text,
    build_glm_guidance_text,
    build_glm_scene_query_text,
    build_hand_pose_payload,
    build_spatial_hint_text,
)
from src.vlm.vlm_service import VLMService


def resolve_device(preference: str) -> str:
    if preference == "cpu":
        return "cpu"
    try:
        import torch

        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"


class Orchestrator:
    def __init__(self, camera_service) -> None:
        self.camera_service = camera_service
        self.app_cfg = load_app_config()
        self.prompt_cfg = load_prompt_config()
        self.threshold_cfg = load_threshold_config()

        app = self.app_cfg["app"]
        perception = self.app_cfg.get("perception", {})
        models = self.app_cfg["models"]
        device = resolve_device(app.get("device_preference", "auto"))
        asr_device = resolve_device(app.get("asr_device_preference", "auto"))

        self.asr_service = AsrService(
            backend=app.get("asr_backend", "text"),
            model_id=models.get("whisper_id"),
            record_seconds=app.get("asr_record_seconds", 8),
            language=app.get("asr_language"),
            device=asr_device,
            min_speech_ms=app.get("asr_min_speech_ms", 250),
            min_silence_ms=app.get("asr_min_silence_ms", 700),
            chunk_ms=app.get("asr_chunk_ms", 200),
            silence_threshold=app.get("asr_silence_threshold", 0.015),
        )
        self.tts_service = TtsService(backend=app.get("tts_backend", "print"))
        vlm_cfg = self.app_cfg.get("vlm", {})
        self.command_player = CommandPlayer(
            self.tts_service,
            debounce_ms=self.threshold_cfg["command_debounce_ms"],
            enabled=vlm_cfg.get("speak_raw_commands", False),
        )
        vlm_device = resolve_device(vlm_cfg.get("device_preference", "cpu"))
        self.narration_player = NarrationPlayer(
            self.tts_service,
            cooldown_ms=vlm_cfg.get("narration_cooldown_ms", 5000),
            priority_window_ms=vlm_cfg.get("speech_priority_window_ms", 1500),
        )
        self.vlm_service = VLMService(
            enabled=vlm_cfg.get("enabled", False),
            model_id=vlm_cfg.get("model_id", ""),
            backend=vlm_cfg.get("backend", "transformers"),
            device=vlm_device,
            max_new_tokens=vlm_cfg.get("max_new_tokens", 96),
            api_key=vlm_cfg.get("api_key", ""),
            api_base_url=vlm_cfg.get("api_base_url", ""),
            timeout_s=vlm_cfg.get("timeout_s", 45),
            thinking_enabled=vlm_cfg.get("thinking_enabled", False),
        )
        self.vlm_cfg = vlm_cfg
        self.guidance_every_frame = vlm_cfg.get("guidance_every_frame", True)
        self.scene_wait_interval_ms = vlm_cfg.get("scene_wait_interval_ms", 1000)
        self.log_vlm_inputs = bool(vlm_cfg.get("log_vlm_inputs", False))
        self.vlm_input_dir = Path(vlm_cfg.get("vlm_input_dir", "outputs/vlm_inputs"))
        self.scene_query_detection_prompt = self._build_scene_query_detection_prompt()
        self.capture_controller = CaptureController(
            camera_service, burst_count=self.app_cfg["capture"]["burst_count"]
        )
        self.session_manager = SessionManager()
        self.asr_min_confidence = app.get("asr_min_confidence", 0.45)
        self.post_tts_cooldown_ms = app.get("asr_post_tts_cooldown_ms", 900)
        self.session_timeout_s = float(self.threshold_cfg.get("session_timeout_s", 30))
        self.session_max_runtime_s = float(
            self.threshold_cfg.get("session_max_runtime_s", 300)
        )
        self.detector = build_detector(
            perception.get("detector_backend", "mock"),
            models["grounding_dino_id"],
            device=device,
        )
        self.scene_detector_threshold = float(
            perception.get("scene_detector_threshold", 0.28)
        )
        self.fallback_detector = MockDetector()
        self.depth_estimator = build_depth_estimator(
            perception.get("depth_backend", "mock"),
            models["depth_anything_id"],
            device=device,
        )
        self.hand_estimator = build_hand_estimator(
            perception.get("hand_backend", "mediapipe")
        )
        self.training_free_prompt_adapter = TrainingFreePromptAdapter()

        self.overlay_enabled = app.get("enable_overlay", False)
        self.camera_preview_enabled = app.get("enable_camera_preview", False)
        self.camera_preview_title = app.get(
            "camera_preview_title", "SmartReach-HSM Camera"
        )
        self.camera_preview_fps = float(app.get("camera_preview_fps", 20.0))
        self.save_overlay = app.get("save_overlay", False)
        self.overlay_dir = Path(app.get("overlay_dir", "outputs/overlays"))
        self._camera_preview_status_lines: list[str] = []
        self._last_camera_preview_at = 0.0

    def run(
        self,
        text_command: str | None = None,
        audio_path: str | None = None,
        image_path: str | None = None,
        session_steps: int = 8,
    ) -> None:
        print("[APP] Starting SmartReach-HSM...")
        interactive_mode = text_command is None and audio_path is None
        next_text_command = text_command
        next_audio_path = audio_path
        try:
            self._start_camera_stream(image_path=image_path)
            while True:
                self._wait_for_post_tts_cooldown(interactive_mode)
                asr_result = self.asr_service.listen_once(
                    preset_text=next_text_command,
                    audio_path=next_audio_path,
                    progress_callback=self._refresh_camera_preview,
                )
                print(
                    "[APP] User text:",
                    {"text": asr_result.text, "confidence": asr_result.confidence},
                )
                if self._should_request_repeat(asr_result, interactive_mode):
                    self.tts_service.speak("我没有听清，请再说一遍。")
                    if not interactive_mode:
                        return
                    next_text_command = None
                    next_audio_path = None
                    continue
                self._run_dialogue_turn(
                    user_text=asr_result.text,
                    image_path=image_path,
                )

                if not interactive_mode:
                    return
                next_text_command = None
                next_audio_path = None
        finally:
            self._close_camera_preview()
            self.camera_service.stop_stream()

    def _run_dialogue_turn(
        self,
        user_text: str,
        image_path: str | None = None,
    ) -> None:
        print("[APP] Capturing dialogue frame...")
        frame = self.capture_controller.get_initial_frame(image_path=image_path)
        self._publish_camera_preview(
            frame.rgb_image,
            [f"Mode: dialogue", f"Frame: {frame.frame_id}"],
        )
        scene_detections = self._detect_scene_candidates(frame.rgb_image)
        focus_detection = None
        depth = self.depth_estimator.predict_depth(frame.rgb_image)
        hand_pose = self.hand_estimator.estimate_hand(frame.rgb_image)
        fusion_state = build_fusion_state(
            frame,
            focus_detection,
            depth.depth_map,
            hand_pose,
            target_locked=False,
        )
        self._maybe_render_dialogue_overlay(
            frame.rgb_image,
            scene_detections,
            depth.depth_map,
            hand_pose,
            fusion_state,
        )
        detection_payload = build_detection_context_payload(
            focus_detection, scene_detections
        )
        hand_pose_payload = build_hand_pose_payload(hand_pose, fusion_state)
        fusion_payload = build_fusion_payload(fusion_state)
        spatial_hint_text = build_spatial_hint_text(fusion_state)
        conversation_history = self.session_manager.render_history(
            max_turns=self.vlm_cfg.get("conversation_history_turns", 6)
        )
        training_free_prefix = self._build_training_free_prefix(
            mode="dialogue_turn",
            context={
                "user_text": user_text,
                "conversation_history": conversation_history,
                "detection_payload": detection_payload,
                "hand_pose_payload": hand_pose_payload,
                "fusion_payload": fusion_payload,
                "spatial_hint_text": spatial_hint_text,
            },
        )
        prompt = build_glm_dialogue_text(
            user_text=user_text,
            conversation_history=conversation_history,
            detection_payload=detection_payload,
            hand_pose_payload=hand_pose_payload,
            fusion_payload=fusion_payload,
            spatial_hint_text=spatial_hint_text,
            training_free_prefix=training_free_prefix,
        )
        self._log_vlm_request(
            mode="dialogue_turn",
            event="dialogue",
            timestamp_ms=frame.timestamp_ms,
            rgb_image=frame.rgb_image,
            depth_map=depth.depth_map,
            prompt=prompt,
            prompt_payload={
                "user_text": user_text,
                "conversation_history": conversation_history,
                "detection": detection_payload,
                "hand_pose": hand_pose_payload,
                "fusion": fusion_payload,
                "spatial_hint_text": spatial_hint_text,
            },
        )
        answer = self._await_multimodal_reply(
            frame.rgb_image,
            depth.depth_map,
            prompt,
        )
        print(f"[VLM] {answer}")
        self.tts_service.speak(answer)
        self.session_manager.append_turn("user", user_text)
        self.session_manager.append_turn("assistant", answer)

    def _collect_detection_context(
        self, rgb_image, preferred_prompt: str = ""
    ) -> tuple[Detection | None, list[Detection]]:
        scene_detections = self._detect_scene_candidates(
            rgb_image, preferred_prompt=preferred_prompt
        )
        focus_detection = self._select_scene_focus_detection(
            rgb_image,
            scene_detections,
            preferred_prompt=preferred_prompt,
        )
        if (
            preferred_prompt == ""
            and focus_detection is not None
            and focus_detection.label.strip().lower() in {"table", "desk"}
        ):
            focus_detection = next(
                (
                    detection
                    for detection in scene_detections
                    if detection.label.strip().lower() not in {"table", "desk"}
                ),
                focus_detection,
            )
        return focus_detection, scene_detections

    def _await_multimodal_reply(self, rgb_image, depth_map, prompt: str) -> str:
        wait_interval_s = self.scene_wait_interval_ms / 1000.0
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(
                self._generate_multimodal_reply, rgb_image, depth_map, prompt
            )
            while True:
                try:
                    narration = future.result(timeout=wait_interval_s)
                    break
                except TimeoutError:
                    self._refresh_camera_preview()
                    self.tts_service.speak("请等待")
                except Exception as exc:
                    print(f"[VLM] failed: {exc}")
                    return "我暂时无法回答这个问题。"
        answer = narration.scene_description.strip()
        if not narration.speak or not answer:
            return "我暂时无法回答这个问题。"
        return answer

    def _generate_multimodal_reply(self, rgb_image, depth_map, prompt: str):
        return self.vlm_service.analyze_multimodal(rgb_image, depth_map, prompt)

    def _run_grasp_session(
        self,
        command: CommandSpec,
        image_path: str | None,
        session_steps: int,
    ) -> None:
        self.session_manager.start(command)
        print("[APP] Capturing initial frame...")
        initial_frame = self.capture_controller.get_initial_frame(image_path=image_path)
        self._publish_camera_preview(
            initial_frame.rgb_image,
            [f"Mode: grasp", f"Frame: {initial_frame.frame_id}"],
        )
        initial_detection = self._detect_target(initial_frame.rgb_image, command, None)
        if initial_detection is None:
            self.tts_service.speak("我暂时没有看到目标物体，请把目标移到镜头前。")
            self.session_manager.stop(success=False)
            return
        print(
            f"[APP] Initial target detected: {initial_detection.label} score={initial_detection.score:.3f}"
        )
        self.command_player.play_if_changed("target found", initial_frame.timestamp_ms)
        initial_depth = self.depth_estimator.predict_depth(initial_frame.rgb_image)
        initial_hand_pose = self.hand_estimator.estimate_hand(initial_frame.rgb_image)
        self._maybe_vlm_narrate(
            event="session_start",
            rgb_image=initial_frame.rgb_image,
            detection=initial_detection,
            depth_map=initial_depth.depth_map,
            hand_pose=initial_hand_pose,
            fusion_state=FusionState(
                target_visible=True,
                hand_visible=initial_hand_pose.hand_present,
                target_locked=True,
                target_center_xy=(
                    (initial_detection.bbox_xyxy[0] + initial_detection.bbox_xyxy[2])
                    / 2,
                    (initial_detection.bbox_xyxy[1] + initial_detection.bbox_xyxy[3])
                    / 2,
                ),
                hand_center_xy=None,
                target_depth=None,
                hand_depth=None,
                dx_norm=None,
                dy_norm=None,
                dz_rel=None,
                target_confidence=initial_detection.score,
                hand_confidence=initial_hand_pose.score,
                frame_id=initial_frame.frame_id,
                timestamp_ms=initial_frame.timestamp_ms,
                frame_size_xy=(
                    initial_frame.rgb_image.shape[1],
                    initial_frame.rgb_image.shape[0],
                ),
            ),
            control_state=ControlState(name="TARGET_ACQUIRED"),
            command=command,
            guidance=GuidanceCommand(
                command="target found", reason="initial detection"
            ),
            timestamp_ms=initial_frame.timestamp_ms,
        )

        previous_state: FusionState | None = None
        previous_box = initial_detection.bbox_xyxy
        control_state: ControlState | None = None
        previous_control_name = ""
        capture_interval_s = self.threshold_cfg["capture_interval_ms"] / 1000.0
        timeout_steps = max(1, int(self.session_max_runtime_s / capture_interval_s))
        max_steps = max(session_steps, timeout_steps)
        start_time = time.monotonic()
        last_tracking_at = start_time

        for _ in range(max_steps):
            now = time.monotonic()
            frame = self.capture_controller.get_session_frame(image_path=image_path)
            self._publish_camera_preview(
                frame.rgb_image,
                [f"Mode: grasp", f"Frame: {frame.frame_id}"],
            )
            detection = self._detect_target(frame.rgb_image, command, previous_box)
            if detection is not None:
                previous_box = detection.bbox_xyxy
            depth = self.depth_estimator.predict_depth(frame.rgb_image)
            hand_pose = self.hand_estimator.estimate_hand(frame.rgb_image)
            raw_state = build_fusion_state(
                frame,
                detection,
                depth.depth_map,
                hand_pose,
                target_locked=detection is not None,
            )
            fusion_state = smooth_state(raw_state, previous_state)
            previous_state = fusion_state
            if fusion_state.target_visible or fusion_state.hand_visible:
                last_tracking_at = now
            timeout_message = self._should_timeout_grasp_session(
                now, start_time, last_tracking_at, fusion_state
            )
            if timeout_message:
                self.tts_service.speak(timeout_message)
                self.session_manager.stop(success=False)
                return
            control_state = next_state(control_state, fusion_state, self.threshold_cfg)
            command_out = decide_command(
                control_state, fusion_state, self.threshold_cfg
            )
            command_out = enforce_safety(command_out, fusion_state, self.threshold_cfg)
            print(
                f"[APP] state={control_state.name} cmd={command_out.command} "
                f"target={fusion_state.target_visible} hand={fusion_state.hand_visible}"
            )
            command_changed = command_out.command != self.command_player.last_command
            self.command_player.play_if_changed(command_out.command, frame.timestamp_ms)
            if (
                self.guidance_every_frame
                or control_state.name != previous_control_name
                or command_changed
            ):
                event = (
                    "frame_guidance"
                    if self.guidance_every_frame
                    else self._map_state_event(control_state.name)
                )
                self._maybe_vlm_narrate(
                    event=event,
                    rgb_image=frame.rgb_image,
                    detection=detection,
                    depth_map=depth.depth_map,
                    hand_pose=hand_pose,
                    fusion_state=fusion_state,
                    control_state=control_state,
                    command=command,
                    guidance=command_out,
                    timestamp_ms=frame.timestamp_ms,
                )
                previous_control_name = control_state.name
            self._maybe_render_overlay(
                frame.rgb_image,
                detection,
                depth.depth_map,
                hand_pose,
                fusion_state,
                control_state.name,
                command_out,
            )
            if (
                command_out.command == "grasp"
                and control_state.stable_frames
                >= self.threshold_cfg["grasp_stable_frames"]
            ):
                self.session_manager.stop(success=True)
                return
            time.sleep(capture_interval_s)

    def _should_timeout_grasp_session(
        self,
        now_s: float,
        start_time_s: float,
        last_tracking_s: float,
        fusion_state: FusionState,
    ) -> str | None:
        if now_s - start_time_s >= self.session_max_runtime_s:
            return "抓取持续时间过长，请重新开始本轮操作。"
        if fusion_state.target_visible or fusion_state.hand_visible:
            return None
        if now_s - last_tracking_s >= self.session_timeout_s:
            return "长时间没有有效跟踪，请重新调整手和目标后再试一次。"
        return None

        self.session_manager.stop(success=False)

    def _wait_for_post_tts_cooldown(self, interactive_mode: bool) -> None:
        if not interactive_mode:
            return
        remaining_ms = (
            self.post_tts_cooldown_ms - self.tts_service.time_since_last_speak_ms()
        )
        if remaining_ms > 0:
            time.sleep(remaining_ms / 1000.0)

    def _should_request_repeat(self, asr_result, interactive_mode: bool) -> bool:
        if not interactive_mode and asr_result.text:
            return False
        if not asr_result.text.strip():
            return True
        return asr_result.confidence < self.asr_min_confidence

    def _run_scene_query(
        self, command: CommandSpec, image_path: str | None = None
    ) -> None:
        print("[APP] Capturing scene query frame...")
        frame = self.capture_controller.get_initial_frame(image_path=image_path)
        self._publish_camera_preview(
            frame.rgb_image,
            [f"Mode: scene query", f"Frame: {frame.frame_id}"],
        )
        _, preferred_prompt = self._extract_scene_query_target(command.query_text)
        scene_detections = self._detect_scene_candidates(
            frame.rgb_image, preferred_prompt=preferred_prompt
        )
        detection = self._select_scene_focus_detection(
            frame.rgb_image, scene_detections, preferred_prompt=preferred_prompt
        )
        depth = self.depth_estimator.predict_depth(frame.rgb_image)
        hand_pose = self.hand_estimator.estimate_hand(frame.rgb_image)
        fusion_state = build_fusion_state(
            frame,
            detection,
            depth.depth_map,
            hand_pose,
            target_locked=detection is not None,
        )
        dashboard = render_overlay(
            frame.rgb_image,
            detection,
            depth.depth_map,
            hand_pose,
            fusion_state,
            "scene query",
            "SCENE_QUERY",
        )
        self._publish_overlay(dashboard, fusion_state.frame_id)
        detection_payload = build_detection_payload(detection)
        scene_detections_payload = build_detection_payloads(scene_detections)
        hand_pose_payload = build_hand_pose_payload(hand_pose, fusion_state)
        fusion_payload = build_fusion_payload(fusion_state)
        spatial_hint_text = build_spatial_hint_text(fusion_state)
        conversation_history = self.session_manager.render_history(
            max_turns=self.vlm_cfg.get("conversation_history_turns", 6)
        )
        training_free_prefix = self._build_training_free_prefix(
            mode="scene_query",
            context={
                "question": command.query_text,
                "conversation_history": conversation_history,
                "detection_payload": detection_payload,
                "scene_detections_payload": scene_detections_payload,
                "hand_pose_payload": hand_pose_payload,
                "fusion_payload": fusion_payload,
                "spatial_hint_text": spatial_hint_text,
            },
        )
        prompt = build_glm_scene_query_text(
            question=command.query_text,
            conversation_history=conversation_history,
            detection_payload=detection_payload,
            scene_detections_payload=scene_detections_payload,
            hand_pose_payload=hand_pose_payload,
            fusion_payload=fusion_payload,
            spatial_hint_text=spatial_hint_text,
            training_free_prefix=training_free_prefix,
        )
        self._log_vlm_request(
            mode="scene_query",
            event="scene_query",
            timestamp_ms=frame.timestamp_ms,
            rgb_image=frame.rgb_image,
            depth_map=depth.depth_map,
            prompt=prompt,
            prompt_payload={
                "question": command.query_text,
                "conversation_history": conversation_history,
                "focus_detection": detection_payload,
                "scene_detections": scene_detections_payload,
                "hand_pose": hand_pose_payload,
                "fusion": fusion_payload,
                "spatial_hint_text": spatial_hint_text,
            },
        )
        answer = self._await_scene_query_answer(
            frame.rgb_image,
            depth.depth_map,
            prompt,
        )
        answer = self._refine_scene_query_answer(
            answer=answer,
            question=command.query_text,
            detection=detection,
            detections=scene_detections,
            hand_pose=hand_pose,
            fusion_state=fusion_state,
        )
        print(f"[VLM-QA] {answer}")
        self.tts_service.speak(answer)
        self.session_manager.append_turn("user", command.query_text)
        self.session_manager.append_turn("assistant", answer)

    def _detect_target(
        self,
        rgb_image,
        command: CommandSpec,
        previous_box: tuple[float, float, float, float] | None,
    ) -> Detection | None:
        detections = self.detector.detect(rgb_image, command.target_prompt_en)
        if not detections:
            print(
                "[APP] Real detector returned no candidates, trying local fallback detector..."
            )
            detections = self.fallback_detector.detect(
                rgb_image, command.target_prompt_en
            )
        height, width = rgb_image.shape[:2]
        return select_target(
            detections,
            command.spatial_hint,
            (width, height),
            previous_box,
            score_weights=self.threshold_cfg,
        )

    def _detect_scene_candidates(
        self, rgb_image, preferred_prompt: str = ""
    ) -> list[Detection]:
        prompt = self.scene_query_detection_prompt
        if preferred_prompt and preferred_prompt not in prompt:
            prompt = f"{preferred_prompt} . {prompt}"
        detections = self.detector.detect(
            rgb_image, prompt, threshold=self.scene_detector_threshold
        )
        if not detections:
            detections = self.fallback_detector.detect(rgb_image, prompt)
        return detections

    def _select_scene_focus_detection(
        self,
        rgb_image,
        scene_detections: list[Detection],
        preferred_prompt: str = "",
    ) -> Detection | None:
        height, width = rgb_image.shape[:2]
        if preferred_prompt:
            detections = self.detector.detect(
                rgb_image,
                preferred_prompt,
                threshold=self.scene_detector_threshold,
            )
            if not detections:
                detections = self.fallback_detector.detect(rgb_image, preferred_prompt)
            if detections:
                return select_target(
                    detections,
                    "none",
                    (width, height),
                    None,
                    score_weights=self.threshold_cfg,
                )
        if not scene_detections:
            return None
        return select_target(
            scene_detections,
            "none",
            (width, height),
            None,
            score_weights=self.threshold_cfg,
        )

    def _await_scene_query_answer(self, rgb_image, depth_map, prompt: str) -> str:
        wait_interval_s = self.scene_wait_interval_ms / 1000.0
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(
                self._generate_scene_query_answer, rgb_image, depth_map, prompt
            )
            while True:
                try:
                    narration = future.result(timeout=wait_interval_s)
                    break
                except TimeoutError:
                    self._refresh_camera_preview()
                    self.tts_service.speak("请等待")
                except Exception as exc:
                    print(f"[VLM-QA] failed: {exc}")
                    return "我暂时无法回答这个问题。"
        answer = narration.scene_description.strip()
        if not narration.speak or not answer:
            return "我暂时无法回答这个问题。"
        return answer

    def _generate_scene_query_answer(self, rgb_image, depth_map, prompt: str):
        narration = self.vlm_service.analyze_multimodal(rgb_image, depth_map, prompt)
        answer = narration.scene_description.strip()
        if narration.speak and answer:
            narration.scene_description = self._match_scene_query_language(
                answer, prompt
            )
            narration.scene_description = self._force_chinese_output(
                narration.scene_description
            )
        return narration

    def _force_chinese_output(self, text: str) -> str:
        cleaned = text.strip()
        if not cleaned:
            return cleaned
        if not any("A" <= char <= "Z" or "a" <= char <= "z" for char in cleaned):
            return cleaned
        parser = getattr(self, "intent_parser", None)
        translator = getattr(parser, "translate_text", None)
        if translator is None:
            return cleaned
        translated = translator(cleaned, "Simplified Chinese")
        return translated.strip() or cleaned

    def _match_scene_query_language(self, answer: str, prompt: str) -> str:
        if not answer.strip():
            return answer
        target_language = (
            "Simplified Chinese"
            if "User question:" in prompt
            and any(
                "\u4e00" <= char <= "\u9fff"
                for char in prompt.split("User question:", 1)[1]
            )
            else "English"
        )
        answer_has_cjk = any("\u4e00" <= char <= "\u9fff" for char in answer)
        if target_language == "Simplified Chinese" and answer_has_cjk:
            return answer
        if target_language == "English" and not answer_has_cjk:
            return answer
        parser = getattr(self, "intent_parser", None)
        translator = getattr(parser, "translate_text", None)
        if translator is None:
            return answer
        translated = translator(answer, target_language)
        return translated.strip() or answer

    def _refine_scene_query_answer(
        self,
        answer: str,
        question: str,
        detection: Detection | None,
        detections: list[Detection],
        hand_pose,
        fusion_state: FusionState,
    ) -> str:
        cleaned_answer = answer.strip()
        fallback = self._build_scene_query_fallback(
            question=question,
            detection=detection,
            detections=detections,
            hand_pose=hand_pose,
            fusion_state=fusion_state,
        )
        if not cleaned_answer:
            return fallback or "我暂时无法回答这个问题。"
        if self._is_low_value_scene_answer(cleaned_answer, question, detections):
            return fallback or cleaned_answer
        return cleaned_answer

    @staticmethod
    def _is_generic_scene_question(question: str) -> bool:
        lowered = question.lower()
        return any(
            keyword in question
            for keyword in [
                "前面有什么",
                "你看到了什么",
                "看到什么",
                "看见什么",
                "能看到什么",
                "什么内容",
                "有什么东西",
                "桌面上有什么",
                "桌上有什么",
                "桌子上有什么",
                "帮我看看",
                "描述一下",
            ]
        ) or any(
            keyword in lowered
            for keyword in [
                "what do you see",
                "what is in front of me",
                "describe the scene",
                "look at",
            ]
        )

    @staticmethod
    def _is_distance_scene_question(question: str) -> bool:
        lowered = question.lower()
        return any(keyword in question for keyword in ["多远", "距离", "多近"]) or any(
            keyword in lowered for keyword in ["distance", "how far", "how close"]
        )

    @staticmethod
    def _is_motion_scene_question(question: str) -> bool:
        lowered = question.lower()
        return any(
            keyword in question
            for keyword in [
                "怎么移动",
                "怎么动",
                "如何移动",
                "如何动",
                "怎样移动",
                "怎样动",
                "怎么拿",
                "如何拿",
                "怎样拿",
                "能拿到",
                "能够拿到",
                "才能拿到",
            ]
        ) or any(
            keyword in lowered
            for keyword in ["how should i move", "how do i move", "how to grab"]
        )

    @staticmethod
    def _question_prefers_chinese(question: str) -> bool:
        return any("\u4e00" <= char <= "\u9fff" for char in question)

    def _localize_scene_label(self, label: str, question: str) -> str:
        zh_map = {
            "cup": "杯子",
            "bottle": "瓶子",
            "plastic bottle": "塑料瓶",
            "water bottle": "水瓶",
            "can": "易拉罐",
            "coke": "可乐罐",
            "coke can": "可乐罐",
            "cola can": "可乐罐",
            "red soda can": "红色可乐罐",
            "red soda coke": "红色可乐罐",
            "milk carton": "牛奶盒",
            "green bottle": "绿色瓶子",
            "box": "盒子",
            "tissue": "纸巾",
            "tissue box": "纸巾盒",
            "tissue pack": "纸巾包",
            "paper towel": "纸巾",
            "mouse": "鼠标",
            "computer mouse": "鼠标",
            "keyboard": "键盘",
            "phone": "手机",
            "book": "书",
            "table": "桌面",
        }
        if self._question_prefers_chinese(question):
            return zh_map.get(label.lower(), label)
        return label

    def _extract_scene_query_target(self, question: str) -> tuple[str, str]:
        lowered = question.lower().strip()
        object_map = self.prompt_cfg.get("object_map", {})
        for payload in object_map.values():
            keywords = payload.get("keywords", [])
            if any(keyword.lower() in lowered for keyword in keywords):
                prompts = payload.get("prompts", [])
                prompt = prompts[0] if prompts else ""
                matched_keyword = next(
                    (keyword for keyword in keywords if keyword.lower() in lowered),
                    "",
                )
                return matched_keyword, prompt
        return "", ""

    def _is_low_value_scene_answer(
        self, answer: str, question: str, detections: list[Detection]
    ) -> bool:
        lowered = answer.lower().strip()
        if not lowered:
            return True
        if any(
            token in lowered for token in ["无法回答", "无法确定", "不确定", "不清楚"]
        ):
            return True
        if not self._is_generic_scene_question(question):
            return False
        if not self._scene_has_non_table_objects(detections):
            return False
        return any(token in lowered for token in ["桌面", "桌子", "table", "desk"])

    @staticmethod
    def _scene_has_non_table_objects(detections: list[Detection]) -> bool:
        return any(
            detection.label.strip().lower() not in {"table", "desk"}
            for detection in detections
        )

    def _build_scene_query_fallback(
        self,
        question: str,
        detection: Detection | None,
        detections: list[Detection],
        hand_pose,
        fusion_state: FusionState,
    ) -> str:
        if detection is not None:
            target_name = self._localize_scene_label(detection.label, question)
            if self._is_motion_scene_question(question):
                relation_text = self._describe_target_relation(
                    target_name, fusion_state
                )
                move_text = self._describe_scene_query_hand_adjustment(fusion_state)
                if move_text:
                    return f"{relation_text}{move_text}"
                return relation_text
            if self._is_distance_scene_question(question):
                return self._describe_target_relation(target_name, fusion_state)
        if self._is_generic_scene_question(question):
            summary = self._summarize_scene_detections(detections, question)
            if summary:
                return summary
        if detection is not None:
            localized_label = self._localize_scene_label(detection.label, question)
            if self._question_prefers_chinese(question):
                return f"前方有一个{localized_label}。"
            return f"There is a {localized_label} in front of you."
        if hand_pose.hand_present:
            if self._question_prefers_chinese(question):
                return "我能看到你的手。"
            return "I can see your hand."
        return ""

    def _summarize_scene_detections(
        self, detections: list[Detection], question: str
    ) -> str:
        labels: list[str] = []
        seen: set[str] = set()
        for detection in detections:
            localized = self._localize_scene_label(detection.label, question)
            if localized in {"桌面", "table"} and self._scene_has_non_table_objects(
                detections
            ):
                continue
            label_key = localized.lower()
            if label_key in seen:
                continue
            seen.add(label_key)
            labels.append(localized)
            if len(labels) >= 4:
                break
        if not labels:
            return ""
        if self._question_prefers_chinese(question):
            return f"桌面上有{'、'.join(labels)}。"
        return f"On the table there are {', '.join(labels)}."

    def _describe_scene_query_hand_adjustment(self, fusion_state: FusionState) -> str:
        if not fusion_state.hand_visible:
            return "请先把手移到镜头里，再朝目标方向靠近。"
        if fusion_state.dx_norm is None or fusion_state.dy_norm is None:
            return "请缓慢把手朝目标移动。"
        if fusion_state.dx_norm < -0.05:
            return "请把手向左移动一点。"
        if fusion_state.dx_norm > 0.05:
            return "请把手向右移动一点。"
        if fusion_state.dy_norm < -0.05:
            return "请把手向上移动一点。"
        if fusion_state.dy_norm > 0.05:
            return "请把手向下移动一点。"
        if fusion_state.dz_rel is not None:
            if fusion_state.dz_rel > 0.08:
                return "请把手向前伸一点。"
            if fusion_state.dz_rel < -0.08:
                return "请把手向后回一点。"
        return "继续缓慢靠近，接近后就可以抓取。"

    def _build_scene_query_detection_prompt(self) -> str:
        candidates: list[str] = []
        object_map = self.prompt_cfg.get("object_map", {})
        for payload in object_map.values():
            for prompt in payload.get("prompts", []):
                prompt_text = str(prompt).strip()
                if prompt_text and prompt_text not in candidates:
                    candidates.append(prompt_text)
        for fallback_label in [
            "tissue box",
            "tissue pack",
            "paper towel",
            "mouse",
            "computer mouse",
            "water bottle",
            "cup",
            "bottle",
            "can",
            "box",
            "phone",
            "book",
            "keyboard",
            "table",
        ]:
            if fallback_label not in candidates:
                candidates.append(fallback_label)
        return " . ".join(candidates)

    @staticmethod
    def _map_state_event(state_name: str) -> str:
        if state_name == "WAIT_HAND":
            return "wait_hand"
        if state_name == "SEARCH_TARGET":
            return "target_lost"
        if state_name == "GRASP_READY":
            return "grasp_ready"
        return "state_change"

    def _maybe_vlm_narrate(
        self,
        event: str,
        rgb_image,
        detection,
        depth_map,
        hand_pose,
        fusion_state: FusionState,
        control_state: ControlState,
        command: CommandSpec,
        guidance: GuidanceCommand,
        timestamp_ms: int,
    ) -> None:
        if not self.vlm_cfg.get("enabled", False):
            return
        fallback_text = self._templated_narration(
            event, command, guidance, fusion_state
        )
        if fallback_text and not self.guidance_every_frame:
            if self.narration_player.can_play(
                fallback_text,
                timestamp_ms,
                guidance.command,
                self.command_player,
            ):
                print(f"[VLM-TEMPLATE] {fallback_text}")
                self.narration_player.play(fallback_text, timestamp_ms)
            return
        if event == "wait_hand" and not self.vlm_cfg.get("trigger_wait_hand", True):
            return
        if event == "target_lost" and not self.vlm_cfg.get("trigger_loss", True):
            return
        if event == "state_change" and not self.vlm_cfg.get(
            "trigger_session_start", True
        ):
            return
        detection_payload = build_detection_payload(detection)
        hand_pose_payload = build_hand_pose_payload(hand_pose, fusion_state)
        fusion_payload = build_fusion_payload(fusion_state)
        spatial_hint_text = build_spatial_hint_text(fusion_state)
        conversation_history = self.session_manager.render_history(
            max_turns=self.vlm_cfg.get("conversation_history_turns", 6)
        )
        training_free_prefix = self._build_training_free_prefix(
            mode="grasp_guidance",
            context={
                "event": event,
                "question": command.query_text,
                "conversation_history": conversation_history,
                "target_name": command.target_name_raw or command.target_prompt_en,
                "guidance_command": guidance.command,
                "detection_payload": detection_payload,
                "hand_pose_payload": hand_pose_payload,
                "fusion_payload": fusion_payload,
                "spatial_hint_text": spatial_hint_text,
            },
        )
        prompt = build_glm_guidance_text(
            event=event,
            conversation_history=conversation_history,
            command=command,
            control_state=control_state,
            guidance=guidance,
            detection_payload=detection_payload,
            hand_pose_payload=hand_pose_payload,
            fusion_payload=fusion_payload,
            spatial_hint_text=spatial_hint_text,
            training_free_prefix=training_free_prefix,
        )
        self._log_vlm_request(
            mode="grasp_guidance",
            event=event,
            timestamp_ms=timestamp_ms,
            rgb_image=rgb_image,
            depth_map=depth_map,
            prompt=prompt,
            prompt_payload={
                "target_name_raw": command.target_name_raw,
                "target_prompt_en": command.target_prompt_en,
                "conversation_history": conversation_history,
                "spatial_hint": command.spatial_hint,
                "control_state": control_state.name,
                "rule_command": guidance.command,
                "rule_reason": guidance.reason,
                "detection": detection_payload,
                "hand_pose": hand_pose_payload,
                "fusion": fusion_payload,
                "spatial_hint_text": spatial_hint_text,
            },
        )
        try:
            narration = self.vlm_service.analyze_multimodal(
                rgb_image, depth_map, prompt
            )
        except Exception as exc:
            print(f"[VLM] narration failed: {exc}")
            if fallback_text:
                print(f"[VLM-TEMPLATE] {fallback_text}")
                self.narration_player.play(fallback_text, timestamp_ms)
            return
        if not narration.speak:
            if fallback_text:
                print(f"[VLM-TEMPLATE] {fallback_text}")
                self.narration_player.play(fallback_text, timestamp_ms)
            return
        narration.scene_description = self._refine_grasp_guidance(
            narration.scene_description,
            command,
            guidance,
            fusion_state,
            fallback_text,
        )
        if not narration.scene_description.strip():
            return
        if self.guidance_every_frame:
            narration.scene_description = self._force_chinese_output(
                narration.scene_description
            )
            print(f"[VLM] {narration.scene_description}")
            self.narration_player.play(narration.scene_description, timestamp_ms)
            return
        if not self.narration_player.can_play(
            narration.scene_description,
            timestamp_ms,
            guidance.command,
            self.command_player,
        ):
            return
        narration.scene_description = self._force_chinese_output(
            narration.scene_description
        )
        print(f"[VLM] {narration.scene_description}")
        self.narration_player.play(narration.scene_description, timestamp_ms)

    def _refine_grasp_guidance(
        self,
        text: str,
        command: CommandSpec,
        guidance: GuidanceCommand,
        fusion_state: FusionState,
        fallback_text: str,
    ) -> str:
        cleaned = self._force_chinese_output(text)
        if not cleaned.strip():
            return fallback_text
        if not fusion_state.hand_visible and fallback_text:
            return fallback_text
        forbidden = [
            "背景",
            "房间",
            "卧室",
            "窗帘",
            "床",
            "英文",
            "hold",
            "left",
            "right",
        ]
        if any(token in cleaned for token in forbidden):
            return fallback_text
        expected_tokens = {
            "place hand in view": ["请", "镜头"],
            "left": ["请", "左"],
            "right": ["请", "右"],
            "up": ["请", "上"],
            "down": ["请", "下"],
            "forward": ["请", "前"],
            "back": ["请", "后"],
            "hold": ["保持"],
            "grasp": ["抓"],
        }
        required_tokens = expected_tokens.get(guidance.command, ["请"])
        if not all(token in cleaned for token in required_tokens):
            return fallback_text
        if len(cleaned) > 60:
            return fallback_text
        return cleaned

    def _log_vlm_request(
        self,
        mode: str,
        event: str,
        timestamp_ms: int,
        rgb_image,
        depth_map,
        prompt: str,
        prompt_payload: dict[str, object],
    ) -> None:
        self._print_vlm_prompt(mode=mode, event=event, prompt=prompt)
        if not getattr(self, "log_vlm_inputs", False):
            return
        output_dir = getattr(self, "vlm_input_dir", Path("outputs/vlm_inputs"))
        output_dir.mkdir(parents=True, exist_ok=True)
        safe_event = event.replace(" ", "_")
        stem = f"{timestamp_ms}_{mode}_{safe_event}"
        rgb_path = output_dir / f"{stem}_rgb.png"
        depth_path = output_dir / f"{stem}_depth.png"
        metadata_path = output_dir / f"{stem}.json"
        cv2.imwrite(str(rgb_path), cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(depth_path), self._normalize_depth_for_logging(depth_map))
        metadata = {
            "mode": mode,
            "event": event,
            "timestamp_ms": timestamp_ms,
            "model_id": self.vlm_cfg.get("model_id", ""),
            "backend": self.vlm_cfg.get("backend", ""),
            "rgb_image": str(rgb_path),
            "depth_image": str(depth_path),
            "prompt": prompt,
            "prompt_payload": prompt_payload,
        }
        metadata_path.write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"[VLM-INPUT] {metadata_path}")

    @staticmethod
    def _print_vlm_prompt(mode: str, event: str, prompt: str) -> None:
        print(f"[VLM-PROMPT] mode={mode} event={event}")
        print(prompt)
        print("[VLM-PROMPT-END]")

    @staticmethod
    def _normalize_depth_for_logging(depth_map) -> np.ndarray:
        valid = depth_map[np.isfinite(depth_map)]
        if valid.size == 0:
            return np.zeros_like(depth_map, dtype=np.uint8)
        min_val = float(valid.min())
        max_val = float(valid.max())
        scale = max(max_val - min_val, 1e-6)
        return np.clip((depth_map - min_val) / scale * 255.0, 0, 255).astype(np.uint8)

    def _templated_narration(
        self,
        event: str,
        command: CommandSpec,
        guidance: GuidanceCommand,
        fusion_state: FusionState,
    ) -> str:
        target_name = command.target_name_raw or command.target_prompt_en or "目标物体"
        command_text = guidance.command
        relation_text = self._describe_target_relation(target_name, fusion_state)
        if command_text == "place hand in view":
            return f"{relation_text}请先把手移到镜头里，再朝它的方向靠近。"
        if command_text == "left":
            return f"{relation_text}请把手向左移动一点。"
        if command_text == "right":
            return f"{relation_text}请把手向右移动一点。"
        if command_text == "up":
            return f"{relation_text}请把手向上移动一点。"
        if command_text == "down":
            return f"{relation_text}请把手向下移动一点。"
        if command_text == "forward":
            return f"{relation_text}请把手向前伸一点。"
        if command_text == "back":
            return f"{relation_text}请把手向后回一点。"
        if command_text == "hold":
            return f"{relation_text}先保持不动。"
        if command_text == "grasp":
            return f"{relation_text}现在可以抓住它。"
        if command_text == "target found":
            if not fusion_state.hand_visible:
                return f"{relation_text}请先把手移到镜头里，再朝它的方向靠近。"
            return f"{relation_text}请根据我的提示慢慢移动手。"
        if command_text == "target lost" or event == "target_lost":
            return (
                f"我暂时看不到{target_name}，请缓慢移动手或目标，让它重新回到镜头里。"
            )
        if event == "session_start":
            if not fusion_state.hand_visible:
                return ""
            return f"我已经看到{target_name}，请根据我的提示慢慢移动手。"
        if event == "grasp_ready":
            return f"你的手已经接近{target_name}，现在可以慢慢抓取。"
        return ""

    def _describe_target_relation(
        self, target_name: str, fusion_state: FusionState
    ) -> str:
        if not fusion_state.target_visible:
            return f"我暂时看不到{target_name}。"
        if not fusion_state.hand_visible:
            target_position = self._describe_target_position_in_frame(fusion_state)
            if target_position:
                return f"我能看到{target_name}，它目前在画面{target_position}。"
            return f"我能看到{target_name}。"
        if fusion_state.dx_norm is None or fusion_state.dy_norm is None:
            return f"我能看到{target_name}。"
        detail = build_spatial_hint_text(fusion_state)
        if "专业空间描述是：" in detail:
            detail = detail.split("专业空间描述是：", 1)[1]
        detail = detail.split("如果视觉画面存在镜像感", 1)[0].strip("；。 ")
        return f"{target_name}相对你的手{detail}。"

    @staticmethod
    def _describe_target_position_in_frame(fusion_state: FusionState) -> str:
        if (
            fusion_state.target_center_xy is None
            or fusion_state.frame_size_xy is None
            or fusion_state.frame_size_xy[0] <= 0
            or fusion_state.frame_size_xy[1] <= 0
        ):
            return ""
        image_w, image_h = fusion_state.frame_size_xy
        target_x, target_y = fusion_state.target_center_xy
        x_norm = target_x / image_w
        y_norm = target_y / image_h
        horizontal = "中央"
        vertical = ""
        if x_norm < 0.35:
            horizontal = "左侧"
        elif x_norm > 0.65:
            horizontal = "右侧"
        if y_norm < 0.35:
            vertical = "上方"
        elif y_norm > 0.65:
            vertical = "下方"
        if horizontal == "中央" and not vertical:
            return "中央"
        if horizontal == "中央":
            return vertical
        if not vertical:
            return horizontal
        return f"{horizontal}{vertical}"

    def _maybe_render_dialogue_overlay(
        self,
        rgb_image,
        detections,
        depth_map,
        hand_pose,
        fusion_state: FusionState,
    ) -> None:
        if not self.overlay_enabled and not self.save_overlay:
            return
        frame = render_overlay(
            rgb_image,
            detections,
            depth_map,
            hand_pose,
            fusion_state,
            "dialogue",
            "DIALOGUE",
        )
        self._publish_overlay(frame, fusion_state.frame_id)

    def _maybe_render_overlay(
        self,
        rgb_image,
        detection,
        depth_map,
        hand_pose,
        fusion_state,
        state_name: str,
        command_out: GuidanceCommand,
    ) -> None:
        if not self.overlay_enabled and not self.save_overlay:
            return
        frame = render_overlay(
            rgb_image,
            detection,
            depth_map,
            hand_pose,
            fusion_state,
            command_out.command,
            state_name,
        )
        self._publish_overlay(frame, fusion_state.frame_id)

    def _publish_overlay(self, frame, frame_id: int) -> None:
        if self.overlay_enabled:
            cv2.imshow("SmartReach-HSM Multi-Modal Dashboard", frame)
            cv2.waitKey(1)
        if self.save_overlay:
            self.overlay_dir.mkdir(parents=True, exist_ok=True)
            output_path = self.overlay_dir / f"frame_{frame_id:04d}.jpg"
            cv2.imwrite(str(output_path), frame)

    def _build_training_free_prefix(
        self,
        mode: str,
        context: dict[str, object],
    ) -> str:
        adapter = getattr(self, "training_free_prompt_adapter", None)
        if adapter is None:
            return ""
        return adapter.render_prefix(mode=mode, context=context)

    def _publish_camera_preview(
        self,
        rgb_image,
        status_lines: list[str] | None = None,
    ) -> None:
        if not getattr(self, "camera_preview_enabled", False):
            return
        self._camera_preview_status_lines = list(status_lines or [])
        frame = render_camera_preview(
            rgb_image,
            title=getattr(self, "camera_preview_title", "SmartReach-HSM Camera"),
            status_lines=self._camera_preview_status_lines,
        )
        cv2.imshow(
            getattr(self, "camera_preview_title", "SmartReach-HSM Camera"), frame
        )
        cv2.waitKey(1)

    def _start_camera_stream(self, image_path: str | None) -> None:
        if image_path:
            return
        self.camera_service.start_stream()
        self._refresh_camera_preview()

    def _refresh_camera_preview(self) -> None:
        if not self.camera_preview_enabled:
            return
        now = time.monotonic()
        min_interval_s = 1.0 / max(self.camera_preview_fps, 1.0)
        if now - self._last_camera_preview_at < min_interval_s:
            return
        rgb_image = self.camera_service.get_latest_rgb_image()
        if rgb_image is None:
            return
        frame = render_camera_preview(
            rgb_image,
            title=self.camera_preview_title,
            status_lines=self._camera_preview_status_lines,
        )
        cv2.imshow(self.camera_preview_title, frame)
        cv2.waitKey(1)
        self._last_camera_preview_at = now

    def _close_camera_preview(self) -> None:
        if self.camera_preview_enabled:
            try:
                cv2.destroyWindow(self.camera_preview_title)
            except cv2.error:
                pass
