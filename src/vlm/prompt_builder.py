from __future__ import annotations

import json
from src.common.schemas import (
    CommandSpec,
    ControlState,
    Detection,
    FusionState,
    GuidanceCommand,
    HandPoseResult,
)


def _preferred_answer_language(question: str) -> str:
    if any("\u4e00" <= char <= "\u9fff" for char in question):
        return "简体中文"
    return "英文"


def _guidance_rule_explanation(command: str) -> str:
    mapping = {
        "left": "请让用户的手向左移动。",
        "right": "请让用户的手向右移动。",
        "up": "请让用户的手向上移动。",
        "down": "请让用户的手向下移动。",
        "forward": "请让用户的手向前靠近目标。",
        "back": "请让用户的手向后回一点。",
        "hold": "请让用户先保持不动。",
        "grasp": "请明确告诉用户现在可以合拢手指抓取。",
        "place hand in view": "请让用户把手移动到镜头内。",
        "target lost": "请让用户缓慢搜索目标物体。",
        "target found": "请开始给出下一步抓取指导，不要只说目标已找到。",
    }
    return mapping.get(command, "请根据当前状态给出下一步操作指导。")


def _apply_training_free_prefix(base_prompt: str, training_free_prefix: str) -> str:
    if not training_free_prefix.strip():
        return base_prompt
    return f"{training_free_prefix}\n\n{base_prompt}"


def _describe_frame_position(fusion_state: FusionState) -> str:
    if (
        fusion_state.target_center_xy is None
        or fusion_state.frame_size_xy is None
        or fusion_state.frame_size_xy[0] <= 0
        or fusion_state.frame_size_xy[1] <= 0
    ):
        return "中央"
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


def build_spatial_hint_text(fusion_state: FusionState) -> str:
    if not fusion_state.target_visible:
        return (
            "当前没有结构化焦点目标。"
            "请主要根据当前RGB图像、检测结果和手部位置关系判断物体位置与是否已经接触或握住。"
        )
    if not fusion_state.hand_visible:
        return (
            f"当前只检测到目标，目标位于画面{_describe_frame_position(fusion_state)}。"
            "由于当前没有检测到手，不能可靠判断目标相对手的钟点方向。"
        )
    return "当前同时检测到目标和手，请结合图像与结构化结果理解它们的相对位置关系。"


def build_vlm_prompt(
    event: str,
    command: CommandSpec,
    control_state: ControlState,
    fusion_state: FusionState,
    guidance: GuidanceCommand,
) -> str:
    return (
        "You are helping a blind user understand the current grasping scene. "
        "Trust the structured state more than uncertain visual guesses. "
        "Do not output motion commands like left, right, forward, back, or grasp. "
        "Answer with one short sentence under 16 words.\n\n"
        f"Event: {event}\n"
        f"Target object: {command.target_prompt_en}\n"
        f"Spatial hint: {command.spatial_hint}\n"
        f"Control state: {control_state.name}\n"
        f"Rule command: {guidance.command}\n"
        f"Target visible: {fusion_state.target_visible}\n"
        f"Hand visible: {fusion_state.hand_visible}\n"
        f"Target confidence: {fusion_state.target_confidence:.2f}\n"
        f"Hand confidence: {fusion_state.hand_confidence:.2f}\n"
        f"dx_norm: {fusion_state.dx_norm}\n"
        f"dy_norm: {fusion_state.dy_norm}\n"
        f"dz_rel: {fusion_state.dz_rel}\n"
        "Panel legend: top-left camera plus runtime, top-right detection, bottom-left depth, bottom-right hand pose."
    )


def build_scene_query_prompt(
    question: str,
    detection: Detection | None,
    hand_pose: HandPoseResult,
    fusion_state: FusionState,
) -> str:
    answer_language = _preferred_answer_language(question)
    detection_label = detection.label if detection is not None else "none"
    detection_score = f"{detection.score:.2f}" if detection is not None else "0.00"
    target_depth = (
        f"{fusion_state.target_depth:.2f}"
        if fusion_state.target_depth is not None
        else "unknown"
    )
    hand_depth = (
        f"{fusion_state.hand_depth:.2f}"
        if fusion_state.hand_depth is not None
        else "unknown"
    )
    return (
        "You are helping a blind user understand a captured scene. "
        "The image is a four-panel dashboard from one RGB frame: top-left camera plus runtime, "
        "top-right object detection, bottom-left depth map, bottom-right hand pose. "
        "Use the panel content together with the structured hints below. "
        f"Answer the user's question directly in one short sentence. You MUST answer in {answer_language}. "
        "Do not invent objects that are not visible. If the scene is unclear, say so briefly.\n\n"
        f"User question: {question}\n"
        f"Top detection: {detection_label}\n"
        f"Detection confidence: {detection_score}\n"
        f"Hand visible: {hand_pose.hand_present}\n"
        f"Hand confidence: {hand_pose.score:.2f}\n"
        f"Target visible: {fusion_state.target_visible}\n"
        f"Estimated target depth: {target_depth}\n"
        f"Estimated hand depth: {hand_depth}\n"
        "Panel legend: top-left camera plus runtime, top-right detection, bottom-left depth, bottom-right hand pose."
    )


def build_detection_payload(detection: Detection | None) -> dict[str, object]:
    if detection is None:
        return {"visible": False}
    x1, y1, x2, y2 = detection.bbox_xyxy
    return {
        "visible": True,
        "label": detection.label,
        "score": round(detection.score, 4),
        "bbox_xyxy": [round(x1, 2), round(y1, 2), round(x2, 2), round(y2, 2)],
    }


def build_detection_payloads(
    detections: list[Detection], max_items: int | None = None
) -> list[dict[str, object]]:
    payloads: list[dict[str, object]] = []
    selected = detections if max_items is None else detections[:max_items]
    for detection in selected:
        payloads.append(build_detection_payload(detection))
    return payloads


def build_detection_context_payload(
    focus_detection: Detection | None, scene_detections: list[Detection]
) -> dict[str, object]:
    payload = {"scene_candidates": build_detection_payloads(scene_detections)}
    if focus_detection is not None:
        payload["focus_detection"] = build_detection_payload(focus_detection)
    return payload


def build_hand_pose_payload(
    hand_pose: HandPoseResult, fusion_state: FusionState
) -> dict[str, object]:
    landmarks = [
        [round(point[0], 2), round(point[1], 2)]
        for point in hand_pose.landmarks_xy[:21]
    ]
    payload: dict[str, object] = {
        "hand_present": hand_pose.hand_present,
        "score": round(hand_pose.score, 4),
        "handedness": hand_pose.handedness,
        "landmarks_xy": landmarks,
    }
    if fusion_state.hand_center_xy is not None:
        payload["palm_center_xy"] = [
            round(fusion_state.hand_center_xy[0], 2),
            round(fusion_state.hand_center_xy[1], 2),
        ]
    return payload


def build_fusion_payload(fusion_state: FusionState) -> dict[str, object]:
    payload: dict[str, object] = {
        "target_visible": fusion_state.target_visible,
        "hand_visible": fusion_state.hand_visible,
        "target_locked": fusion_state.target_locked,
        "target_confidence": round(fusion_state.target_confidence, 4),
        "hand_confidence": round(fusion_state.hand_confidence, 4),
        "target_depth": fusion_state.target_depth,
        "hand_depth": fusion_state.hand_depth,
        "dx_norm": fusion_state.dx_norm,
        "dy_norm": fusion_state.dy_norm,
        "dz_rel": fusion_state.dz_rel,
    }
    if fusion_state.target_center_xy is not None:
        payload["target_center_xy"] = [
            round(fusion_state.target_center_xy[0], 2),
            round(fusion_state.target_center_xy[1], 2),
        ]
    if fusion_state.hand_center_xy is not None:
        payload["hand_center_xy"] = [
            round(fusion_state.hand_center_xy[0], 2),
            round(fusion_state.hand_center_xy[1], 2),
        ]
    return payload


def build_glm_dialogue_text(
    user_text: str,
    conversation_history: str,
    detection_payload: dict[str, object],
    hand_pose_payload: dict[str, object],
    fusion_payload: dict[str, object] | None = None,
    spatial_hint_text: str = "",
    training_free_prefix: str = "",
) -> str:
    answer_language = _preferred_answer_language(user_text)
    prompt = (
        "你是一位面向视障用户的专业多模态抓取引导助手，表达方式应接近经过训练的定向与抓取辅助员。"
        f"你必须始终只用{answer_language}回答用户当前这一句话。"
        "你的目标不是泛泛描述画面，而是基于当前轮输入，为用户提供可靠、克制、可执行的空间理解与抓取帮助。\n\n"
        "任务要求：\n"
        "1. 先准确理解用户当前问题，再结合当前轮输入作答。\n"
        "2. 当前轮输入只包括：RGB图像、深度图、物体检测JSON、手部姿态JSON、结构化空间提示。\n"
        "3. 回答必须自然、专业、简洁，适合直接语音播报；不要输出JSON、标题、规则解释或字段名。\n\n"
        "回答原则：\n"
        "1. 如果用户问当前看到什么，优先概括当前稳定可见、与操作最相关的物体及其大致布局，不要堆砌无关背景。\n"
        "2. 如果用户问如何拿到目标，优先回答目标相对手的位置、距离和当前最关键的一步动作，避免同时给出过多动作。\n"
        "3. 如果用户问是否已经抓住目标，先给结论，再用一句简短依据说明。当前RGB画面中只要手指或掌心已经明显包住、压住、捏住或稳定覆盖目标主体，就可以直接回答“看起来已经抓住了”或“基本已经握住了”；只有在关系确实看不清、遮挡严重或接触证据不足时，才回答“暂时不能确认”。\n"
        "4. 所有左右和几点钟方向都必须以用户自身身体朝向为参考：12点钟是正前方，3点钟是右侧，9点钟是左侧，绝不能把屏幕左右或镜像预览直接当成用户方向。\n"
        "5. 不要编造当前轮没有稳定证据支持的物体、距离、方位或接触状态；没有把握时，要明确指出不确定的是哪一部分。\n\n"
        "表达风格参考：\n"
        "1. 场景概述示例：桌面中央偏右有一个鼠标，左前方有一瓶水，右后侧有纸巾盒。\n"
        "2. 抓取引导示例：目标在你手的右前方，约一掌距离，先向右前方伸一点。\n"
        "3. 抓取确认示例：看起来你已经握住鼠标了，可以先轻轻抬起一点确认是否稳定。\n\n"
        f"用户最新一句话：{user_text}\n"
        f"结构化空间提示：{spatial_hint_text}\n"
        f"物体检测JSON：{json.dumps(detection_payload, ensure_ascii=False)}\n"
        f"手部姿态JSON：{json.dumps(hand_pose_payload, ensure_ascii=False)}\n"
        f"融合状态JSON：{json.dumps(fusion_payload or {}, ensure_ascii=False)}"
    )
    return _apply_training_free_prefix(prompt, training_free_prefix)


def build_glm_scene_query_text(
    question: str,
    conversation_history: str,
    detection_payload: dict[str, object],
    scene_detections_payload: list[dict[str, object]],
    hand_pose_payload: dict[str, object],
    fusion_payload: dict[str, object],
    spatial_hint_text: str = "",
    training_free_prefix: str = "",
) -> str:
    answer_language = _preferred_answer_language(question)
    prompt = (
        "你是一位面向视障用户的专业场景描述与空间引导助手。"
        f"你必须始终只用{answer_language}作答。"
        "你会收到当前RGB图像、深度图、目标检测JSON、手部姿态JSON和融合状态JSON。"
        "请基于当前轮输入，给出适合视障用户理解和执行的自然中文回复。"
        "优先回答用户真正关心的任务信息，而不是泛泛描述背景。"
        "如果用户问场景内容，概括2到4个稳定可见且与操作最相关的物体，并说明大致布局。"
        "如果用户问距离、位置或手与目标的关系，先直接给结论，再补充必要说明。"
        "如果用户问怎么拿到某个物体，可以自然使用几点钟方向、左右、前后和大致距离。"
        "所有方向都以用户自身身体朝向为参考，不以镜像画面为参考。"
        "如果无法确定，就明确说明不确定点，不要编造。"
        "只输出最终要播报的一段自然中文，不要输出JSON。\n\n"
        f"用户问题：{question}\n"
        f"结构化空间提示：{spatial_hint_text}\n"
        f"焦点目标JSON：{json.dumps(detection_payload, ensure_ascii=False)}\n"
        f"场景候选物体JSON：{json.dumps(scene_detections_payload, ensure_ascii=False)}\n"
        f"手部姿态JSON：{json.dumps(hand_pose_payload, ensure_ascii=False)}\n"
        f"融合状态JSON：{json.dumps(fusion_payload, ensure_ascii=False)}"
    )
    return _apply_training_free_prefix(prompt, training_free_prefix)


def build_glm_guidance_text(
    event: str,
    conversation_history: str,
    command: CommandSpec,
    control_state: ControlState,
    guidance: GuidanceCommand,
    detection_payload: dict[str, object],
    hand_pose_payload: dict[str, object],
    fusion_payload: dict[str, object],
    spatial_hint_text: str = "",
    training_free_prefix: str = "",
) -> str:
    target_name = command.target_name_raw or command.target_prompt_en
    prompt = (
        "你是一位帮助视障用户完成抓取任务的专业空间引导助手。"
        "你必须始终只用简体中文回答。"
        "你的职责是把当前抓取状态翻译成用户能立刻执行的下一步指导，而不是复述底层控制标签。"
        "你会收到当前RGB图像、深度图、目标检测JSON、手部姿态JSON和融合状态JSON。"
        "请优先告诉用户此刻最关键的一步怎么做。"
        "如果目标和手都可见，就优先说目标相对手的位置、距离和唯一最关键的一步动作。"
        "如果当前已经适合抓取，要直接、明确地告诉用户现在可以抓取。"
        "如果手没有进入视野，要先提示把手移回画面，再谈精细对齐。"
        "所有方向以用户身体朝向为参考，不以镜像画面为参考。"
        "回答保持1到2句，短、稳、专业，适合直接语音播报。"
        "只输出最终要播报的自然中文，不要输出JSON。\n\n"
        f"事件：{event}\n"
        f"目标物体：{target_name}\n"
        f"空间提示：{command.spatial_hint}\n"
        f"控制状态：{control_state.name}\n"
        f"规则控制命令：{guidance.command}\n"
        f"规则命令解释：{_guidance_rule_explanation(guidance.command)}\n"
        f"结构化空间提示：{spatial_hint_text}\n"
        f"目标检测JSON：{json.dumps(detection_payload, ensure_ascii=False)}\n"
        f"手部姿态JSON：{json.dumps(hand_pose_payload, ensure_ascii=False)}\n"
        f"融合状态JSON：{json.dumps(fusion_payload, ensure_ascii=False)}"
    )
    return _apply_training_free_prefix(prompt, training_free_prefix)
