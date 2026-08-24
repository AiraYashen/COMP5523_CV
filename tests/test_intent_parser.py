from src.llm.intent_parser import IntentParser


PROMPTS = {
    "object_map": {
        "coke": {
            "keywords": ["coke", "cola", "可乐"],
            "prompts": ["red soda can"],
        },
        "milk": {
            "keywords": ["milk", "牛奶"],
            "prompts": ["milk carton"],
        },
        "tissue_box": {
            "keywords": ["tissue", "纸巾", "抽纸"],
            "prompts": ["tissue box"],
        },
    }
}


def test_parse_grasp_command() -> None:
    parser = IntentParser(PROMPTS)
    command = parser.parse_intent("help me grab the coke can in front")
    assert command.action == "grasp"
    assert command.target_prompt_en == "red soda can"
    assert command.spatial_hint == "front"


def test_parse_stop_command() -> None:
    parser = IntentParser(PROMPTS)
    command = parser.parse_intent("停止")
    assert command.action == "stop"


def test_parse_unknown_command() -> None:
    parser = IntentParser(PROMPTS)
    command = parser.parse_intent("grab something")
    assert command.action == "unknown"
    assert command.need_confirmation is True


def test_parse_scene_query_command() -> None:
    parser = IntentParser(PROMPTS)
    command = parser.parse_intent("前面有什么？")
    assert command.action == "scene_query"
    assert command.query_text == "前面有什么？"


def test_parse_scene_query_from_english_prompt() -> None:
    parser = IntentParser(PROMPTS)
    command = parser.parse_intent("what do you see in front of me")
    assert command.action == "scene_query"
    assert "what do you see" in command.query_text


def test_parse_grasp_command_with_question_mark_stays_grasp() -> None:
    parser = IntentParser(PROMPTS)
    command = parser.parse_intent("grab the coke can?")
    assert command.action == "grasp"
    assert command.target_prompt_en == "red soda can"


def test_parse_hand_distance_question_is_scene_query() -> None:
    parser = IntentParser(PROMPTS)
    command = parser.parse_intent("这个瓶子离我的手有多远")
    assert command.action == "scene_query"
    assert "多远" in command.query_text


def test_parse_repetitive_low_information_text_stays_unknown() -> None:
    parser = IntentParser(PROMPTS)
    command = parser.parse_intent("谢谢你 谢谢你 谢谢你 谢谢你")
    assert command.action == "unknown"
    assert command.confirmation_question == "请再说一遍。"


def test_parse_tabletop_scene_question_without_punctuation() -> None:
    parser = IntentParser(PROMPTS)
    command = parser.parse_intent("桌面上有什么东西")
    assert command.action == "scene_query"
    assert command.query_text == "桌面上有什么东西"


def test_parse_chinese_grasp_guidance_for_supported_target() -> None:
    parser = IntentParser(PROMPTS)
    command = parser.parse_intent("我的手应该怎么移动能拿到纸巾")
    assert command.action == "grasp"
    assert command.target_prompt_en == "tissue box"


def test_parse_chinese_grasp_guidance_for_unknown_target_falls_back_to_scene_query() -> (
    None
):
    parser = IntentParser(PROMPTS)
    command = parser.parse_intent("我的手应该怎么移动能拿到遥控器")
    assert command.action == "scene_query"
    assert command.query_text == "我的手应该怎么移动能拿到遥控器"
