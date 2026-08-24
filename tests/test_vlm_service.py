import numpy as np

from src.vlm.vlm_service import VLMService


def test_parse_response_json_success() -> None:
    result = VLMService._parse_response(
        '{"scene_status":"ok","scene_description":"The can is ahead.","uncertainty":"low","speak":true}'
    )
    assert result.scene_status == "ok"
    assert result.scene_description == "The can is ahead."
    assert result.speak is True


def test_parse_response_falls_back_for_plain_text() -> None:
    result = VLMService._parse_response("The can is slightly right of the hand.")
    assert result.scene_status == "unknown"
    assert "slightly right" in result.scene_description


def test_build_glm_payload_contains_two_images_and_text() -> None:
    service = VLMService(
        enabled=True,
        model_id="glm-4.6v",
        backend="glm_api",
        api_key="dummy",
        api_base_url="https://example.com",
    )
    rgb = np.zeros((8, 8, 3), dtype=np.uint8)
    depth = np.ones((8, 8), dtype=np.float32)
    payload = service._build_glm_payload(rgb, depth, "analyze this scene")
    content = payload["messages"][0]["content"]
    assert payload["model"] == "glm-4.6v"
    assert content[0]["type"] == "image_url"
    assert content[1]["type"] == "image_url"
    assert content[2]["text"] == "analyze this scene"
    assert payload["thinking"] == {"type": "disabled"}


def test_build_responses_payload_contains_input_images_and_text() -> None:
    service = VLMService(
        enabled=True,
        model_id="gpt-5.4",
        backend="responses_api",
        api_key="dummy",
        api_base_url="https://example.com/v1/responses",
    )
    rgb = np.zeros((8, 8, 3), dtype=np.uint8)
    depth = np.ones((8, 8), dtype=np.float32)
    payload = service._build_responses_payload(rgb, depth, "analyze this scene")
    content = payload["input"][0]["content"]
    assert payload["model"] == "gpt-5.4"
    assert payload["input"][0]["role"] == "user"
    assert content[0]["type"] == "input_image"
    assert content[1]["type"] == "input_image"
    assert content[2]["type"] == "input_text"
    assert content[2]["text"] == "analyze this scene"


def test_extract_responses_output_text_handles_openai_style_output() -> None:
    raw_payload = {
        "output": [
            {
                "type": "message",
                "content": [
                    {"type": "output_text", "text": '{"scene_status":"ok"}'}
                ],
            }
        ]
    }
    assert VLMService._extract_responses_output_text(raw_payload) == '{"scene_status":"ok"}'


def test_extract_message_content_handles_list_blocks() -> None:
    message = {
        "content": [
            {"type": "output_text", "text": '{"scene_status":"ok"}'},
        ]
    }
    assert VLMService._extract_message_content(message) == '{"scene_status":"ok"}'


def test_extract_message_content_handles_dict_block() -> None:
    message = {"content": {"text": '{"scene_status":"ok"}'}}
    assert VLMService._extract_message_content(message) == '{"scene_status":"ok"}'
