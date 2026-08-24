from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from src.audio.asr_service import AsrService
from src.audio.tts_service import TtsService


def test_asr_generate_kwargs_include_language() -> None:
    service = AsrService(language="zh")
    assert service._generate_kwargs() == {"task": "transcribe", "language": "zh"}


def test_asr_confidence_penalizes_repetition() -> None:
    service = AsrService()
    repetitive = service._estimate_confidence("谢谢你 谢谢你 谢谢你 谢谢你")
    normal = service._estimate_confidence("帮我拿前面的可乐")
    assert repetitive < 0.45
    assert normal > repetitive


def test_asr_confidence_rejects_punctuation_noise() -> None:
    service = AsrService()
    assert service._estimate_confidence(".!") == 0.0
    assert service._estimate_confidence(",!") == 0.0


def test_required_chunks_uses_ceiling() -> None:
    service = AsrService(chunk_ms=200)
    assert service._required_chunks(250) == 2
    assert service._required_chunks(700) == 4


def test_finalize_audio_rejects_too_short_capture() -> None:
    service = AsrService(sample_rate=16000, min_speech_ms=250)
    audio = np.ones(2000, dtype=np.float32)
    finalized = service._finalize_audio(audio)
    assert finalized.size == 0


def test_finalize_audio_keeps_long_enough_capture() -> None:
    service = AsrService(sample_rate=16000, min_speech_ms=250, silence_threshold=0.001)
    audio = np.ones(8000, dtype=np.float32) * 0.2
    finalized = service._finalize_audio(audio)
    assert finalized.size > 0


def test_tts_service_tracks_last_speak_time() -> None:
    service = TtsService(backend="print")
    service.speak("hello")
    assert service.time_since_last_speak_ms() >= 0.0
    time.sleep(0.01)
    assert service.time_since_last_speak_ms() >= 10.0


def test_asr_validate_local_model_assets_rejects_incomplete_shards(
    tmp_path: Path,
) -> None:
    model_dir = tmp_path / "whisper-large-v3"
    model_dir.mkdir()
    (model_dir / "model.safetensors.index.fp32.json").write_text("{}", encoding="utf-8")
    try:
        AsrService._validate_local_model_assets(str(model_dir))
    except RuntimeError as exc:
        assert "weight shards are incomplete" in str(exc)
    else:
        raise AssertionError("expected incomplete model shards to be rejected")
