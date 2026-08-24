from __future__ import annotations

import aifc
from collections.abc import Callable
import math
import os
import re
import time
from pathlib import Path
import wave
import warnings

import numpy as np
from urllib3.exceptions import NotOpenSSLWarning

from src.common.schemas import AsrResult


warnings.filterwarnings("ignore", category=NotOpenSSLWarning)
warnings.filterwarnings("ignore", message=".*return_token_timestamps.*")
warnings.filterwarnings(
    "ignore", message=".*Transcription using a multilingual Whisper.*"
)

os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")


class AsrService:
    def __init__(
        self,
        backend: str = "text",
        model_id: str | None = None,
        sample_rate: int = 16000,
        record_seconds: int = 8,
        language: str | None = None,
        device: str = "cpu",
        min_speech_ms: int = 250,
        min_silence_ms: int = 600,
        chunk_ms: int = 200,
        silence_threshold: float = 0.015,
    ) -> None:
        self.backend = backend
        self.model_id = model_id
        self.sample_rate = sample_rate
        self.record_seconds = record_seconds
        self.language = language
        self.device = device
        self.min_speech_ms = min_speech_ms
        self.min_silence_ms = min_silence_ms
        self.chunk_ms = chunk_ms
        self.silence_threshold = silence_threshold
        self.pipeline = None

    def _load_whisper(self):
        if self.pipeline is not None:
            return self.pipeline
        import torch
        from transformers import pipeline

        model_id = self.model_id or "models/whisper-tiny"
        self._validate_local_model_assets(model_id)
        print(f"[ASR] Loading Whisper model from `{model_id}`...")
        if self.language:
            print(f"[ASR] Forcing language: {self.language}")
        torch_dtype = torch.float16 if self.device in {"cuda", "mps"} else torch.float32
        try:
            self.pipeline = pipeline(
                task="automatic-speech-recognition",
                model=model_id,
                tokenizer=model_id,
                feature_extractor=model_id,
                device=self.device,
                torch_dtype=torch_dtype,
            )
        except Exception:
            if self.device != "cpu":
                print("[ASR] Falling back to CPU for Whisper loading...")
                self.device = "cpu"
                self.pipeline = pipeline(
                    task="automatic-speech-recognition",
                    model=model_id,
                    tokenizer=model_id,
                    feature_extractor=model_id,
                    device="cpu",
                    torch_dtype=torch.float32,
                )
            else:
                raise
        return self.pipeline

    @staticmethod
    def _validate_local_model_assets(model_id: str) -> None:
        model_path = Path(model_id)
        if not model_path.exists() or not model_path.is_dir():
            return
        has_weight_file = any(
            child.name.startswith(("model-", "pytorch_model-"))
            or child.name in {"model.safetensors", "pytorch_model.bin"}
            for child in model_path.iterdir()
            if child.is_file()
        )
        has_shard_index = any(
            child.name.startswith(
                ("model.safetensors.index", "pytorch_model.bin.index")
            )
            for child in model_path.iterdir()
            if child.is_file()
        )
        if has_shard_index and not has_weight_file:
            raise RuntimeError(
                f"Whisper model directory '{model_id}' exists but the weight shards are incomplete. "
                "Re-download the full model files before running ASR."
            )

    def _record_microphone(
        self,
        progress_callback: Callable[[], None] | None = None,
    ) -> np.ndarray:
        import sounddevice as sd

        chunk_frames = max(1, int(self.sample_rate * self.chunk_ms / 1000))
        max_chunks = max(1, int(self.record_seconds * 1000 / self.chunk_ms))
        chunks: list[np.ndarray] = []

        print(f"[ASR] Listening for {self.record_seconds} seconds...")
        with sd.InputStream(
            samplerate=self.sample_rate,
            channels=1,
            dtype="float32",
            blocksize=chunk_frames,
        ) as stream:
            for _ in range(max_chunks):
                if progress_callback is not None:
                    progress_callback()
                audio, overflowed = stream.read(chunk_frames)
                if overflowed:
                    print(
                        "[ASR] Microphone overflow detected; continuing with latest buffer."
                    )
                chunks.append(audio[:, 0].copy())

        if not chunks:
            return np.array([], dtype=np.float32)
        return self._finalize_audio(np.concatenate(chunks))

    def _required_chunks(self, duration_ms: int) -> int:
        return max(1, math.ceil(duration_ms / self.chunk_ms))

    def _trim_silence(self, audio: np.ndarray) -> np.ndarray:
        if audio.size == 0:
            return audio
        mask = np.abs(audio) >= self.silence_threshold / 2
        if not np.any(mask):
            return np.array([], dtype=np.float32)
        indices = np.flatnonzero(mask)
        start = max(0, int(indices[0]) - int(self.sample_rate * 0.08))
        end = min(audio.shape[0], int(indices[-1]) + int(self.sample_rate * 0.12))
        return audio[start:end].astype(np.float32)

    def _finalize_audio(self, audio: np.ndarray) -> np.ndarray:
        trimmed = self._trim_silence(audio)
        minimum_audio_ms = max(self.min_speech_ms * 2, 400)
        minimum_audio_samples = int(self.sample_rate * minimum_audio_ms / 1000)
        if trimmed.shape[0] < minimum_audio_samples:
            return np.array([], dtype=np.float32)
        return trimmed

    def _read_audio_file(self, audio_path: str) -> np.ndarray:
        path = Path(audio_path)
        suffix = path.suffix.lower()
        if suffix in {".wav", ".wave"}:
            with wave.open(str(path), "rb") as handle:
                frames = handle.readframes(handle.getnframes())
                sample_rate = handle.getframerate()
                channels = handle.getnchannels()
                audio = (
                    np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
                )
        elif suffix in {".aif", ".aiff", ".aifc"}:
            with aifc.open(str(path), "rb") as handle:
                frames = handle.readframes(handle.getnframes())
                sample_rate = handle.getframerate()
                channels = handle.getnchannels()
                audio = (
                    np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
                )
        else:
            raise RuntimeError(f"Unsupported audio file format: {path.suffix}")

        if channels > 1:
            audio = audio.reshape(-1, channels).mean(axis=1)
        if sample_rate != self.sample_rate:
            duration = audio.shape[0] / sample_rate
            target_length = int(duration * self.sample_rate)
            old_positions = np.linspace(0.0, 1.0, num=audio.shape[0], endpoint=False)
            new_positions = np.linspace(0.0, 1.0, num=target_length, endpoint=False)
            audio = np.interp(new_positions, old_positions, audio).astype(np.float32)
        return audio

    def _generate_kwargs(self) -> dict[str, str]:
        kwargs = {"task": "transcribe"}
        if self.language:
            kwargs["language"] = self.language
        return kwargs

    @staticmethod
    def _looks_repetitive(text: str) -> bool:
        normalized = re.sub(r"\s+", " ", text).strip()
        if not normalized:
            return False
        tokens = normalized.split(" ")
        if len(tokens) >= 4 and len(set(tokens)) <= max(1, len(tokens) // 4):
            return True
        compact = normalized.replace(" ", "")
        if len(compact) >= 8 and len(set(compact)) <= max(2, len(compact) // 6):
            return True
        return False

    @staticmethod
    def _looks_like_punctuation_noise(text: str) -> bool:
        compact = re.sub(r"\s+", "", text)
        if not compact:
            return False
        return all(
            not char.isalnum() and not ("\u4e00" <= char <= "\u9fff")
            for char in compact
        )

    def _estimate_confidence(self, text: str) -> float:
        cleaned = re.sub(r"\s+", " ", text).strip()
        if not cleaned:
            return 0.0
        if self._looks_like_punctuation_noise(cleaned):
            return 0.0
        confidence = 0.85
        compact = cleaned.replace(" ", "")
        if len(compact) < 3:
            confidence -= 0.4
        elif len(compact) < 5:
            confidence -= 0.2
        if self._looks_repetitive(cleaned):
            confidence -= 0.45
        unique_ratio = len(set(compact)) / max(1, len(compact))
        if unique_ratio < 0.35:
            confidence -= 0.2
        return max(0.0, min(1.0, confidence))

    def listen_once(
        self,
        preset_text: str | None = None,
        audio_path: str | None = None,
        progress_callback: Callable[[], None] | None = None,
    ) -> AsrResult:
        start = time.time()
        if preset_text is not None:
            text = preset_text.strip()
            confidence = 1.0 if text else 0.0
        elif self.backend == "text":
            text = input("Command> ").strip()
            confidence = 1.0 if text else 0.0
        elif self.backend == "whisper":
            transcriber = self._load_whisper()
            if audio_path is not None:
                print(f"[ASR] Reading audio from `{audio_path}`...")
                audio = self._read_audio_file(audio_path)
            else:
                audio = self._record_microphone(progress_callback=progress_callback)
            if audio.size == 0:
                text = ""
                confidence = 0.0
            else:
                result = transcriber(
                    {"raw": audio, "sampling_rate": self.sample_rate},
                    generate_kwargs=self._generate_kwargs(),
                )
                if isinstance(result, list):
                    text = " ".join(
                        str(item.get("text", "")).strip() for item in result
                    )
                else:
                    text = str(result.get("text", "")).strip()
                confidence = self._estimate_confidence(text)
        else:
            raise RuntimeError(
                f"ASR backend '{self.backend}' is not configured yet for live microphone capture."
            )
        latency_ms = int((time.time() - start) * 1000)
        print(f"[ASR] Transcribed: {text or '<empty>'}")
        return AsrResult(text=text, confidence=confidence, latency_ms=latency_ms)
