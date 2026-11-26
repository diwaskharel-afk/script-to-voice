"""Text-to-speech providers: ElevenLabs (REST) and a free mock that makes tones."""
from __future__ import annotations

import io
import math
import struct
import time
import wave
from dataclasses import dataclass
from typing import Protocol

import requests

from .config import Settings
from .models import StvError, count_words

API_BASE = "https://api.elevenlabs.io/v1"


class TTSError(StvError):
    pass


@dataclass
class Synthesis:
    data: bytes
    ext: str  # "mp3" or "wav"


class TTS(Protocol):
    def synthesize(self, text: str, *, previous_text: str = "", next_text: str = "") -> Synthesis: ...


class ElevenLabsTTS:
    """One request per clip. The same voice and settings are used for every clip, and
    neighbouring text is passed along so intonation stays consistent across clips."""

    def __init__(self, settings: Settings, session: requests.Session | None = None):
        if not settings.elevenlabs_api_key:
            raise TTSError("ELEVENLABS_API_KEY is not set")
        self.s = settings
        self.http = session or requests.Session()

    @property
    def _headers(self) -> dict:
        return {"xi-api-key": self.s.elevenlabs_api_key, "Accept": "audio/mpeg"}

    def build_payload(self, text: str, previous_text: str = "", next_text: str = "") -> dict:
        payload: dict = {
            "text": text,
            "model_id": self.s.elevenlabs_model,
            "voice_settings": {
                "stability": self.s.stability,
                "similarity_boost": self.s.similarity_boost,
                "style": self.s.style,
            },
        }
        if self.s.supports_context:  # v3 rejects previous_text / next_text
            if previous_text:
                payload["previous_text"] = previous_text
            if next_text:
                payload["next_text"] = next_text
        if self.s.pronunciation_dictionary_ids:
            payload["pronunciation_dictionary_locators"] = [
                {"pronunciation_dictionary_id": i} for i in self.s.pronunciation_dictionary_ids
            ]
        return payload

    def synthesize(self, text: str, *, previous_text: str = "", next_text: str = "") -> Synthesis:
        url = f"{API_BASE}/text-to-speech/{self.s.elevenlabs_voice_id}"
        payload = self.build_payload(text, previous_text, next_text)
        last = ""
        for attempt in range(4):
            try:
                r = self.http.post(
                    url, params={"output_format": "mp3_44100_128"},
                    headers=self._headers, json=payload, timeout=120,
                )
            except requests.RequestException as e:
                last = str(e)
                time.sleep(2 ** attempt)
                continue
            if r.status_code == 200:
                return Synthesis(r.content, "mp3")
            last = f"HTTP {r.status_code}: {r.text[:300]}"
            if r.status_code in (429, 500, 502, 503, 504):  # rate limit / transient
                time.sleep(2 ** attempt)
                continue
            if r.status_code in (401, 403):
                raise TTSError(f"ElevenLabs rejected the API key or plan ({last})")
            if r.status_code == 404:
                raise TTSError(f"ElevenLabs voice not found: {self.s.elevenlabs_voice_id}")
            if r.status_code == 422:
                raise TTSError(f"ElevenLabs rejected the request (check model/voice settings): {last}")
            raise TTSError(f"ElevenLabs error {last}")
        raise TTSError(f"ElevenLabs request failed after retries: {last}")

    def list_voices(self) -> list[tuple[str, str]]:
        r = self.http.get(f"{API_BASE}/voices", headers={"xi-api-key": self.s.elevenlabs_api_key}, timeout=30)
        if r.status_code != 200:
            raise TTSError(f"could not list voices: HTTP {r.status_code}")
        return [(v["voice_id"], v["name"]) for v in r.json().get("voices", [])]


class MockTTS:
    """Makes a tone whose length matches the word count, like speech would. No credits.

    Deterministic, so tests can assert exact fit-check outcomes. `seconds_per_word`
    can be raised to simulate a slow narrator that forces shortening.
    """

    def __init__(self, seconds_per_word: float = 0.4, sample_rate: int = 44100):
        self.spw = seconds_per_word
        self.sr = sample_rate
        self.calls = 0

    def synthesize(self, text: str, *, previous_text: str = "", next_text: str = "") -> Synthesis:
        self.calls += 1
        seconds = max(0.3, count_words(text) * self.spw)
        n = int(seconds * self.sr)
        freq = 220 + (sum(map(ord, text)) % 200)
        frames = b"".join(
            struct.pack("<h", int(9000 * math.sin(2 * math.pi * freq * i / self.sr))) for i in range(n)
        )
        buf = io.BytesIO()
        with wave.open(buf, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(self.sr)
            w.writeframes(frames)
        return Synthesis(buf.getvalue(), "wav")
