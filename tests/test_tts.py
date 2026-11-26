import pytest

from script_to_voice.config import Settings
from script_to_voice.media import audio_duration
from script_to_voice.tts import ElevenLabsTTS, MockTTS, TTSError


class FakeResponse:
    def __init__(self, status=200, content=b"MP3DATA", text=""):
        self.status_code, self.content, self.text = status, content, text


class FakeSession:
    def __init__(self, *responses):
        self.responses, self.calls = list(responses), []

    def post(self, url, **kw):
        self.calls.append((url, kw))
        return self.responses.pop(0)


def settings(**kw):
    return Settings(elevenlabs_api_key="k", elevenlabs_voice_id="VOICE", **kw)


def test_payload_includes_context_and_voice_settings():
    tts = ElevenLabsTTS(settings(), FakeSession(FakeResponse()))
    out = tts.synthesize("Hello", previous_text="Before", next_text="After")
    url, kw = tts.http.calls[0]
    assert out.data == b"MP3DATA" and out.ext == "mp3"
    assert url.endswith("/text-to-speech/VOICE")
    assert kw["headers"]["xi-api-key"] == "k"
    assert kw["json"]["previous_text"] == "Before" and kw["json"]["next_text"] == "After"
    assert kw["json"]["voice_settings"]["stability"] == 0.5


def test_v3_omits_context():
    tts = ElevenLabsTTS(settings(elevenlabs_model="eleven_v3"), FakeSession(FakeResponse()))
    tts.synthesize("Hello", previous_text="Before", next_text="After")
    body = tts.http.calls[0][1]["json"]
    assert "previous_text" not in body and "next_text" not in body


def test_pronunciation_dictionaries_sent():
    tts = ElevenLabsTTS(settings(pronunciation_dictionary_ids=["d1"]), FakeSession(FakeResponse()))
    tts.synthesize("Hello")
    assert tts.http.calls[0][1]["json"]["pronunciation_dictionary_locators"] == [
        {"pronunciation_dictionary_id": "d1"}
    ]


def test_retries_rate_limit_then_succeeds(monkeypatch):
    monkeypatch.setattr("script_to_voice.tts.time.sleep", lambda s: None)
    tts = ElevenLabsTTS(settings(), FakeSession(FakeResponse(429, text="slow down"), FakeResponse()))
    assert tts.synthesize("Hi").data == b"MP3DATA"
    assert len(tts.http.calls) == 2


@pytest.mark.parametrize("status,needle", [(401, "API key"), (404, "voice not found"), (422, "rejected")])
def test_clear_errors(status, needle):
    tts = ElevenLabsTTS(settings(), FakeSession(FakeResponse(status, text="nope")))
    with pytest.raises(TTSError, match=needle):
        tts.synthesize("Hi")


def test_gives_up_after_retries(monkeypatch):
    monkeypatch.setattr("script_to_voice.tts.time.sleep", lambda s: None)
    tts = ElevenLabsTTS(settings(), FakeSession(*[FakeResponse(503) for _ in range(4)]))
    with pytest.raises(TTSError, match="after retries"):
        tts.synthesize("Hi")


def test_missing_key():
    with pytest.raises(TTSError):
        ElevenLabsTTS(Settings())


def test_mock_tts_length_tracks_word_count():
    syn = MockTTS(0.4).synthesize("one two three four five")
    assert audio_duration(syn.data) == pytest.approx(2.0, abs=0.02)
