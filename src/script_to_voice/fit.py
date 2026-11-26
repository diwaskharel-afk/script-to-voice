"""The fit check: make every narration line fit the seconds its section has.

ElevenLabs cannot be told "say this in exactly N seconds", so we measure and adjust:

    generate -> measure
      fits the slot                      -> keep                    (ok)
      slightly too long (<= max_speedup) -> speed up with ffmpeg    (sped_up)
      too long                           -> ask the LLM for fewer words, regenerate  (shortened)
      still too long after N attempts    -> best effort, flag for a human            (overflow)
"""
from __future__ import annotations

import tempfile
from pathlib import Path

from .cache import ClipCache
from .config import Settings
from .llm import LLM, ScriptContext
from .media import change_tempo, audio_duration, load_audio
from .models import (
    Clip, Section, StvError, count_words, ERROR, OK, OVERFLOW, SHORTENED, SPED_UP, SKIPPED,
)
from .pronunciation import Glossary
from .tts import TTS


class FitEngine:
    def __init__(self, tts: TTS, llm: LLM, cache: ClipCache, settings: Settings,
                 ctx: ScriptContext, glossary: Glossary | None = None):
        self.tts, self.llm, self.cache, self.s, self.ctx = tts, llm, cache, settings, ctx
        self.glossary = glossary or Glossary()

    # ------------------------------------------------------------------ generation
    def _cache_key(self, spoken: str) -> str:
        s = self.s
        return ClipCache.key(
            provider=s.tts_provider, voice=s.elevenlabs_voice_id, model=s.elevenlabs_model,
            stability=s.stability, similarity=s.similarity_boost, style=s.style,
            dictionaries=s.pronunciation_dictionary_ids, text=spoken,
        )  # neighbours are deliberately not part of the key: editing one line must not regenerate the others

    def _synth(self, text: str, prev: str, nxt: str):
        """Return (decoded AudioSegment, raw bytes, ext). Uses the cache; counts as one attempt only if it hit the API."""
        spoken = self.glossary.apply(text)
        key = self._cache_key(spoken)
        hit = self.cache.get(key)
        if hit:
            data, ext = hit
            fresh = False
        else:
            syn = self.tts.synthesize(spoken, previous_text=self.glossary.apply(prev),
                                      next_text=self.glossary.apply(nxt))
            self.cache.put(key, syn.data, syn.ext)
            data, ext, fresh = syn.data, syn.ext, True
        return load_audio(data, self.s.sample_rate), data, ext, fresh

    # ------------------------------------------------------------------ the loop
    def fit(self, section: Section, prev_text: str, next_text: str, out_path: Path) -> Clip:
        text = section.script.strip()
        clip = Clip(section_index=section.index, slot=section.slot, original_text=text, final_text=text)
        if not text:
            clip.status, clip.message = SKIPPED, "no narration for this section"
            return clip

        out_path.parent.mkdir(parents=True, exist_ok=True)
        target = max(0.5, section.slot - self.s.tail_padding)
        shortened = False
        speed = 1.0
        tmp_ctx = tempfile.TemporaryDirectory(prefix="stv_fit_")
        tmp_dir = Path(tmp_ctx.name)

        try:
            for attempt in range(self.s.max_shorten_attempts + 1):
                seg, data, ext, fresh = self._synth(text, prev_text, next_text)
                clip.attempts += int(fresh)
                dur = len(seg) / 1000.0

                if dur <= target:
                    seg.export(str(out_path), format="wav")
                    return self._done(clip, text, out_path, SHORTENED if shortened else OK, 1.0,
                                      f"{dur:.1f}s in {section.slot:.1f}s slot")

                ratio = dur / target
                if ratio <= self.s.max_speedup:
                    speed = min(ratio * 1.005, self.s.max_speedup)
                    raw = tmp_dir / f"raw.{ext}"
                    raw.write_bytes(data)
                    change_tempo(raw, out_path, speed)
                    return self._done(clip, text, out_path, SHORTENED if shortened else SPED_UP, speed,
                                      f"{dur:.1f}s sped up x{speed:.2f} to fit {section.slot:.1f}s")

                if attempt < self.s.max_shorten_attempts:
                    old_words = count_words(text)
                    goal = max(1, int(old_words * (target / dur) * 0.95))
                    try:
                        new = self.llm.shorten(text, goal, self.ctx).strip()
                    except StvError as e:
                        clip.message = f"could not shorten: {e}"
                        break
                    if not new or count_words(new) >= old_words:
                        break  # the model did not help; stop spending credits
                    text, shortened = new, True
                    continue

            # Out of attempts: use the last clip at max speed and flag it.
            raw = tmp_dir / f"raw.{ext}"
            raw.write_bytes(data)
            speed = min(self.s.max_speedup, max(1.0, dur / target))
            change_tempo(raw, out_path, speed)
            return self._done(clip, text, out_path, OVERFLOW, speed,
                              f"still {dur / speed:.1f}s for a {section.slot:.1f}s slot - needs a human")
        except StvError as e:
            clip.status, clip.message = ERROR, str(e)
            return clip
        finally:
            tmp_ctx.cleanup()

    def _done(self, clip: Clip, text: str, path: Path, status: str, speed: float, message: str) -> Clip:
        clip.path = str(path)
        clip.final_text = text
        clip.status = status
        clip.speed = round(speed, 3)
        clip.duration = round(audio_duration(path), 3)
        clip.message = message
        return clip
