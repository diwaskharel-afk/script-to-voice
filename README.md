# Script to Voice

Upload a training video and its documentation. Get back **one audio file of narration, as long as the video, with the right gaps**, that drops straight onto the video. Built from [Plan.md](Plan.md).

```
video + docs ──► Gemini finds the moments ──► LLM writes one line per moment
                                                        │
        one audio file ◄── place clips on a blank track ◄── ElevenLabs voices each line
        (gaps = silence)                                      └─ fit check: measure, speed up, or shorten
```

## Quick start

```powershell
python -m venv .venv
.\.venv\Scripts\pip install -e ".[dev]"

.\.venv\Scripts\stv demo                     # whole pipeline on fake data, no keys, no credits
.\.venv\Scripts\streamlit run streamlit_app.py
```

ffmpeg is bundled through `imageio-ffmpeg`; a system ffmpeg (or `FFMPEG_BINARY`) is used if present.

For real runs copy `.env.example` to `.env` and add `GEMINI_API_KEY` and `ELEVENLABS_API_KEY` (or paste them in the app sidebar). With a key present the matching service is used automatically; without it that side stays mocked.

## Using it

**App:** five tabs. Input, Sections (editable slots), Script (editable per line, with a word-budget check and a "check against docs" pass), Voice & fit (per-clip player and status), Result (download audio and `.srt`, build a preview with the audio over the video).

**CLI:**

```powershell
# full run from a video
stv run --name onboarding --video demo.mp4 --docs guide.pdf --notes "new hires, relaxed" --persona "Friendly trainer" --glossary examples/glossary.txt --preview

# hand-written slots, no video analysis (build-order step 2)
stv run --name test --sections examples/sample_sections.json --duration 60 --docs examples/sample_doc.md

# after editing projects/<name>/project.json by hand: only changed lines cost credits
stv build projects/onboarding

stv voices            # list your ElevenLabs voices
```

Exit code is 1 when any line could not be fitted, so it works in scripts.

## The fit check

ElevenLabs cannot be told "say this in exactly N seconds", so every line is measured after generation:

| Result | Action | Status |
|---|---|---|
| fits `slot - tail_padding` | keep | `ok` |
| up to 1.15x too long | speed up with ffmpeg `atempo` (free, pitch-preserving) | `sped_up` |
| more than that | ask the LLM for the same meaning in fewer words, regenerate (max 2 times) | `shortened` |
| still too long | best effort at max speed, **flag for a human** | `overflow` |

At assembly an overflowing clip is trimmed (with a short fade) where the next line starts, rather than talking over it. Set `on_overflow = "overlap"` to disable that.

## Cost control

Clips are cached by text + voice + model + voice settings, so re-running spends nothing and editing one line regenerates only that line. Neighbouring text is deliberately not in the cache key. Speeding up is done locally, so it never costs credits. Use mock mode while tuning the script.

## Layout

```
src/script_to_voice/
  models.py      Section, Clip, timecode parsing
  sections.py    clean up / load timestamps
  llm.py         Gemini (video -> sections, script, shorten, verify) + mock, personas
  tts.py         ElevenLabs REST client + mock
  fit.py         the fit-check loop
  assemble.py    pydub assembly, .srt
  media.py       ffmpeg helpers (tempo, duration, chunking, preview mux)
  pipeline.py    one function per step
  project.py     on-disk project (resume, edit, rerun)
  cache.py  pronunciation.py  docs.py  config.py  cli.py
streamlit_app.py
personas/        extra personas, one .md file each
examples/        sample slots, docs, glossary
tests/           59 tests, no network
```

## Decisions and limits

- **Gemini timestamps** drift on long videos, so videos over 10 minutes are split, analysed in pieces, and offset back. Fast cuts can be missed (about 1 frame per second sampled). Review the Sections tab.
- **ElevenLabs v3** does not take `previous_text`/`next_text` or `<break>` tags; the client omits the context for v3. Timed pauses need Multilingual v2 or Flash.
- **Script accuracy:** the writer is told to use only the docs and what is on screen, and a separate pass flags unsupported claims. It is a safety net, not a guarantee. Read the script.
- **Pronunciation:** the glossary rewrites terms only in the text sent to the voice model; the script keeps real spelling. ElevenLabs pronunciation dictionary IDs can be added via `Settings.pronunciation_dictionary_ids`.
- Output is audio only, as planned. For manual polish, import `narration.wav` into Dubbing Studio or any editor.

## Not yet verified

The Gemini and ElevenLabs calls were written against their documentation and tested with fakes, **not against live accounts**. Before relying on it: run one short real video, check that the section timestamps match the footage, that your ElevenLabs plan allows the chosen model and voice, and that Gemini's model name (`GEMINI_MODEL`) is current.
