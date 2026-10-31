# Automated script-to-voice workflow for technical training videos

**Goal:** reduce repeated manual recording. Upload a video and its documentation, get back one audio file of narration, with the right gaps, that drops straight onto the video.

**Verdict:** feasible. The AI and voice steps already exist in the current Streamlit app. The new work is getting timestamps from the video and a fit-and-assemble step. The one thing ElevenLabs cannot do is "say this sentence in exactly N seconds", so the plan handles that with a measure-and-adjust loop.

## What exists today
A Streamlit app with two layers:
1. **Layer 1:** upload a PDF (optionally the video), add context about the video wanted. Personas and skills built into the app mean the user doesn't repeat instructions. An LLM API writes the script.
2. **Layer 2:** the ElevenLabs API turns the script into voice.

## What is new
- Timestamps from the video (so each line knows when it starts and how many seconds it has).
- A fit check so each line fits its time slot.
- An assembly step that places the clips with gaps between them.

## The flow
1. **Input.** The user uploads the video and documentation, and adds a short note on tone.
2. **Timestamps.** Gemini watches the video and lists the moments, e.g. 8 sections, each with a start time and the seconds available.
3. **Script.** The persona and skills prompt writes one line per section, told how many seconds it has. Rule of thumb: about 2.5 spoken words per second, so 8 seconds is roughly 20 words.
4. **Voice.** ElevenLabs generates one clip per section, with the same voice, and the neighbouring text passed in so tone stays consistent across clips.
5. **Fit check.** Measure each clip against its slot (see below).
6. **Assemble.** Make one blank audio track as long as the video and drop each clip at its start time. The gaps are simply the silence in the blank track. Export one audio file.
7. **Review (optional).** Play the audio over the video, or open it in Dubbing Studio or any editor, and fix any line by hand.

## How much control ElevenLabs gives
| Want | Supported? |
|---|---|
| Choose who speaks | Yes. A voice is chosen per request, so each section can use a different one. |
| Faster or slower | Yes, within limits. Speed setting, 1.0 normal. Treat about 0.8-1.2 as safe. |
| Mood and expressiveness | Yes. Stability and style settings. |
| Short pauses inside a line | Partly. `<break time="1.5s" />` works up to about 3 seconds on Multilingual v2 and Flash models, not on v3. v3 uses tags like `[pause]` with no set duration. |
| Make a sentence last exactly N seconds | No direct setting found. Length is known only after generating. |
| Know how long each line came out | Yes. A "with timestamps" endpoint returns word and character timing, and the file can also just be measured. |
| Keep separate clips sounding like one speaker | Yes. Each request can be told the text before and after it. |

## The fit check: sentence needs 8 seconds, slot is 3
1. Generate the line.
2. Measure how long it came out.
3. Fits the slot: keep it.
4. Slightly too long: speed it up a little, up to about 1.15x.
5. Still too long: send it back to the AI ("same meaning, fewer words") and regenerate.
6. Still does not fit: flag it for a human.

Speed alone is not enough. Cranking it up sounds rushed, so shortening the text does most of the work.

## Connecting the clips with gaps
This does not need to be built from scratch.
- **Automatic, free:** a small script using ffmpeg or pydub places each clip at its start time on a blank track. Known gotchas with ffmpeg's `amix` and `adelay` are quieter output and timing quirks. Pydub's overlay approach avoids both, but it is untested here, so test it.
- **Manual last pass:** ElevenLabs Dubbing Studio has a timeline for placing and nudging clips against the video. Regenerating a clip keeps its length fixed, and it exports WAV, MP3, or timeline files. Good for polishing, not for full automation. The separate Dubbing API is built for translating existing speech, not for writing new narration.

## Getting timestamps from the video
Gemini can watch an uploaded video and return a timestamped list of what happens. It samples about one frame per second, so very fast cuts can be missed. A third-party blog reports timestamps drifting on long videos, so chunk long videos into pieces. For short training videos this should be fine, but test it on real footage.

## Stack
**APIs**
| What | Used for |
|---|---|
| Gemini API | Watches the video and returns timestamps. Can also write the script. |
| ElevenLabs API | Turns each script line into a voice clip. |
| A script-writing LLM (optional) | Writes the lines using the personas and skills. Gemini can do this, so it is not a separate API unless wanted. |

Minimum: **two APIs**, Gemini and ElevenLabs.

**Free tools**
| What | Used for |
|---|---|
| ffmpeg | Measures clip length, speeds clips up slightly, exports the final file. |
| pydub | Makes the blank track and places each clip at its start time. |
| Streamlit | The existing two-layer app. |
| Python | The glue that runs the loop: generate, measure, fit check, shorten, assemble. |

**Optional:** ElevenLabs Dubbing Studio for manual fixes; any video editor to place the audio over the video.

**Built in-house:** only the glue code (fit-check loop and assembly), roughly a few hundred lines of Python.

**Not needed:** a separate speech-timing service, a database (unless past projects should be saved), video rendering (the output is audio only).

## Build order
1. **Prove the gap logic.** Build the assemble step with fake audio clips. This costs no ElevenLabs credits.
2. **Add the fit-check loop** with ElevenLabs and a fixed, hand-written list of slots.
3. **Add Gemini timestamps** so the slots come from the video instead of by hand.
4. **Add a preview** that plays the audio over the video in the app.
5. **Polish:** persona tuning, pronunciation of technical terms, caching unchanged clips to save credits.

## Risks
- **Script accuracy.** LLMs can invent steps, so check the script against the documentation.
- **Technical terms and product names** can be mispronounced. ElevenLabs offers pronunciation dictionaries, or terms can be spelled phonetically in the script.
- **Cost.** Each regeneration uses characters, so cache clips that have not changed.
- **Timestamp accuracy** from the video model needs testing on real footage.
- **Plan limits.** Voice and model availability vary by ElevenLabs plan, so check the tier.

## Open questions
- Audio-only deliverable, or also played over the video inside the app?
- Which ElevenLabs model? v2 and Flash allow timed pauses. v3 is more expressive, but its timing limits were not confirmed. Check the official v3 docs or test.
- One third-party tutorial claims ElevenLabs has no timestamp feature, but the official docs show one. The official docs are the better source, but verify on the actual account.

## Caveat
This plan comes from web searches and the app description, not from testing against a live account or reading the app's code. Confirm limits and pricing before building.

## Sources
- [Pipecat ElevenLabs settings](https://docs.pipecat.ai/api-reference/server/services/tts/elevenlabs)
- [ElevenLabs convert with timestamps](https://elevenlabs.io/docs/nl/api-reference/text-to-speech/convert-with-timestamps.md)
- [ElevenLabs streaming with timestamps](https://elevenlabs.io/docs/api-reference/streaming-with-timestamps)
- [ElevenLabs v3 audio tags](https://elevenlabs.io/blog/eleven-v3-audio-tags-precision-delivery-control-for-ai-speech)
- [ElevenLabs Dubbing Studio](https://elevenlabs.io/docs/product-guides/products/dubbing/dubbing-studio)
- [ffmpeg-python adelay example](https://github.com/kkroening/ffmpeg-python/issues/281)
- [Gemini video understanding](https://ai.google.dev/gemini-api/docs/video-understanding?hl=zh-cn)
- [Gemini timestamp accuracy blog](https://workalizer.com/insights/gemini/mastering-gemini-api-accurate-video-timestamps-for-enhanced-analysis)
