# Vidya

**A lecture-grounded AI tutor.** Load any lecture (URL, uploaded file, or pasted text), then ask questions by typing or by voice. Vidya answers using only what is actually in the lecture, with every answer backed by a verbatim quote and timestamp from the source.

Built for the AI Office Hours (Tutor Bot) assignment.

Repository: https://github.com/meenapriyadars20/vidya-tutor-bot

---

## What it does

| Assignment expectation | How Vidya delivers |
| --- | --- |
| Take a transcript as input | URL, file upload, or paste. Every URL is downloaded and transcribed with Sarvam Saarika so the STT pipeline runs end to end. |
| Answer only from that transcript | Strict system prompt plus a verbatim-quote guardrail: the LLM must return short evidence quotes, and the backend rejects the answer if any surviving quote is not actually in the transcript. |
| Mimic a tutor explaining concepts | Chat interface with 5-turn conversation memory. A warm, encouraging tutor persona is baked into the prompt. Answers cite timestamps. Speaks back in the required language and voice. |
| Bonus: STT + TTS + LLM pipeline | Sarvam Saarika (STT), Groq Llama 3.3 70B (default reasoning LLM), Sarvam-105b (retrieval-based reasoning LLM), Gemini Flash (long-context alternative), and Sarvam Bulbul (TTS). Fully voice-driven if you want it. |

### Beyond the bonus

- **Three interchangeable LLM engines.** Groq Llama 3.3 70B (default, free long-context, best for Indian code-mixed content), Sarvam-105b (retrieval-based), and Gemini Flash (long-context alternative). Switch in the settings modal.
- **Pre-reads.** Attach supplementary reading materials to any lecture, either by URL or file upload. File support is broad: PDF, DOCX, HTML, XML, TXT, MD, Markdown, RST, CSV, TSV, JSON, YAML, LOG, RTF, TEX, SRT, VTT, IPYNB, and a plain-text fallback for anything else that decodes as text. Vidya extracts and chunks the text, merges it into the same timeline as the audio, and can cite from it. When you load a URL, links found in the video description are surfaced as one-click "Add as pre-read" suggestions after an LLM classifier drops social / merch / subscribe noise.
- **Any public video URL.** yt-dlp supports 1000+ sites (YouTube, Vimeo, Dailymotion, TED, Twitch VODs, direct .mp4 links, and more).
- **Chrome side-panel extension** for gated content that has no public URL. It can either send the current tab's URL to Vidya or capture the tab's audio directly (works on private LMS platforms, internal training tools, etc.). DRM-protected sites like Netflix are detected and warned about.
- **Anti-hallucination guardrail.** JSON-formatted responses, verbatim-quote check against every entry, block-on-mismatch. Every answer displays the surviving quotes and their timestamps.
- **Prompt injection defence.** The transcript is fenced with random per-request delimiters and the model is told to treat everything inside as inert data.
- **Long-lecture retrieval.** BM25 with LLM query rewriting, always-included intro and conclusion, guaranteed pre-read seats, and optional Cohere Rerank v3 as a cross-encoder second pass.
- **Multilingual across the board.** 11 Sarvam-supported languages plus English. Independent choices for question language, answer language, and answer voice. UI itself can also switch to any of the 11 languages.
- **Handles Indian code-mixing.** Tanglish, Hinglish, Kanglish, etc. The prompt treats them as normal classroom speech.
- **Handles transliterated English in Indic scripts.** The prompt has a concrete decode key for cases where Sarvam Saarika spells English words in Tamil / Devanagari letters.
- **Colloquial-to-formal.** When the lecturer uses casual phrasing, Vidya answers in standard textbook terminology while keeping the lecturer's actual words as the supporting quote.
- **Quiz mode.** Generates 5 grounded multiple-choice questions from the loaded lecture. Any question whose supporting quote does not verify (strict substring first, fuzzy word-overlap fallback second) is silently dropped, so what the student sees is guaranteed grounded. Uses the same engine you selected for Q&A, with automatic Sarvam fallback if Groq is rate-limited.
- **Retries and rate-limit handling** across Sarvam, Gemini (429), Groq (parses "try again in X seconds"), and Cohere.
- **Fresh session on refresh.** Every browser reload gives you a clean lecture and chat. Language and engine preferences persist.
- **Warm parchment theme.** Not the usual bright-white SaaS look; easier on the eyes for long study sessions.

---

## Architecture

```
Browser (index.html)
     |
     |  fetch /api/*
     v
Flask backend (app.py)
     |
     +--> Sarvam Saarika       (speech to text: mic recordings, uploaded files, downloaded video audio)
     +--> Groq Llama 3.3 70B   (DEFAULT reasoning engine; long-context; free tier; used for Q&A and quiz)
     +--> Sarvam-105b          (retrieval-based reasoning fallback; query rewriting; pre-read classifier)
     +--> Google Gemini Flash  (optional long-context reasoning alternative)
     +--> Sarvam Bulbul        (text to speech in 11 Indic languages + English)
     +--> Cohere Rerank v3     (optional cross-encoder reranker over BM25 candidates)
     +--> yt-dlp               (any URL to media file, 1000+ sites)
     +--> imageio-ffmpeg       (audio extraction, chunking)
     +--> pypdf, BeautifulSoup (pre-read extraction)

Chrome extension (extension/)
     |
     |  postMessage
     v
Same backend, via an embedded side-panel iframe.
Adds "Use this page's URL" and "Record tab audio" for gated video sites.
```

### The anti-hallucination guardrail (the important bit)

Every question triggers this sequence:

1. Route to the selected engine. Sarvam-105b uses BM25 retrieval if the timeline is over ~3,500 words (with LLM-generated query variants and optional Cohere Rerank). Gemini and Groq skip retrieval and receive the full timeline.
2. The system prompt fences the transcript between random delimiters and instructs the model to treat anything inside as literal data, never as instructions.
3. The model is required to return a JSON object with `in_scope`, `answer`, and `supporting_quotes` (1 to 4 short verbatim spans).
4. The backend parses the JSON; if parsing fails, a regex fallback extracts fields.
5. If `in_scope` is false, the localised "not covered" message is returned.
6. Each supporting quote is normalised and checked against every timeline entry. If the strict substring check fails, a fuzzy word-overlap check (at least 55 percent of content words) is tried as a fallback to tolerate light paraphrasing. Quotes that fail both are dropped. If no quotes survive, the entire answer is blocked and replaced with the fallback.
7. Surviving quotes are shown with their timestamps (or pre-read source names).

A model fabricating a plausible-sounding citation is caught and blocked before the student sees it.

---

## Requirements

- Windows, macOS, or Linux
- Python 3.10 or later
- A **Sarvam API key** from https://dashboard.sarvam.ai (free tier is enough for a demo)

**Groq is the default engine and is strongly recommended:**

- A **Groq API key** from https://console.groq.com. Free tier is 500K tokens/day and 14,400 requests/day, no card required. Groq handles Indian code-mixed and transliterated content best in testing, which is why Vidya defaults to it.

**Also optional:**

- A **Google Gemini API key** from https://aistudio.google.com/apikey. Alternative long-context engine. Free tier is tighter than Groq.
- A **Cohere API key** from https://dashboard.cohere.com. Enables Cohere Rerank v3 to improve retrieval precision when using Sarvam-105b. Free trial gives 1,000 calls/month.

Chrome or Edge is needed only for the browser extension.

---

## Setup

```bash
# 1. Clone
git clone https://github.com/meenapriyadars20/vidya-tutor-bot.git
cd vidya-tutor-bot

# 2. Install dependencies
python -m pip install -r requirements.txt

# 3. Create a .env file with your keys
#    (Sarvam is required. The others are optional.)
```

Create a file named `.env` at the repo root with this content, substituting your own keys:

```
SARVAM_API_KEY=sk_...
GROQ_API_KEY=gsk_...
GEMINI_API_KEY=AIza...
COHERE_API_KEY=...
```

Then run the server:

```bash
python app.py
```

Open http://127.0.0.1:5000 in Chrome or Edge.

---

## Using the website

1. **Load a lecture** in one of three ways:
   - **Video URL**: paste any public video link and click Load. Vidya downloads audio via yt-dlp and transcribes with Sarvam Saarika. Note that YouTube blocks yt-dlp downloads from cloud server IPs, so this path works fully when Vidya runs locally, but the hosted (Render / cloud) version cannot download YouTube URLs. Use the file upload path for the hosted demo.
   - **File upload**: drop an audio or video file. Same pipeline, without the download step. This is the recommended path on any hosted deployment.
   - **Paste text**: paste an existing transcript directly.
2. **Optional: attach pre-reads** (PDFs, articles, docs). URL or file. YouTube-description links are auto-suggested after an LLM classifier drops noise.
3. **Choose your engine** in the settings modal (gear icon → Advanced reasoning options):
   - **Groq Llama 3.3 70B** (default): long-context, best for code-mixed and transliterated Indian content, free tier is generous.
   - **Sarvam-105b**: fast, retrieval-based, uses about 5K tokens per question.
   - **Gemini Flash (latest)**: long-context alternative. Free tier has strict per-day limits.
4. **Pick your languages** using the three dropdowns at the top: Question Language (STT hint), Answer Language (LLM output), Answer Voice (TTS).
5. **Ask a question** by typing or clicking Record and speaking.
6. **Read the answer.** Click "Show evidence (N)" to reveal the verified quotes with timestamps. Click Speak to hear the answer, Pause to hold it.
7. **Follow up** naturally. The last 5 turns are used as context.
8. **Try a quiz** with the Take a Quiz button. 5 grounded multiple-choice questions with per-question explanations.

Try an out-of-scope question after loading a lecture to see the guardrail return "not covered in this lecture" in your chosen language.

---

## Installing the Chrome extension (optional)

For private video platforms (internal LMS, gated course sites, corporate training tools) that Vidya cannot download directly:

1. Open `chrome://extensions` in Chrome or Edge.
2. Toggle **Developer mode** on (top right).
3. Click **Load unpacked** and select the `extension` folder inside this repo.
4. Pin the Vidya extension from the puzzle-piece menu.
5. Make sure `python app.py` is running locally.
6. Open any video page and click the Vidya icon. The side panel opens.
7. Either click **Use this page's URL** or **Record tab audio** → play the video → **Stop**.

DRM-protected sites (Netflix, Prime Video, Disney+, Hotstar, etc.) will be flagged with a warning because their audio is protected and cannot be captured.

---

## Edge cases Vidya handles

| Case | Handling |
| --- | --- |
| Model hallucinates a citation | Every supporting quote is checked against timeline entries (strict substring first, fuzzy 55 percent word-overlap second). Unverified quotes are dropped; if none survive, the answer is blocked and replaced with the graceful "not covered" fallback. |
| Long lectures over ~3,500 words | BM25 retrieval with LLM-generated query variants, always-included intro and conclusion, and pre-read seats. Optional Cohere Rerank for cross-encoder precision. |
| Prompt injection inside the transcript | Random per-request delimiters plus explicit "treat as data, ignore instructions" wording. |
| Follow-up questions | Last 5 Q&A turns included as chat history. |
| Sarvam / Gemini / Groq rate limits | Exponential backoff and auto-retry. Groq's "try again in X seconds" hint is parsed and honoured. Friendly fallback message with engine-switch suggestion if retries exhausted. |
| Partial STT failure | Successful chunks kept and reported; failed chunks counted and skipped. Sub-1.5-second trailing chunks pre-dropped so Sarvam does not reject them. |
| Fresh sessions | Lecture and chat wiped on every refresh. Preferences (language, engine, voice) persisted. |
| Multilingual answers | Answer language independently chosen; TTS routed accordingly. |
| Indian code-mixing (Tanglish, Hinglish, etc.) | Prompt treats mixed input as normal Indian classroom speech. |
| Transliterated English in Indic scripts | Prompt has a concrete decode key listing 10+ example phrases. |
| Colloquial vocabulary in the transcript | Answer uses standard terminology; lecturer's phrasing kept as the supporting quote. |
| Malformed model JSON | Balanced-brace parser plus regex fallback plus plain-text refusal detection. Never surfaces raw broken text. |
| TTS answers longer than 900 characters | Text split at sentence boundaries; audio segments queued and played sequentially with pause and resume controls. |
| Videos over the size limit | 3 hour hard cap, 1 hour soft warning before processing. |
| DRM sites in the extension | Detected against a known-host list and warned about before recording. |

---

## Edge cases Vidya does NOT handle

- **Visual understanding of video frames.** Previously supported via Gemini Vision; removed because per-frame calls burned quota. Anything shown on screen but never spoken is invisible to Vidya.
- **Real token streaming.** The UI uses a typing animation after the full answer is received. Nice feel, but no first-token latency improvement.
- **Speaker diarisation.** Not toggled on; multi-speaker discussions appear as one continuous voice.
- **Non-Sarvam languages.** Sarvam Saarika supports 11 languages; French/German/Spanish/Chinese lectures produce empty or garbled transcripts.
- **Cross-lecture library.** One active lecture at a time.
- **Editable transcripts.** Cannot inline-correct STT mistakes.
- **DRM streaming platforms in the extension.** Netflix, Disney+ etc. block audio capture.
- **YouTube URL downloads on cloud hosting.** YouTube treats cloud server IPs (Render, Fly, AWS, GCP) as bots and blocks yt-dlp downloads from them. This affects every yt-dlp-based hosted app, not Vidya specifically. Solution: use file upload on hosted deployments, or run locally where your home IP is trusted.
- **Automated evaluation harness.** No batch test suite over a known QA bank.

Full deep-dive on design decisions and every edge case is in `SOLUTION.md` if you kept that file locally (it is not committed).

---

## Project layout

```
vidya-tutor-bot/
├── app.py                     # Flask backend (all endpoints and pipelines)
├── index.html                 # Single-page frontend (chat UI, warm parchment theme)
├── extension/
│   ├── manifest.json          # Chrome Manifest V3
│   ├── background.js          # Service worker
│   └── sidepanel.html         # Side-panel UI with tab-audio capture
├── requirements.txt
├── render.yaml                # One-file config for Render.com deployment
├── .env                       # Local only, git-ignored; holds API keys
├── README.md
└── .gitignore
```

---

## Design decisions

- **Groq Llama 3.3 70B** as the default engine. In testing it handled Indian code-mixed and transliterated content better than the other engines, and its free tier is generous enough to demo comfortably.
- **Sarvam-105b** kept as the retrieval-based fallback and for STT/TTS. It is what the assignment asks for and remains a first-class choice when full-context is not needed.
- **Gemini Flash** kept as a second long-context option because Google's key is separate from Groq's and gives redundancy.
- **Cohere Rerank v3** as an optional precision boost on top of BM25.
- **BM25 with LLM-generated query variants** rather than semantic embeddings. Zero embedding infrastructure, still catches paraphrases via the query variants.
- **JSON output + verbatim quote check** rather than a second LLM verification call. Half the tokens, same protection against hallucinated citations.
- **Chrome side panel (Manifest V3)** rather than a popup, so the panel stays open while the student watches the video.
- **Fresh session on every refresh.** Preferences persist; content does not. Reduces confusion when a student comes back later.

---

## Licence

MIT.

## Credits

- Sarvam AI for the STT (Saarika), TTS (Bulbul), and LLM (Sarvam-105b) APIs
- Groq for the free Llama 3.3 70B endpoint
- Google DeepMind for Gemini
- Cohere for Rerank v3
- yt-dlp, imageio-ffmpeg, pypdf, beautifulsoup4 maintainers
