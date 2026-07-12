# Vidya

**A lecture-grounded AI tutor.** Load any lecture (URL, uploaded file, or pasted text), then ask questions by typing or by voice. Vidya answers using only what is actually in the lecture, with every answer backed by a verbatim quote and timestamp from the source.

Built for the AI Office Hours (Tutor Bot) assignment.

---

## What it does

| Assignment expectation | How Vidya delivers |
| --- | --- |
| Take a transcript as input | URL, file upload, or paste. YouTube captions or Sarvam STT depending on source. |
| Answer only from that transcript | Strict system prompt plus a verbatim-quote guardrail: the LLM must return a quote it copied from the transcript, and the backend rejects the answer if the quote is not actually in the transcript. |
| Mimic a tutor explaining concepts | Chat interface with 5-turn conversation memory. Answers cite a timestamp. Speaks back in the same language as the student's question. |
| Bonus: STT + TTS + LLM pipeline | Sarvam Saarika (STT), Sarvam-105b (LLM), Sarvam Bulbul (TTS). Fully voice-driven if you want it. |

### Beyond the bonus

- **Multimodal understanding.** Optional Gemini 1.5 Flash key lets Vidya watch the video too. Frames are sampled every 10s and described (diagrams, chart axes, whiteboard equations, slide text). Visual entries are interleaved with audio on a single timestamped timeline. The tutor can cite visual moments the same way it cites audio moments.
- **Pre-reads.** Attach supplementary reading materials (webpages, PDFs, plain text) to any lecture, either by URL or by file upload. Vidya extracts and chunks the text, merges it into the same timeline as the audio and visuals, and can cite from it. When you load a YouTube URL, any links found in the video's description are surfaced as one-click "Add as pre-read" suggestions. Answers whose supporting quote came from a pre-read are labeled *"From pre-read: source-name"* instead of a timestamp.
- **Any public video URL.** yt-dlp supports 1000+ sites (YouTube, Vimeo, Dailymotion, TED, Twitch VODs, direct .mp4 links, and more).
- **Chrome side-panel extension** for gated content that has no public URL. It can either send the current tab's URL to Vidya, or capture the tab's audio directly (works on private LMS platforms, internal training tools, etc.). DRM-protected sites like Netflix are detected and warned about.
- **Anti-hallucination guardrail.** JSON-formatted responses, verbatim-quote check, block-on-mismatch. Every answer displays the quote and its timestamp.
- **Prompt injection defence.** The transcript is fenced with random per-request delimiters and the model is told to treat everything inside as inert data.
- **Long-lecture retrieval.** For lectures over roughly 3,000 words, a BM25 retriever picks the most relevant timeline windows for each question so nothing overflows the model context.
- **Chat memory** across up to 5 prior turns, so follow-up questions work naturally.
- **Persistence.** Transcript and chat survive a browser refresh via localStorage.
- **Retries and rate-limit handling** on both Sarvam and Gemini calls.
- **Answer-language auto-detection** for TTS across 10 Indian languages plus English.

---

## Architecture

```
Browser (index.html)
     |
     |  fetch /api/*
     v
Flask backend (app.py)
     |
     +--> Sarvam Saarika       (speech to text: mic, file chunks, audio-only URL downloads)
     +--> Sarvam-105b          (grounded question answering)
     +--> Sarvam Bulbul        (text to speech)
     +--> Gemini 1.5 Flash     (optional: frame-by-frame vision descriptions)
     +--> yt-dlp               (video URL to media file)
     +--> youtube-transcript-api (fast path: existing YouTube captions)
     +--> imageio-ffmpeg       (audio extraction, chunking, frame sampling)

Chrome extension (extension/)
     |
     |  postMessage
     v
Same backend, via an embedded side-panel iframe.
Also captures tab audio for gated video sites.
```

### Anti-hallucination guardrail (the important bit)

Every question triggers this sequence:

1. If the timeline is longer than ~3,000 words, BM25 retrieves the top 8 most relevant timeline windows for the question. Otherwise the full timeline is used.
2. The system prompt fences the timeline between random delimiters and instructs the model to treat anything inside as literal data, never as instructions.
3. The model is required to return a JSON object with `in_scope`, `answer`, and `supporting_quote`.
4. The backend parses the JSON. If parsing fails, the answer is replaced with the "not covered" fallback.
5. If `in_scope` is false or no quote was returned, the fallback is used.
6. The backend normalises the quote (lowercase, strip punctuation, collapse whitespace) and checks whether it is a substring of any timeline entry. If it is not, the model hallucinated its citation, so the answer is discarded and the fallback is used.
7. If the quote is found, the entry it came from provides the displayed timestamp.

This means a hallucinated answer that fakes a quote from the lecture is always caught and blocked before the student sees it.

---

## Requirements

- Windows, macOS, or Linux
- Python 3.10+
- A **Sarvam API key** from https://dashboard.sarvam.ai (free tier is enough for a demo)
- Optional: a **Google Gemini API key** from https://aistudio.google.com/apikey (free tier: 1500 requests/day) to unlock visual understanding
- Chrome or Edge for the extension

---

## Setup

```bash
# 1. Clone
git clone https://github.com/<your-username>/vidya.git
cd vidya

# 2. Install dependencies
python -m pip install flask flask-cors requests python-dotenv \
                     youtube-transcript-api imageio-ffmpeg yt-dlp \
                     pypdf beautifulsoup4

# 3. Run the server
python app.py
```

Then open http://127.0.0.1:5000 in your browser.

Paste your Sarvam API key into the setup box and click Save. Optionally paste a Gemini key too.

---

## Using the website

1. **Load a lecture** in one of three ways:
   - **Video URL**: paste any public link and click Load. YouTube URLs use captions when available (fast). Everything else is downloaded and transcribed with Sarvam. Tick "Include visuals" if you also want Gemini to describe the video frames.
   - **File upload**: drop an audio or video file. Same pipeline as a URL, without the download step.
   - **Paste text**: paste an existing transcript directly.
2. **Ask a question** by typing or by clicking Record to speak.
3. **Read the answer** with its supporting quote and timestamp. Click Speak to hear it in the source's language.
4. Ask **follow-up questions**; the last 5 turns of chat are used as context.

Try an out-of-scope question after loading a lecture to see the guardrail block the answer and return "not covered in this lecture."

---

## Installing the Chrome extension

For private video platforms (internal LMS, gated course sites, corporate training tools) that Vidya cannot download directly:

1. Open `chrome://extensions` in Chrome or Edge.
2. Toggle **Developer mode** on (top right).
3. Click **Load unpacked** and select the `extension` folder inside this repo.
4. Pin the Vidya extension from the puzzle-piece menu.
5. Make sure `python app.py` is running.
6. Open any video page, click the Vidya icon, and the side panel opens.
7. Either click **Use this page's URL** to send the current URL to Vidya, or click **Record tab audio**, play the video, then click Stop to send the captured audio for transcription. The panel then talks to your local Vidya server.

DRM-protected sites (Netflix, Prime Video, Disney+, Hotstar, etc.) will be flagged with a warning because their audio is protected and cannot be captured.

---

## Edge cases Vidya handles

| Case | Handling |
| --- | --- |
| Long lectures over ~3,000 words | BM25 retrieval fetches only the most relevant timeline windows for each question. |
| Prompt injection inside the transcript | Random per-request delimiters plus explicit instructions to treat the transcript as inert data. |
| Model hallucinates a citation | Verbatim-quote check against the timeline. Answer discarded if the quote does not appear. |
| Follow-up questions | Last 5 Q&A turns are included as chat history. |
| Sarvam or Gemini rate limits | Exponential backoff, one retry per chunk, cap of 3 concurrent STT and 4 concurrent vision calls. |
| Partial STT failure | Successful chunks are kept and reported; failed chunks are counted and skipped. |
| Sessions across a browser refresh | Timeline and chat are persisted in localStorage. |
| Multilingual answers | Answer language is detected from the reply script (Devanagari, Bengali, Tamil, etc.) and passed to Bulbul so the spoken output is in the right voice. |
| Silent segments or caption-less videos | If audio yields nothing but visuals do (with Gemini enabled), Vidya still works from visuals alone. If neither yields content, the user is told honestly. |
| Videos over the size limit | 3 hour hard cap, 1 hour soft warning before processing. |
| DRM sites in the extension | Detected against a known-host list and warned about before recording. |
| Malformed model output | JSON parse failure returns the "not covered" fallback rather than surfacing broken text. |

---

## Project layout

```
vidya/
├── app.py                     # Flask backend (all endpoints and pipelines)
├── index.html                 # Main website (chat UI, purple/blue/white theme)
├── extension/
│   ├── manifest.json          # Chrome Manifest V3
│   ├── background.js          # Service worker
│   └── sidepanel.html         # Side-panel UI with tab-audio capture
├── .env                       # Auto-created; holds SARVAM_API_KEY and GEMINI_API_KEY
├── README.md
└── .gitignore
```

---

## Design decisions

- **Sarvam-105b** for question answering. Its 8k+ context handles typical lecture sizes; retrieval covers the rest.
- **Sarvam Saarika 2.5** for STT. Chunking at 30s keeps each call well inside the per-request duration limit and enables parallel transcription.
- **Sarvam Bulbul 2** for TTS with automatic language routing based on the answer's script.
- **Gemini 1.5 Flash** for vision because it has a generous free tier and returns concise, structured descriptions for lecture-style frames.
- **yt-dlp** rather than a YouTube-only downloader because the assignment naturally extends to any public video source.
- **BM25 with per-request idf** rather than semantic embeddings because it has no dependency footprint, no API cost, and is more than good enough at picking a handful of relevant lecture windows.
- **JSON output plus quote check** rather than a second LLM verification call because it costs half as many tokens and catches the same class of hallucination.
- **Chrome side panel** (Manifest V3) rather than a popup because a panel stays open while the user watches the video.

---

## What is not included

- Streaming answers (works fine without it for the demo)
- Speaker diarisation (Sarvam supports it; not needed for lectures)
- KaTeX rendering for math (cosmetic)
- User accounts (this runs locally)

---

## Deploying a live demo

If you want your interviewer to click a link instead of installing Python locally, the fastest path is Render.com. Everything you need is in the repo.

### One-time setup

1. Push the repo to GitHub (private is fine).
2. Sign up at https://render.com with your GitHub account.
3. Click **New +** → **Web Service** → connect your GitHub repo.
4. Fill in:
   - **Environment**: Python 3
   - **Build command**: `pip install -r requirements.txt`
   - **Start command**: `gunicorn app:app --bind 0.0.0.0:$PORT --workers 2 --timeout 300`
   - **Instance type**: Free
5. Under **Environment Variables**, add:
   - `SARVAM_API_KEY`
   - `GEMINI_API_KEY` (optional)
   - `COHERE_API_KEY` (optional)
6. Click **Create Web Service**. First build takes 2 to 5 minutes.
7. Render gives you a URL like `https://vidya-tutor-bot.onrender.com`. Send this to your interviewer.

### requirements.txt

Create a `requirements.txt` at the repo root with:

```
flask
flask-cors
requests
python-dotenv
youtube-transcript-api
imageio-ffmpeg
yt-dlp
pypdf
beautifulsoup4
gunicorn
```

### Warning about live deploys

Anyone who opens the link uses your API keys. Set spending caps on Sarvam and Gemini before sharing. Consider adding a password check to the app if you plan to share widely.

## Licence

MIT.

## Credits

- Sarvam AI for the STT, TTS, and LLM APIs
- Google DeepMind for Gemini
- yt-dlp maintainers
- youtube-transcript-api
- imageio-ffmpeg
