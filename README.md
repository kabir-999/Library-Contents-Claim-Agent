# Library Contents Claim Agent

A live voice-and-vision agent that inventories and values a home library in **one continuous camera sweep** and produces a verifiable insurance contents claim packet (`claim_packet.json` + PDF + saved evidence frames).

* Perception is classical CV + detectors (**YOLO, PaddleOCR, OpenCV**), never an LLM.
* Prices come only from retrieved listings (URL + date + currency + condition). **No price is ever generated from model memory.**
* Totals are computed in Python from validated entries. **No LLM touches a number.**
* If scale, a price, or an identification can't be established, the value is left **blank** and the item goes to the **review queue**.

> **Status honesty:** read [DEVELOPMENT_STATUS.md](DEVELOPMENT_STATUS.md) for exactly what has and hasn't been verified. The pass-bar evaluation against a real room has to be run by you with a tape measure; the harness is built ([docs/evaluation.md](docs/evaluation.md)) but **no accuracy claims are made here**.

## Quick start (≈15 min, most of it downloads)

Prereqs: Python 3.12, Node 20+, (optional) [Ollama](https://ollama.com), a laptop/phone with camera + mic.

```bash
# 1. config
cp .env.example .env            # add keys you have (see "Configuration"); none are required to start

# 2. backend
cd backend
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt          # ultralytics+torch, paddle, whisper: ~2-3 GB
uvicorn app.main:app --host 127.0.0.1 --port 8000

# 3. (optional but recommended) local Qwen for orchestration + vision
ollama serve &                     # then:
ollama pull qwen2.5:7b-instruct && ollama pull qwen2.5vl:7b

# 4. frontend (new terminal)
cd frontend && npm install && npm run dev      # http://localhost:3000
```

Open http://localhost:3000, pick a country, allow camera + mic, press **Start**.
`GET http://localhost:8000/api/health` shows which models/APIs are live.

**Phone as the camera** (needed for room measurement: it uses the orientation sensors): browsers only expose camera/mic/sensors on HTTPS. Easiest: `cloudflared tunnel --url http://localhost:3000` (and a second tunnel for port 8000, then set `NEXT_PUBLIC_API_URL=https://<backend-tunnel>` before `npm run dev`; add the frontend origin to `LCA_CORS_ORIGINS`). A laptop webcam on `localhost` works for everything except room measurement.

### Run the tests
```bash
cd backend && source .venv/bin/activate && python -m pytest        # unit + stubbed end-to-end pipeline
cd ../frontend && npx tsc --noEmit && npx next build
```

## What happens in a sweep
1. **Start claim** – country/currency → sweep ID; permissions; spoken + written instructions.
2. **Live** – browser sends ~3 JPEG frames/s plus device orientation; mic audio is segmented by a client-side VAD. Per frame the server: scores blur/glare/exposure (OpenCV) → saves the frame → registers it to the previous one (ORB+RANSAC homography) → runs YOLO → tracks books across frames (mosaic coordinates, IoU + spine-centre matching, ≥2 hits to confirm) → emits guidance ("slow down", "move closer", "glare", "I may have missed a shelf", "re-capture W1-S2"). A background worker OCRs each track's best frame and verifies the text against Google Books + Open Library, so titles appear live.
3. **Stop** – independent async stages run: book identification → duplicate merge → measurement validation → pricing; non-book classification → pricing; room geometry; then review queue, totals, JSON, PDF. Hard deadline 270 s of the 300 s budget; stages that overrun are cut and recorded.
4. **Review** – dashboard, summary, packet (JSON/PDF, evidence frames with boxes, source links, second-country comparison, manual two-point calibration on any frame).

## Measurement: what is and isn't defensible
A pixel is only turned into a centimetre through `app/measure/calibration.py`, from one of: **a visible A4/Letter sheet** (homography from its four corners; auto-detected), a **two-point click** with a tape-measured distance, a **user-stated shelf height/width** ("this bookcase is 200 cm tall", scaled from the detector's box edge, confidence capped at 0.5), **depth data** or a **pinhole** model (interfaces exist; pinhole capped at 0.35 because intrinsics are approximate). Because scale depends on camera distance, a calibration made in frame A is transferred to book frames via frame-to-frame homographies (direct ORB match, else the tracked chain with a confidence discount). No calibration ⇒ `spine_height_cm`/`spine_thickness_cm` stay `null` and the book is queued (`Uncalibrated measurement`). Every measurement stores value, method, confidence, frame, calibration id. Truncated (frame-clipped) boxes are refused, not under-measured.

**Room** (`app/measure/room.py`): floor-plane ranging. With the phone at known height *h* and pitch θ below horizon at a floor/wall junction, distance = *h*/tan θ; heading from the compass gives a polygon (shoelace area, min-area rectangle for length/width, works for L-shapes). Ceiling height from the ceiling junction above the same corner. Pitch noise (±2° assumed) is propagated into the reported confidence. Needs: camera height, ≥3 floor marks, orientation sensors; else those fields are `null` + review entries. Wall area is gross (openings not subtracted). Shelved wall area = calibrated bookshelf boxes (w×h).

## Voice (honest description)
Whisper (faster-whisper, utterance-level) for STT; Kokoro (server) or the browser's `speechSynthesis` for TTS. It is **half-duplex with barge-in, not full-duplex**: a client VAD detects speech onset *while the agent is talking*, instantly cancels playback, tells the server to cancel the pending reply, and the new utterance is handled. Guidance never speaks over the user. Use headphones; with speakers the echo canceller may or may not prevent false barge-ins. Corrections go through a deterministic grammar first (`agent/corrections.py`, tested), local Qwen as fallback restricted to a closed action set; replies are templated from structured results so the LLM can't state invented counts.

## Models & configuration
All via `.env` (see `.env.example`).

| Role | Default | Notes |
|---|---|---|
| Local reasoning / correction fallback | Ollama `qwen2.5:7b-instruct` | `LCA_LOCAL_LLM_PROVIDER=none` disables |
| Hard text tasks (OCR→title split, candidate pick) | Groq `openai/gpt-oss-120b` (`GROQ_API_KEY`) | falls back to local, then to deterministic matcher |
| Visual interpretation of non-book objects | Ollama `qwen2.5vl:7b` | description/material only; brands must be re-confirmed by OCR |
| Book detector | YOLO COCO `book` class + classical spine splitting | `LCA_YOLO_SPINE_WEIGHTS` accepts custom fine-tuned weights (we do not train) |
| Object detector | YOLO-World open vocabulary (bookshelf, cabinet, lamp, rug, framed painting …), COCO fallback | |
| OCR | PaddleOCR, both 90° rotations tried | |
| STT / TTS | faster-whisper `base.en` / Kokoro or browser | |
| Metadata | Google Books + Open Library | verify title/author/publisher/ISBN |
| Prices | eBay Browse API (new+used, per-country marketplace), Serper.dev (free tier, recommended) or SerpAPI Google Shopping: new + used, split by retailer/title | **needs keys**; with none, prices stay blank |
| FX | Frankfurter (ECB), only for flagged conversions | |

**Pricing rules implemented:** listings must match the ISBN/title and are rejected if lots, bundles, signed, first editions, ARCs, ebooks, etc.; amount = trimmed median of listing prices, with the listing closest to the median as the cited URL; used value = median of *active used* listings (asking prices, not sold prices — say so when presenting); edition unverified ⇒ priced against the newest known edition ISBN as "equivalent copy" and noted; signed/first-edition/pre-1925/over-threshold ⇒ `needs_appraisal`, not priced, not in totals. Non-books get an interquartile **range** of ≥3 local listings with all URLs; artwork/portraits are `needs_appraisal` unless the owner says "print".

## Retained / changed / discarded from the reference
Reference: [`insurance_claim_live_agent_team`](https://github.com/Shubhamsaboo/awesome-llm-apps/tree/main/voice_ai_agents/insurance_claim_live_agent_team) (inspected; not forked or copied).

| | |
|---|---|
| **Retained (pattern)** | One browser↔FastAPI WebSocket carrying camera JPEG frames and 16 kHz mic audio; live agent state mirrored to the UI as it changes; spoken corrections applied to the record immediately; a background "agent team" that runs after live capture and produces a downloadable packet with evidence for **human review**; session cleanup; local-first defaults. |
| **Changed** | Gemini Live (one cloud multimodal model doing seeing + hearing + speaking) → **modular local-first stack** (YOLO/OCR/OpenCV perception, Whisper STT, Kokoro/browser TTS, Qwen + GPT-OSS for text). ADK workflow graph → plain `asyncio` stage graph with per-stage timing, deadline and degradation. In-memory sessions → **SQLite + filesystem evidence** (frames, crops, exports). Free-form packet (ZIP of notes) → **strict Pydantic contract + PDF**. Vanilla JS UI → Next.js/TS/Tailwind. Frames judged by the LLM → frames **measured and counted by CV** with quality scoring. |
| **Discarded** | `google-adk`/`google-genai`/Vertex dependencies, the optional avatar, incident sketches, mock policy directory & FNOL scripted intake, and *letting the LLM decide what is in the camera frame* (the reference trusts the live model's vision; the brief forbids that for counting/measuring). |

## Deviations from the specified stack (and why)
* **YOLO has no pretrained book-spine class.** COCO's `book` box is often a merged run, so `vision/spines.py` splits runs with vertical-edge energy + column-colour change (OpenCV). Accuracy on dense shelves is the biggest open risk; plug in fine-tuned spine weights via `LCA_YOLO_SPINE_WEIGHTS` (training is out of scope).
* **Next.js 15 / React 19 / Tailwind 4** (current stable at build time).
* **No keyless source for physical-book prices exists that is both legal and reliable**, so pricing needs at least one key: eBay, or the free Serper.dev / SerpAPI; Google Books `saleInfo` is deliberately *not* used (it prices ebooks, not replacement copies).
* **Full-duplex voice is not implemented** (see Voice).

## Repo map
`backend/app/{vision,identify,pricing,measure,voice,llm,agent,claim,pipeline,eval}` — one module per concern listed in the brief (book/object detection, OCR, identification, book/non-book pricing, image quality, calibration, spine + room measurement, STT/TTS, orchestration, calculations, JSON/PDF, storage). `frontend/` — the five screens. `docs/` — architecture, evaluation, failure log, cost/latency, demo script.
