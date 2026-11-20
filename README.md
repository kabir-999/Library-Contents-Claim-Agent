# Library Contents Claim Agent

A live voice-and-vision agent that inventories and values a home library in **one continuous camera sweep** and produces a verifiable insurance contents claim packet (`claim_packet.json` + PDF + saved evidence frames).

* Perception is detectors + OCR (**YOLO, CLIP gate, PaddleOCR, Tesseract, OpenCV**). A vision LLM is only a budgeted fallback reader for covers OCR fails on, and its text must be corroborated by our own OCR and confirmed in a book catalogue.
* Prices come only from retrieved listings (URL + date + currency + condition). **No price is ever generated from model memory.**
* Totals are computed in Python from validated entries. **No LLM touches a number.**
* If scale, a price, or an identification can't be established, the value is left **blank** and the item goes to the **review queue**.

> **Status honesty:** read [DEVELOPMENT_STATUS.md](DEVELOPMENT_STATUS.md) for exactly what has and hasn't been verified. The pass-bar evaluation against a real room has to be run by you with a tape measure; the harness is built ([docs/evaluation.md](docs/evaluation.md)) but **no accuracy claims are made here**.

## Setup (about 15 minutes, mostly downloads)

**You need:** macOS or Linux, Python 3.12, Node 20+, `git`, and [Tesseract](https://github.com/tesseract-ocr/tesseract) (`brew install tesseract` on macOS, `apt install tesseract-ocr` on Ubuntu).
Optional: [Ollama](https://ollama.com) for a local LLM. API keys are optional to start, but **prices stay blank without at least one price key** (below).

```bash
# 1. config (never commit .env)
cp .env.example .env            # then fill in the keys you have: see "Keys" below

# 2. backend (terminal 1)
cd backend
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt        # ultralytics+torch, PaddleOCR, Whisper, CLIP: ~3 GB
uvicorn app.main:app --host 127.0.0.1 --port 8000
#   First start downloads model weights (YOLO, CLIP ~340 MB, PaddleOCR). Wait for the log line:
#   "book gate (CLIP) ready"  - until then no detection counts as a book.

# 3. frontend (terminal 2)
cd frontend && npm install && npm run dev      # http://localhost:3000
```

Open http://localhost:3000 in **Safari or Chrome**, choose the country, pick a camera (see below), press **Start**. `GET http://localhost:8000/api/health` shows which models and APIs are live. `docker`, cloud accounts and a database are **not** needed.

### Keys (`.env`)
| Key | What for | Free? |
|---|---|---|
| `SERPER_API_KEY` ([serper.dev](https://serper.dev)) | **Prices**: Google Shopping, new + used, any country (the only source for India) | free tier, no card |
| `EBAY_CLIENT_ID` / `EBAY_CLIENT_SECRET` ([developer.ebay.com](https://developer.ebay.com), *Production* keyset) | Prices on eBay marketplaces (US, UK, DE, ...; not India) | free |
| `GROQ_API_KEY` ([console.groq.com](https://console.groq.com)) | Text LLM to split noisy OCR + a vision model that reads covers when OCR fails; budgeted, never used for prices or totals | free tier |
| `GOOGLE_BOOKS_API_KEY` | Optional. Without it Google Books is skipped; Open Library (keyless) identifies books | free |

Set at least one of Serper or eBay, or every price stays blank (by design).

### Using your phone as the camera
Everything still runs on the laptop; the phone only supplies video and audio.
* **A. Phone as a webcam (simplest).** iPhone + Mac: Continuity Camera (same Apple ID, Wi-Fi and Bluetooth on, or plug in the USB cable). Any phone: install **Iriun Webcam** (iPhone/Android + Mac driver). Then on the start screen click **Detect cameras** and choose it. If Chrome doesn't list it, use Safari or fully quit and reopen Chrome.
* **B. Open the app on the phone.** Phones only allow the camera on HTTPS. Put the Mac and phone on one network that lets devices talk to each other (a phone hotspot works; many office/campus Wi-Fi networks block it). Run `./scripts/phone.sh` from the repo root (it makes a self-signed certificate and serves both servers over HTTPS), then on the phone open `https://<mac-ip>:8000/api/health` once and accept the certificate warning, then open `https://<mac-ip>:3000`. On iPhone also allow Settings > Privacy > Local Network for the browser. Tap **Start** to unlock the agent's voice; use **Enable / test voice** if you hear nothing.
* Room measurement needs the phone's motion sensors (option B). A laptop webcam or a webcam-style phone gives floor/wall area as "not available".

### Troubleshooting
| Symptom | Fix |
|---|---|
| "address already in use" on 3000/8000 | `lsof -ti tcp:3000,8000 | xargs kill`, then start again (`scripts/phone.sh` does this itself) |
| No books ever counted for the first seconds | CLIP is still loading; wait for "book gate (CLIP) ready" in the backend log |
| Every price is blank | no `SERPER_API_KEY` / eBay keys in `.env` (or the free quota is used up) |
| `__webpack_modules__ is not a function` | stale cache after a build: stop the frontend, `rm -rf frontend/.next`, `npm run dev` |
| No agent voice on a phone | tap Start (voice unlock needs a tap), check volume and the silent switch, tap *Enable / test voice* |
| Groq 429 | rate limit; the app backs off and keeps going without the large models |

### Run the tests
```bash
cd backend && source .venv/bin/activate && python -m pytest        # 81 tests: unit + stubbed end-to-end pipeline (no network, no models needed)
cd ../frontend && npx tsc --noEmit
```
Do not run `next build` while `npm run dev` is running; they share `frontend/.next`.

## What happens in a sweep
1. **Start claim**: country/currency, camera choice; the agent explains the sweep (spoken + written).
2. **Record + live preview** (default): the browser sends about **6-7 JPEG frames per second**; the server stores **every** one. In parallel a light **preview** looks at about 1 frame/s so the screen fills in ("Found so far"), and the agent speaks short guidance ("slow down", "glare", "I may have missed a shelf"), says what it logged, and asks the *original or print?* question for framed art. The preview never writes frames and never feeds the packet. Corrections you speak are applied live to the preview and recorded with a timestamp, to be applied to what was on screen at that moment in the final analysis. Pressing **Stop** (or saying "done") stops capture and releases the camera and microphone. Set `LCA_LIVE_PREVIEW=false` for recording only.
3. **Analyse** (`pipeline/batch.py`, after Stop): drop blurry and near-duplicate frames (at most `LCA_BATCH_MAX_FRAMES`, at least ~30) -> YOLO book + object detection over lists of frames -> text-line discovery (PaddleOCR detector) for spines the box detectors miss -> in-order tracking with image registration -> CLIP **book/not-book gate** -> every physical object becomes one track with one id.
4. **Identify and price, per id** (not per frame): Tesseract + PaddleOCR read the best sightings (zoomed in/out views included) -> clean consensus words query Open Library/Google Books -> candidates are scored against all raw reads, restricted to the author's works when an author is visible -> same-work duplicates merge. Unresolved books get a rescue pass and (budgeted) a vision-model scene read, always re-verified against the catalogue. Pricing runs lookups in parallel with time caps; items price as ranges; art needs appraisal.
5. **Packet**: measurement validation, room geometry, review queue, totals (Python only), JSON, PDF, evidence frames. Hard deadline 270 s of the 300 s budget.
6. **Read-back**: when the packet is ready the agent *speaks* a short summary (counts, totals, what needs review); nothing else is spoken after Stop.
7. **Review**: summary, book/item tables, review queue, same books priced for a second country, links to the PDF and the JSON.

`LCA_LIVE_PROCESSING=true` switches to fully live analysis (every frame processed during the sweep, no batch pass); it is heavier and not recommended.

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
| Book detector | YOLO COCO `book` + YOLO-World open-vocabulary `book`, PaddleOCR text-line discovery for missed spines | `LCA_YOLO_SPINE_WEIGHTS` accepts custom fine-tuned weights (we do not train) |
| Is it a book? | CLIP ViT-B/32 zero-shot (multi-zoom) + readable-text rule | rejects whiteboards, ceilings, chairs, devices |
| Object detector | YOLO-World open vocabulary (bookshelf, cabinet, lamp, rug, framed painting …), COCO fallback | |
| OCR | PaddleOCR (both 90° rotations) + Tesseract, multi-zoom, consensus across sightings | |
| Vision reader (budgeted) | Groq `qwen/qwen3.8-27b` transcribes a cover when OCR fails; accepted only if our own OCR corroborates and the catalogue confirms | never prices, never counts |
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

## Pricing rules for specific markets
* **India (`no_fx`)**: prices come only from local listings (Google Shopping via Serper, `gl=in`); a book with no local price stays **blank**, never converted from USD. Other markets: local first, converted (and labelled `converted`) only if no local price exists.
* Prices are asking prices of active listings (not sold prices). Used value needs at least two used listings (`LCA_MIN_LISTINGS`).
