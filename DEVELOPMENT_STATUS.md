# Development status

Last updated 2026-11-20. This file says what has and has not been verified. **No pass bar has been measured on a real room yet.**

## Verified (automated, this machine)
* `cd backend && source .venv/bin/activate && python -m pytest` -> **81 passed** (no network, no models needed): claim totals and exclusion rules, schema round-trip, calibration (A4 homography, two-point, cross-frame scale transfer), room geometry, matcher anti-hallucination rules, price aggregation/filtering, guidance rules, correction grammar, tracking, storage, evaluation scoring, a stubbed end-to-end pipeline (schema-valid packet + PDF, every `frame_ref` resolves), keyframe selection, LLM call budget, work-key de-duplication, multi-frame consensus helpers.
* `cd frontend && npx tsc --noEmit` passes.
* Replays of real recorded sweeps through the real code: record -> Stop -> batch analysis -> identification -> pricing -> packet works end to end (about 18-30 s after Stop for 2-minute sweeps; stage timings in `docs/cost_latency.md`).

## Live-verified with real services / hardware (small captures only)
* **Services:** Serper Google Shopping (US/GB/IN), eBay Browse (production keyset; token retry), Groq text model, Groq vision model (`qwen/qwen3.8-27b`), Open Library. Google Books is skipped unless a key is set (keyless requests are rate-limited to uselessness).
* **Models:** YOLO (COCO) and YOLO-World on Apple MPS, CLIP ViT-B/32 book gate, PaddleOCR 3.x (full and text-detector-only), Tesseract.
* **Captures:** about 2 books by one author on a desk/office, plus items (shelf, chair, whiteboard, devices). Both books identified from tilted, partly cut-off spines after the fixes in `docs/failure_log.md` section A2.
* **Phone:** the app opened over HTTPS on an iPhone through a phone hotspot (`scripts/phone.sh`); the camera, then the agent's voice, were confirmed working by the user; results page rendered on the phone.
* **Pricing:** India is priced from local listings only (no FX): for example a ~INR 400-1,400 range per book; unfindable titles stay blank.

## Current behaviour (differs from the first version)
* **Record + live preview (default):** every frame (~6-7/s) is stored; a light preview (~1 frame/s, from memory) fills "Found so far", lets the agent speak guidance and narration and ask *original or print?* for art; it never feeds the packet. The packet comes from the **batch analysis after Stop** (`pipeline/batch.py`), with one unique id per object. Stop releases the camera and mic. After Stop, only the final summary is spoken.
* **Settings:** `LCA_LIVE_PREVIEW` (true), `LCA_LIVE_PROCESSING` (false), `LCA_BATCH_MAX_FRAMES` (200), `LCA_BOOK_GATE` (true), `LCA_FAST_DETECT` (true), `LCA_EDGE_PROFILE` (false). See `.env.example`.
* **Identification:** Tesseract + PaddleOCR (zoom in/out, several sightings, consensus words), author-first catalogue filtering, `work_key` de-duplication, rescue pass, budgeted vision-model scene read (always re-verified against the catalogue).

## NOT verified - do not assume these work
* **Accuracy on a real room: nothing measured.** No >=60-book, 2-shelving-unit room has been captured. Book count, title accuracy, spine dimensions, price accuracy, non-book recall, floor area and wall area against ground truth are all **unmeasured**; `data/eval/ground_truth.xlsx` is an empty template.
* **Metric scale and room area never checked against a ruler or tape**: the A4-sheet, two-point and known-dimension methods and the floor-plane room ranging are only unit-tested (synthetic) and replayed on small captures. Room area needs a phone with motion sensors, camera height and >=3 marked corners; a laptop webcam gives "not available".
* **No demo video, no real claim packet/report/evidence frames for a real room exist yet** (see `docs/capture_and_eval_checklist.md`).
* **Live preview in a real browser sweep** was verified by replaying recorded frames through the live agent, not by a full live phone sweep with the preview on.
* Whisper speech-to-text and barge-in on the phone were not confirmed (the user confirmed the agent's voice output only). Voice is half-duplex with barge-in, not full-duplex.
* Smart glasses: the glasses view and the small-model profile (`LCA_EDGE_PROFILE`, one crop tested) exist; no glasses were tried, and many consumer glasses have no browser camera access.
* Ollama is optional and not running here; Qwen-VL item descriptions therefore fall back to category-level descriptions.
* Prices are asking prices of active listings; used value needs >=2 used listings, so it is often blank. Serper's `link` is a Google Shopping page, not always the retailer's own page.
* Cost per sweep: the packet's estimate uses placeholder rates in `app/metrics.py`; real bills not checked.

## Known weak spots
Spines seen only as a thin fragment cannot be identified without guessing (by design they stay "unidentified"); dense shelves with dozens of spines in view are untested; dark or glossy spines and tilted shelves are the hardest cases; two identical copies of one title merge into one entry (by request, to avoid duplicates); wall area is gross (doors/windows not subtracted).

## Pending / next steps (mostly physical)
1. Capture a real room and fill `data/eval/ground_truth.xlsx` by hand BEFORE tuning (`docs/capture_and_eval_checklist.md`).
2. Record the sweep and the one-take demo; run `python -m app.eval.evaluate ...`; copy the honest results into `docs/evaluation.md`, the three worst errors into `docs/failure_log.md` section B and the timings into `docs/cost_latency.md`.
3. `python scripts/assemble_submission.py --sweep <sweep_id> --video <recording.mp4>` builds `submission/` (Demo Video, claim_packet.json, Claim Report.pdf, Ground Truth & Results.xlsx, Architecture.pdf, Failure Log.pdf, Evidence Frames/). `Architecture.pdf`, `Failure Log.pdf` (placeholder timings) and the empty workbook already exist.
4. Stop tracking `data/` (`git rm -r --cached data`), make the GitHub repo private and share it, regenerate the Groq/Serper/eBay keys that were pasted into chat.

## Commands
```bash
# backend (from repo root)
cd backend && source .venv/bin/activate && uvicorn app.main:app --host 127.0.0.1 --port 8000
# frontend (second terminal)
cd frontend && npm run dev                      # http://localhost:3000   (do not run `next build` at the same time)
# phone over HTTPS (frees ports 3000/8000, prints the links to open on the phone)
./scripts/phone.sh
# tests
cd backend && python -m pytest && cd ../frontend && npx tsc --noEmit
# ground-truth template / evaluation / submission folder
cd backend && python -m app.eval.ground_truth ../data/eval/ground_truth.xlsx
python -m app.eval.evaluate ../data/eval/ground_truth.xlsx ../data/sweeps/<id>/exports/claim_packet.json ../data/eval/results
cd .. && python scripts/assemble_submission.py --sweep <id> --video <recording.mp4>
```
