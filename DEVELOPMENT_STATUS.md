# Development status

## Verified (automated, this machine)
* `cd backend && source .venv/bin/activate && python -m pytest` → **81 passed**: claim totals/exclusion rules, schema round-trip, calibration (A4 homography, two-point, cross-frame scale transfer), room geometry (rectangle, L-shape, ceiling/wall area), matcher anti-hallucination rules, price aggregation/filtering, guidance rules, correction grammar, tracking across a synthetic pan, storage, evaluation-scoring logic, and a **stubbed end-to-end pipeline** (detectors/OCR/APIs faked) that produces a schema-valid packet + PDF with every `frame_ref` resolving.
* `npx tsc --noEmit` and `npx next build` pass.
* Live smoke test: backend started, a sweep created, synthetic frames sent over the WebSocket, `stop` → packet + PDF produced via the real code path. YOLO (COCO weights) loads and runs on Apple MPS.

## Live-verified (2026-11-19, real APIs, no real books)
* Serper.dev Google Shopping (US/GB/IN) returns real rows with prices + retailer names; Groq `gpt-oss-120b` JSON call; Open Library lookup; identification of a clean OCR string -> `identified 0.95`; PaddleOCR on a rotated synthetic spine.
* Pricing "The Remains of the Day": US new 12.35 USD / used 19.0 USD (3 listings); GB new converted from US; IN unreliable (see below).
* Bugs found by these live runs and fixed: UK `gl` code (`uk` -> `gb`; `uk` silently returned USD results), bare `$` mislabelled as local currency, essays/study guides passing as book listings (now: title must lead the listing, essay words + Grin/Hausarbeiten blocked), ISBN queries for unverified editions returning unrelated books, single-listing prices (now need >=2).
* Limits found: Serper `link` is a Google Shopping page, not the retailer's own page; Google Books returns 429 without `GOOGLE_BOOKS_API_KEY` (Open Library covers); Ollama not running; India book prices still unreliable (one 5,324 INR result for a ~400 INR book) -> spot-check against the links.

## Changed since (2026-11-20)
Record-then-analyse by default (`LCA_LIVE_PROCESSING=false`; start-screen checkbox switches live inventory on), batch frame analysis (`pipeline/batch.py`),
CLIP book gate, Tesseract fast reader + PaddleOCR text-line discovery, vision-model scene read (budgeted), author-first catalogue filtering, `work_key` de-duplication,
India priced from local sources only (no FX), Stop releases camera/mic. Status of the sections below is otherwise unchanged: **no pass bar has been measured**,
no >=60-book room has been captured, no demo video/packet/ground-truth sheet exists yet.

**Update:** the default run now records every frame (~6-7/s) AND shows a light live preview (~1 frame/s: on-screen inventory, spoken guidance and narration, the original-or-print question); the packet still comes only from the batch pass after Stop, and the agent speaks a final summary. The remaining blockers are physical: real capture, ground truth, demo video (see `docs/capture_and_eval_checklist.md`).

## NOT verified (do not assume these work)
* **Accuracy on a real room: nothing measured.** No books were photographed, so detection/OCR/identification/measurement/pricing accuracy are unknown. The pass-bar table is unfilled (`docs/evaluation.md`).
* **No real packet/PDF/evidence from a real demo exists yet** (deliverables 4–6, 7 results, 10 numbers, failure-log section B depend on it). `data/eval/ground_truth.xlsx` is an empty template you must fill by hand.
* **Pricing returns nothing without keys**: set `EBAY_CLIENT_ID/SECRET` and/or `SERPAPI_API_KEY`. The eBay/SerpAPI/Groq/Google Books/Open Library clients were written against their documented APIs but **not exercised live**.
* PaddleOCR install was still in progress when last checked (`/private/tmp/paddle.log`); its 3.x API handling is untested. Without it, every book stays `unidentified` (OCR evidence required).
* Ollama is installed but not running and Qwen models are not pulled; YOLO-World needs CLIP (ultralytics may try to install it) and otherwise falls back to COCO classes (no rugs/lamps/cabinets/framed art).
* Whisper, Kokoro, browser VAD/barge-in, phone orientation sensors, camera in a browser: not exercised.
* Spine splitting of COCO `book` boxes on dense shelves is the largest accuracy risk.

## Pending / next steps
1. `ollama serve`; `ollama pull qwen2.5:7b-instruct qwen2.5vl:7b`; add API keys; confirm `/api/health`.
2. Fill `data/eval/ground_truth.xlsx` for a ≥60-book, 2-unit, ≥8-object room.
3. Record the sweep (`docs/demo_instructions.md`), run `app.eval.evaluate`, copy honest results + 3 worst errors into `docs/failure_log.md` §B and timings into `docs/cost_latency.md`.
4. Iterate on detection/splitting using the ground truth; consider fine-tuned spine weights via `LCA_YOLO_SPINE_WEIGHTS`.
5. Second-country comparison: after the sweep, Packet tab → pick country → "Price the first 10…" (needs pricing keys).

## Known limitations
Half-duplex voice with barge-in (not full-duplex); used value = active asking prices; wall area is gross; room needs a phone with orientation sensors, camera height and ≥3 marked corners; homography assumes planar shelf fronts; cost-rate defaults in `app/metrics.py` are placeholders.

## Commands
```bash
# backend (from repo root)
cd backend && source .venv/bin/activate && uvicorn app.main:app --host 127.0.0.1 --port 8000
# frontend (second terminal)
cd frontend && npm run dev          # http://localhost:3000
# tests / ground-truth template / evaluation
cd backend && python -m pytest
python -m app.eval.ground_truth ../data/eval/ground_truth.xlsx
python -m app.eval.evaluate ../data/eval/ground_truth.xlsx ../data/sweeps/<id>/exports/claim_packet.json ../data/eval/results
```
