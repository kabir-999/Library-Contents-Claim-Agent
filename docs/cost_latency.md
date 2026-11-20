# Cost and latency per sweep

Measured automatically for every sweep and written into `claim_packet.json` under `processing` (per-stage seconds) and `run_stats` (frame counts, avg detection/OCR ms, LLM token counts, API call counts, estimated cost). **Numbers below are to be copied from the real demo sweep's packet — none are reported until it exists.**

| Stage | Seconds | Status |
|---|---|---|
| _fill from `processing[]`_ | | |

Cost model (`app/metrics.py`): `groq_tokens/1e6 × LCA_COST_GROQ_PER_MTOK + serpapi_calls × LCA_COST_SERPAPI_PER_SEARCH`. The defaults in the code are **placeholders — set the current provider rates in `.env`**. eBay Browse, Google Books, Open Library and Frankfurter calls are counted but have no per-call charge on their free tiers; local models (YOLO, Paddle, Whisper, Qwen) cost electricity only.


## Measured on an Apple-silicon MacBook Air (2026-11-20, small captures)
| Stage | Latency |
|---|---|
| Book detection (COCO + open-vocab YOLO), per frame | ~46 ms (two models, ~22 ms each, batch does not help) |
| Object detection, per frame | ~20-40 ms |
| Book/not-book gate (CLIP), per new track | 15-120 ms |
| Spine OCR: Tesseract / PaddleOCR | ~0.12 s / ~0.8-1.5 s per crop |
| Catalogue identification (Open Library, parallel queries) | 0.9-2.1 s per book |
| Pricing, 4 books in parallel (Serper, India) | 4.0 s |
| Batch frame analysis, 328 recorded frames (117 s sweep) | 21 s (135 keyframes: load 3.3, detect 10.1, text discovery 4.5, tracking 3.3) |
| Stop -> packet, 54 s sweep | ~30 s before the latest speed fixes; not re-measured end to end |

Large-model calls are budgeted (live: 6/min, 30/sweep; end-of-sweep pass separate). $ per sweep: still placeholders in `app/metrics.py`; Serper bills per query (about 2-3 queries per priced book).
