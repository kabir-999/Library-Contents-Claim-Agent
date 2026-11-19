# Cost and latency per sweep

Measured automatically for every sweep and written into `claim_packet.json` under `processing` (per-stage seconds) and `run_stats` (frame counts, avg detection/OCR ms, LLM token counts, API call counts, estimated cost). **Numbers below are to be copied from the real demo sweep's packet — none are reported until it exists.**

| Stage | Seconds | Status |
|---|---|---|
| _fill from `processing[]`_ | | |

Cost model (`app/metrics.py`): `groq_tokens/1e6 × LCA_COST_GROQ_PER_MTOK + serpapi_calls × LCA_COST_SERPAPI_PER_SEARCH`. The defaults in the code are **placeholders — set the current provider rates in `.env`**. eBay Browse, Google Books, Open Library and Frankfurter calls are counted but have no per-call charge on their free tiers; local models (YOLO, Paddle, Whisper, Qwen) cost electricity only.
