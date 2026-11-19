# Evaluation protocol (real room, real ground truth)

**No accuracy numbers are reported in this repo until this protocol has been run on a real room.** The harness is implemented and unit-tested on a synthetic fixture (which proves only the scoring logic).

## Test room requirements
≥60 books · ≥2 shelving units · ≥8 non-book objects · one enclosed room.

## Steps
1. `cd backend && source .venv/bin/activate && python -m app.eval.ground_truth ../data/eval/ground_truth.xlsx`
2. **Before** running the system, fill the workbook by hand:
   * `Books` – every physical book (two people count), `legible=Y` where a human can read the spine; transcribe the printed title.
   * `Spines_measured` – 20 spines with a ruler (height, thickness).
   * `Prices_checked` – 15 books priced by hand today (URL, date, currency, condition); mark `replacement` or `used`.
   * `NonBook` – the complete object list with categories.
   * `Room` – tape-measured length/width/height/floor area/wall area/shelved wall area, true book count.
3. Run one sweep (≤3 min) following `docs/demo_instructions.md`; the first-run packet is the one to evaluate — **do not discard difficult sweeps/frames.**
4. `python -m app.eval.evaluate ../data/eval/ground_truth.xlsx ../data/sweeps/<id>/exports/claim_packet.json ../data/eval/results`
   → `results.json`, `results.xlsx` (summary + per-row detail), `results.md`.

## Scoring rules (strict, so a pass can't be manufactured)
| Metric | Bar | How scored |
|---|---|---|
| Book count | ±5% | `totals.book_count` vs true count |
| Title identification | ≥70% | legible GT spines matched (fuzzy title ≥85, one-to-one) to a system book |
| Confidently wrong | ≤3% | system `identified`, confidence ≥0.80, title matches no GT title; ÷ legible spines |
| Spine dimensions | ±15% | all 20 books, height **and** thickness; blank/unmatched = fail |
| Book prices | ±25% | all 15 prices; blank = fail |
| Non-book detection | ≥80% | GT objects with a same-category system item (needs ≥8 GT objects) |
| Floor / wall area | ±10% / ±15% | blank = fail |
| Packet generation | <300 s | `packet_generation_total` |

"Within X% on N books" is ambiguous in the brief; the strict reading (every row) is used for PASS. Per-row errors and the median are in `results.xlsx` so a looser reading can be argued openly, but the label stays FAIL.

## Results
_Not yet run._ Fill `data/eval/results/` and copy the table into DEVELOPMENT_STATUS.md after the real run.
