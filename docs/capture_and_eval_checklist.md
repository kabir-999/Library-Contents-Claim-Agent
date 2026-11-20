# What only you can do (and exactly how)

The code is ready for all of this; none of it can be done in software. Do the **ground truth first**, before changing any thresholds.

## 1. Pick the space (needs permission if it isn't yours)
>= 60 books across >= 2 shelving units, >= 8 non-book items, one enclosed room. No people or documents with personal data in view.

## 2. Ground truth BEFORE the sweep
`cd backend && python -m app.eval.ground_truth ../data/eval/ground_truth.xlsx` creates the sheet. Fill it by hand:
1. **True book count** (count every book, flat and stacked ones included).
2. **Titles**: write every spine you can read with your own eyes (title, author).
3. **20 spines measured with a ruler** (height and thickness, cm) - spread across shelves, include thin and thick.
4. **15 prices**: look each book up yourself (same source type the app uses: new listing in your country) and write the price + URL + date.
5. **Non-book items**: list every item (category).
6. **Room, tape-measured**: length, width, ceiling height (m); area covered by shelving. Note if it isn't a rectangle.

## 3. Prepare the capture
* Put a flat **A4 or Letter sheet** on a shelf edge, in view, for the first shelf (this is the metric scale).
* Phone: run `./scripts/phone.sh`, open the HTTPS page (see README), enter the **camera height** (measure it: floor to phone, cm).
* Good light, no glare; walk slowly, one continuous pass under 3 minutes.
* Point at each floor/wall corner and say **"mark corner"**; point at the ceiling above one and say **"ceiling corner"** (needs >= 3 floor corners).
* Say at least one correction out loud (e.g. "that's a first edition", "skip this shelf"). Answer the art question (original or print).

## 4. Record the demo (one take, <= 6 min, not edited)
Screen-record the phone or laptop with audio from the start screen to the claim packet: greeting, sweep with the agent talking, the inventory filling in, Stop, the spoken summary, the results, the PDF/JSON. Do not cut or speed it up.

## 5. Score it honestly
```bash
cd backend && source .venv/bin/activate
python -m app.eval.evaluate ../data/eval/ground_truth.xlsx ../data/sweeps/<sweep_id>/exports/claim_packet.json ../data/eval/results
```
Copy **all** numbers (including the misses) into `docs/evaluation.md`; put the three worst errors, their root causes and the measured fixes into `docs/failure_log.md` section B, and the per-stage timings from the results page's "Processing" list into `docs/cost_latency.md`.

## 6. Submit
Private GitHub repo shared with the reviewers; `.env` is gitignored (check with `git status`); attach the video, `claim_packet.json` + PDF + the saved frames in `data/sweeps/<sweep_id>/frames`, the ground-truth sheet with results, the architecture note and the failure log.
