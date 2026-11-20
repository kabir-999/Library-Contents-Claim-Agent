# Failure log

## A. Development-time defects found and fixed (not real-room results)
1. **Everything became "needs review" — group-identity bug** (`identify/matcher.py::decide`). `group_works()` was called twice, so the top candidate's own group was compared by identity against a *different* list and treated as a competing work → `ambiguous=True` always, so no book could ever reach `identified`. Caught by `test_clear_match_identified`. Fix: compute groups once and compare against that list. Lesson: confidence logic needs tests for the *positive* path, not only the guardrails.
2. **Calibration was not transferable between frames (design flaw, caught before shipping).** First design stored one px/cm per calibration and applied it to every book. Scale depends on camera distance, so a sheet calibrated at 40 cm would mis-scale a shelf shot from 90 cm by 2×. Fix: replaced phase-correlation tracking with ORB+RANSAC homographies; calibrations are bound to a frame and carried to other frames through `H(book frame→calibration frame)` (direct match, else tracked chain with confidence ×0.7). Test: `test_measure_transfers_scale_across_frames_via_homography`.
3. **Crash while writing evidence crops** (`pipeline/postprocess.py::save_crops`): `cache.get(path) or cv2.imread(path)` evaluates a NumPy array's truth value → `ValueError`. Caught by the stubbed end-to-end test. Fix: explicit `is None` checks.

4. **Zero books on a real (non-shelf) capture** — *measured fix.* First live test (one flat book on a desk, 108 frames) produced 0 confirmed books and an all-zero claim. Cause: COCO `book` confidence was below the 0.12 threshold on most frames, so detections were isolated and no track ever reached the 2-hit confirmation. Replaying the 24 saved frames offline: **10 raw detections/frame on some frames, 0 on others at 0.12 → 10 confirmed tracks (hits 10,7,4,4,4,3,…) at 0.05**. Fix: `LCA_BOOK_CONF=0.05` (recall first; OCR + metadata verification filter false boxes). Open risk: more false boxes; not yet measured on a real shelf.
5. **PaddleOCR never loaded (silent total failure).** PaddleOCR 3.x raises `ValueError: Unknown argument: show_log`; I only caught `TypeError`, so OCR was reported unavailable and every book would have stayed `unidentified`. Found only by running OCR on an image, not by the unit tests (which stub OCR). Fix: try 3.x then 2.x constructors, catching both. Verified on a rotated synthetic spine; real-spine accuracy still unmeasured.

### A2. Defects found while testing on a small real capture (2 books, one author, a desk/office; 2026-11-19/20)
| Defect | Root cause | Fix (measured) |
|---|---|---|
| One book listed as 2-3 entries ("THE ABC MURDERS" + "The A.B.C. Murders"; "…None" + "…None (adaptation)") | tracker boxes fragmented; catalogue title strings differ | `work_key` normalisation + fragment merge; 4 sweep entries collapse to 2 (unit-tested) |
| Tilted spine cut into fake columns | spine splitter ran on any wide box | only split runs with >=3 narrow pieces; aspect gate |
| Dark tilted spine never detected; later, fast mode silently dropped the only detector that found it | COCO `book` class misses it; I removed the open-vocab pass for speed | open-vocab + PaddleOCR text-line discovery; found by replaying real frames |
| "25 books" for 2 (whiteboard, ceiling, shelf objects, a chair counted) | detector recall set high with no verification | CLIP book/not-book gate; false positives scored 0.03-0.30, real books 0.54-0.98 |
| Stop -> packet 48 s | pricing 27 s (sequential lookups) + unbounded rescue passes | parallel lookups + time caps; 4 books priced in 4.0 s |
| India priced via USD->INR or garbage listings | no local source fallback | India never converted; local Google Shopping only; blank if none |

**Not yet a real-room evaluation.** Accuracy on a >=60-book, 2-unit room is **unmeasured**.

## B. Real-room failures (to be filled after the evaluation run)
Record the three worst errors from `data/eval/results`: what the system output, the ground truth, root cause (trace the frame, OCR text, candidates), the fix, and the re-run delta. Candidates I expect to be worst, to check first — **hypotheses, not findings**:
* merged spine boxes on dense shelves (COCO `book` boxes → classical splitting under-/over-splits);
* ORB registration failing on repetitive spine texture → new segments → uncalibrated books;
* OCR of decorative/vertical fonts → partial text → `needs_review`.
