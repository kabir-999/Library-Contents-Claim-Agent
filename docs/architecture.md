# Architecture note (one page)

```
 Browser (Next.js)                                  FastAPI (Python 3.12)
 ┌──────────────────────────┐   WebSocket            ┌──────────────────────────────────────────────────────┐
 │ camera 3fps JPEG ────────┼──── frame + orientation ─►  SweepSession (hot path, per frame)                │
 │ DeviceOrientation ───────┼─────────────────────────►   quality (blur/glare/light) → save frame (SQLite+FS)│
 │ mic → client VAD ────────┼──── utterance PCM16 ─────►   ORB+RANSAC homography → mosaic coords + segments  │
 │ speechSynthesis / WAV ◄──┼──── agent_say / guidance ─   YOLO book + YOLO-World objects (thread)          │
 │ barge-in: cancel TTS ────┼──── barge_in ────────────►   Tracker (IoU + spine centre, ≥2 hits) → shelves  │
 │ live counts + feed ◄─────┼──── state / events ──────    A4 auto-calibration · guidance rules            │
 └──────────────────────────┘                          │   OCR queue ─► PaddleOCR ─► Google Books/OpenLib  │
                                                       │        └► matcher (OCR-evidence-bound) ◄ Groq/Qwen │
   Voice: Whisper STT ─► corrections (regex ► Qwen, closed action set) ─► inventory update ─► templated reply
                                                       └───────────────────┬──────────────────────────────┘
                                          Stop ──────────────────────────► │ post-sweep, async, 270 s deadline
   ┌─────────────────────────────────────────────────────────────────────┐ ▼
   │ books: finish OCR/identify → duplicate merge → measure (calibration │──┐
   │        + homography) → price (eBay/SerpAPI, FX only if converting)  │  │   Python-only totals
   │ items: classify (YOLO-World + Qwen-VL, OCR-verified brand) → price  │──┼─► review queue ─► claim_packet.json
   │ room : floor-plane ranging (height, pitch, heading) → polygon area  │──┘                 └► claim_report.pdf
   └─────────────────────────────────────────────────────────────────────┘     evidence: frames/ crops/ (every frame_ref resolves)
```

**Why this shape.** Perception that must be *countable and measurable* (books, scale, room) is CV with geometry and evidence; LLMs are confined to roles where output is re-validated: splitting noisy OCR into title/author (accepted only if the words appear in the OCR), choosing among metadata candidates (accepted only if OCR supports the pick; never raises confidence), mapping speech to a closed set of actions. Prices/measurements/totals have no LLM path.

**Key invariants** — (1) unknown is `null`, never 0; (2) a monetary value needs a URL, retrieval date and matching currency or it is excluded from totals; (3) no calibration ⇒ no centimetres; (4) edition/ISBN only from evidence; (5) every `frame_ref` is a stored frame; (6) each stage fails independently and the packet still ships.

**Scale transfer.** Metric scale depends on camera distance, so each calibration is bound to its frame and carried to other frames by homography (direct ORB match, else tracked chain ×0.7 confidence). Calibrations are per wall segment; a segment with no calibration stays blank.

**Known weak points.** Dense-shelf spine splitting from COCO boxes; planar-shelf assumption for homographies; compass drift in room heading; used-value is asking-price based; half-duplex (not full-duplex) voice. See `failure_log.md`.

## Per-object pipeline order (live)

detect (YOLO + open-vocab) → **book gate** (CLIP zero-shot book vs whiteboard/wall/person…, borderline resolved by readable
text) → track/dedupe → fast label (Tesseract + one price search, shown as `~` provisional) → authoritative read (PaddleOCR,
vision scene reader) → catalogue verification → pricing. Anything the gate rejects is kept as `not_book` (so the tracker
doesn't recreate it) and is invisible to every later stage, the counts, and the packet. CLIP loads in a background thread
at backend start (first run downloads ~340 MB); until ready, a stricter fallback applies (confident detection or readable text).

## Record now, analyse after Stop (default: `LCA_LIVE_PROCESSING=false`)

During the sweep the server only stores frames (plus cheap blur/dark/glare tips). Corrections spoken during the sweep are timestamped.
On Stop, `pipeline/batch.py` analyses everything in bulk: quality + near-duplicate filter → keyframes (capped by `LCA_BATCH_MAX_FRAMES`,
default 150) → YOLO detection over lists of frames → text-line discovery on frames with no confident detection → in-order tracking,
book/not-book gate and corrections. Each physical object becomes one track with one unique id; OCR, catalogue and pricing then run
per id rather than per frame, and same-work titles are merged (`work_key`). `LCA_LIVE_PROCESSING=true` restores the old live mode.
Measured on an M-series MacBook Air: ~2-minute sweep (328 frames) → 135 keyframes analysed in ~21 s (detect 10 s, text discovery 4.5 s,
tracking 3.3 s, load/filter 3.3 s). Detection is compute-bound (~22 ms/frame per model), so 100-250 fps is not reachable on this hardware.
