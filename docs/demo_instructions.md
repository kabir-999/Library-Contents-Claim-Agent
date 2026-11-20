# Recording the demo (single take, ≤6 minutes, unedited)

**Before recording (not part of the take)**
* Backend + frontend running (README). `GET /api/health` shows Groq/eBay/SerpAPI/OCR ✓ as available. `ollama list` shows the Qwen models.
* Use a phone over HTTPS for room measurement (README), headphones on, good light, no window glare.
* Tape an **A4 sheet** flat on a shelf edge in view (gives the scale) — or know a shelf's height to say aloud.
* Know the phone's height above the floor (hold it at chest height; measure once).
* Screen-record the laptop browser showing the live UI; keep the phone feed visible if you can.

**Take (≈5:30)**
| t | Do | Say |
|---|---|---|
| 0:00 | Open the app, choose country + currency, enter camera height, press Start | — |
| 0:20 | Sweep shelf unit 1 left→right, top→bottom, slowly | — |
| 0:50 | Move too fast for a second | (agent says slow down) |
| 1:10 | Point at a book you own as a first edition | “That is a first edition.” |
| 1:30 | Point at a few books that aren't yours | “Those books belong to someone else.” |
| 1:50 | Sweep unit 2; point at a framed picture | agent asks original or print → “That is a print, not an original painting.” |
| 2:20 | Interrupt the agent mid-sentence once | (barge-in demo) |
| 2:30 | Point at floor corners, say “mark corner” ×4; ceiling above one: “ceiling corner” | |
| 2:50 | Say “skip this shelf” on a shelf of someone else's | |
| 3:00 | Say “done” | |
| 3:00–5:00 | Show: processing stages → Summary (totals, room, review queue) → Inventory (shelves, links open to listings) → Packet (JSON, PDF, evidence frame, **second-country** table: pick another country, press the button) | |

Stop the screen recording only when the packet screen has been shown. If something fails on camera, leave it; failures belong in `docs/failure_log.md`.
