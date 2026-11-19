"""FastAPI app: REST for claims/evidence/exports + one WebSocket per sweep."""
from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field

from .agent import live
from .agent.session import SweepSession
from .claim import calculations as calc
from .claim.pdf import write_pdf
from .config import get_settings
from .llm.providers import router
from .pipeline import postprocess
from .pipeline.second_country import compare
from .pricing import book_pricing, providers
from .pricing.markets import MARKETS, market
from .schemas import ClaimPacket
from .storage import storage
from .vision import models, ocr
from .voice import stt, tts

s = get_settings()
logging.basicConfig(level=s.lca_log_level, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("lca")
app = FastAPI(title="Library Contents Claim Agent")
@app.exception_handler(ValueError)
async def _value_error(_, exc: ValueError):
    return JSONResponse({"detail": str(exc)}, status_code=404 if "sweep id" in str(exc) else 422)


app.add_middleware(CORSMiddleware, allow_origins=s.cors_list, allow_methods=["*"], allow_headers=["*"])


class StartClaim(BaseModel):
    country: str = Field(min_length=2, max_length=40)
    currency: str = Field(default="", max_length=3)
    device: str = Field(default="", max_length=200)
    camera_height_cm: Optional[float] = Field(default=None, gt=30, lt=250)


def _packet(sweep_id: str) -> ClaimPacket:
    d = storage().get(sweep_id, "packet", "claim")
    if not d:
        raise HTTPException(404, "packet not generated yet")
    return ClaimPacket.model_validate(d)


@app.get("/api/health")
async def health():
    return {"ok": True, "models": models.status(), "llm": {"groq_configured": router().groq_ok, "local_provider": s.lca_local_llm_provider,
            "local_model": s.lca_local_llm_model}, "pricing_sources": book_pricing.available_sources(), "tts": tts.engine(),
            "ocr_enabled": s.lca_ocr_enabled, "markets": {k: v["name"] for k, v in MARKETS.items()}}


@app.get("/api/markets")
async def markets():
    return [{"code": k, **v} for k, v in MARKETS.items()]


@app.post("/api/sweeps")
async def start_claim(b: StartClaim):
    mk = market(b.country)
    currency = (b.currency or mk["currency"]).upper()
    if not currency:
        raise HTTPException(422, "currency required for unknown country")
    sid = "sw_" + time.strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:6]
    storage().create_sweep(sid, mk["code"], currency, b.device, {"camera_height_cm": b.camera_height_cm})
    return {"sweep_id": sid, "country": mk["code"], "currency": currency}


@app.get("/api/sweeps")
async def list_sweeps():
    return storage().list_sweeps()


@app.get("/api/sweeps/{sid}")
async def get_sweep(sid: str):
    r = storage().get_sweep(sid)
    if not r:
        raise HTTPException(404)
    return r


@app.get("/api/sweeps/{sid}/events")
async def events(sid: str, after: int = 0):
    return storage().events(sid, after)


@app.get("/api/sweeps/{sid}/packet")
async def packet_json(sid: str):
    return JSONResponse(json.loads(_packet(sid).model_dump_json()))


@app.get("/api/sweeps/{sid}/packet/download")
async def packet_download(sid: str):
    f = storage().sweep_dir(sid, "exports") / "claim_packet.json"
    if not f.exists():
        raise HTTPException(404)
    return FileResponse(f, media_type="application/json", filename="claim_packet.json")


@app.get("/api/sweeps/{sid}/report.pdf")
async def report_pdf(sid: str):
    f = storage().sweep_dir(sid, "exports") / "claim_report.pdf"
    if not f.exists():
        raise HTTPException(404)
    return FileResponse(f, media_type="application/pdf", filename="claim_report.pdf")


@app.get("/api/frames/{frame_id}.jpg")
async def frame(frame_id: str):
    r = storage().get_frame(frame_id)
    if not r or not Path(r["path"]).exists():
        raise HTTPException(404, "frame not found")
    return FileResponse(r["path"], media_type="image/jpeg")


@app.get("/api/sweeps/{sid}/crops/{ent}.jpg")
async def crop(sid: str, ent: str):
    f = storage().sweep_dir(sid, "crops") / f"{Path(ent).name}.jpg"
    if not f.exists():
        raise HTTPException(404)
    return FileResponse(f, media_type="image/jpeg")


class SecondCountry(BaseModel):
    country: str
    n: int = Field(default=10, ge=1, le=25)


@app.post("/api/sweeps/{sid}/second-country")
async def second_country(sid: str, b: SecondCountry):
    p = _packet(sid)
    if market(b.country)["code"] == p.sweep.country:
        raise HTTPException(422, "second country must differ from the claim country")
    p.second_country_pricing = await compare(p, b.country, b.n)
    st = storage(); st.put(sid, "packet", "claim", json.loads(p.model_dump_json()))
    ex = st.sweep_dir(sid, "exports")
    (ex / "claim_packet.json").write_text(p.model_dump_json(indent=2))
    (ex / "second_country_pricing.json").write_text(p.second_country_pricing.model_dump_json(indent=2))
    write_pdf(p, ex / "claim_report.pdf", st.sweep_dir(sid, "crops"), st.sweep_dir(sid, "frames"))
    return json.loads(p.second_country_pricing.model_dump_json())


class Resolve(BaseModel):
    resolved: bool = True
    note: str = ""


@app.post("/api/sweeps/{sid}/review/{ref}/resolve")
async def resolve(sid: str, ref: str, b: Resolve):
    p = _packet(sid)
    for q in p.review_queue:
        if q.ref_id == ref:
            q.resolved = b.resolved
    storage().put(sid, "packet", "claim", json.loads(p.model_dump_json()))
    (storage().sweep_dir(sid, "exports") / "claim_packet.json").write_text(p.model_dump_json(indent=2))
    return {"ok": True}


class TwoPoint(BaseModel):
    frame_id: str
    p1: tuple[float, float]
    p2: tuple[float, float]
    real_cm: float = Field(gt=0, lt=1000)


@app.post("/api/sweeps/{sid}/calibration/two-point")
async def two_point(sid: str, b: TwoPoint):
    """Manual calibration on a saved frame (tape-measured distance). Re-runs measurement + packet; no other stage re-run."""
    sess = live.SESSIONS.get(sid)
    if not sess:
        raise HTTPException(409, "session no longer in memory; calibrate during or right after the sweep")
    try:
        cal = sess.calibrate_two_point(b.frame_id, b.p1, b.p2, b.real_cm)
    except ValueError as e:
        raise HTTPException(422, str(e))
    packet = await postprocess.run(sess)
    return {"calibration": cal, "totals": json.loads(packet.totals.model_dump_json())}


@app.post("/api/sweeps/{sid}/finish")
async def finish(sid: str):
    """Finish a sweep whose WebSocket dropped before 'stop' (e.g. phone locked). Idempotent."""
    sess = live.SESSIONS.get(sid)
    if not sess:
        raise HTTPException(409, "session not in memory")
    if sid in live.PACKET_TASKS:
        return {"ok": True, "already": True}
    sess.ended_at = sess.ended_at or time.time()
    sess.store.update_sweep(sid, state="processing")
    live.PACKET_TASKS[sid] = asyncio.create_task(postprocess.run(sess))
    return {"ok": True}


@app.post("/api/sweeps/{sid}/reprocess")
async def reprocess(sid: str):
    sess = live.SESSIONS.get(sid)
    if not sess:
        raise HTTPException(409, "session not in memory")
    p = await postprocess.run(sess)
    return {"ok": True, "seconds": p.model_extra["run_stats"]["post_sweep_processing_s"]}


@app.websocket("/ws/sweeps/{sid}")
async def ws(websocket: WebSocket, sid: str):
    row = storage().get_sweep(sid)
    await websocket.accept()
    if not row:
        await websocket.close(code=4404); return
    sess = live.SESSIONS.get(sid)
    if sess is None:
        sess = SweepSession(sid, row["country"], row["currency"], row["device"], row["meta"].get("camera_height_cm"))
        live.SESSIONS[sid] = sess
    agent = live.LiveAgent(websocket, sess)
    await agent.run()
    task = live.PACKET_TASKS.get(sid)
    if task and not task.done() and agent.closed is False:
        await asyncio.shield(task)
