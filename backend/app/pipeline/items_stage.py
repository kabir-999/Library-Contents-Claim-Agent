"""Non-book classification (YOLO class -> claim category, optional Qwen-VL description) + pricing."""
from __future__ import annotations

import asyncio
import io
import logging

import cv2
from rapidfuzz import fuzz

from ..llm.providers import router
from ..pricing import nonbook_pricing as nbp
from ..schemas import Dim3, Item, ItemCost
from ..vision import ocr
from ..vision.object_detector import CATEGORY, COCO_MAP
from .measure_stage import measure_item

log = logging.getLogger(__name__)
VLM_PROMPT = ('Describe the main object in this image for an insurance inventory. Return JSON: '
              '{"description": short noun phrase, "material": visible material or null, "brand_model": brand/model ONLY if its text is '
              'clearly legible in the image else null, "is_framed_artwork": bool}. Do not guess brands.')


async def classify_item(session, t) -> Item:
    cat = CATEGORY.get(t.label) or COCO_MAP.get(t.label) or "other"
    desc, material, brand, flags = t.label, "", "", []
    crop_jpg = None
    fr = session.store.get_frame(t.best_frame_id)
    if fr and t.best_bbox:
        img = cv2.imread(fr["path"])
        if img is not None:
            x1, y1, x2, y2 = [int(v) for v in t.best_bbox]
            crop = img[max(0, y1):y2, max(0, x1):x2]
            if crop.size:
                crop_jpg = cv2.imencode(".jpg", crop)[1].tobytes()
                session.store.sweep_dir(session.id, "crops")
                cv2.imwrite(str(session.store.sweep_dir(session.id, "crops") / f"{t.tid}.jpg"), crop)
    if crop_jpg:
        v = await router().vision(crop_jpg, VLM_PROMPT)
        if isinstance(v, dict):
            desc = (v.get("description") or desc)[:120]
            material = v.get("material") or ""
            b = (v.get("brand_model") or "").strip()
            if b:  # evidence-bound: the brand must be readable by OCR on the crop, otherwise it's a hallucination
                r = await asyncio.to_thread(ocr.read_spine, cv2.imdecode(__import__("numpy").frombuffer(crop_jpg, "uint8"), 1), [0, 0, 10 ** 6, 10 ** 6], "horizontal_stack")
                if r.get("text") and fuzz.partial_ratio(b.lower(), r["text"].lower()) >= 85:
                    brand = b
                else:
                    flags.append("brand_unverified_by_ocr")
            if v.get("is_framed_artwork") and cat not in ("artwork", "portrait"):
                cat = "artwork"
    if not brand and crop_jpg and cat not in ("artwork", "portrait", "bookshelf", "cabinet", "table", "chair", "sofa", "desk", "rug", "mirror"):
        try:   # read the product's own label: brand/model text is evidence, no LLM involved
            import re as _re
            from ..identify import fast as _fast
            cimg = cv2.imdecode(__import__("numpy").frombuffer(crop_jpg, "uint8"), 1)
            lines = await asyncio.to_thread(_fast.read_lines, cimg) if _fast.available() else []
            txt = " ".join(lines) or (await asyncio.to_thread(ocr.read_spine, cimg, [0, 0, 10 ** 6, 10 ** 6], "horizontal_stack")).get("text", "")
            words = _re.findall(r"[A-Za-z0-9'&.+-]{2,}", txt)
            if len(" ".join(words)) >= 4:
                brand = " ".join(words)[:60]; flags.append("brand_from_ocr"); desc = f"{desc}: {brand}"[:120]
        except Exception:
            pass
    user_note = t.meta.get("user_note", "")
    dims = measure_item(session, t, cat)
    item = Item(id=t.tid, category=cat, description=desc, brand_model=brand, frame_ref=t.best_frame_id,
                dimensions_cm=Dim3(w=dims["w"], h=dims["h"], d=None) if dims else Dim3(),
                dimension_method=dims["method"] if dims else "", dimension_confidence=dims["confidence"] if dims else None,
                confidence=round(min(1.0, t.conf), 3), material=material, bbox=t.best_bbox, user_note=user_note, flags=flags)
    if "occluded_or_clipped" in t.flags:
        item.flags.append("occluded")
    if t.meta.get("excluded"):
        item.status = "excluded"; item.flags.append("user_excluded")
    return item


async def price_items(session, items: list[Item], sem: asyncio.Semaphore) -> None:
    async def one(it: Item):
        try:
            await _one(it)
        except Exception:
            log.exception("pricing failed for %s; left unpriced", it.id)
            it.status = "unpriced"; it.flags.append("price_unavailable: error")

    async def _one(it: Item):
        if it.status == "excluded":
            return
        note = it.user_note
        if it.category in nbp.APPRAISAL_CATEGORIES and nbp.needs_appraisal(it.category, note):
            it.status = "needs_appraisal"; return
        async with sem:
            r = await nbp.price_item(it.category, it.brand_model, it.description, it.material, session.country, session.currency)
        it.replacement_cost = r["cost"]
        it.status = "priced" if r["priced"] else "unpriced"
        if not r["priced"]:
            it.flags.append(f"price_unavailable: {r['reason']}")
    await asyncio.gather(*(one(i) for i in items))
