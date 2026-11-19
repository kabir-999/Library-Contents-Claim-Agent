"""Pydantic models.

`claim_packet.json` follows the contract in the brief exactly (no required field
removed or renamed). Unknown values are `None`, never 0: zero is a real value.
Extra fields are additive (evidence, method, confidence, review flags).
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, field_validator

BookStatus = Literal["identified", "partial", "unidentified", "needs_review", "excluded", "needs_appraisal"]
ItemStatus = Literal["priced", "range", "unpriced", "needs_appraisal", "needs_review", "excluded"]


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class Measurement(BaseModel):
    """A metric value is only ever emitted together with how it was obtained."""
    value: Optional[float] = None
    unit: str = "cm"
    method: Optional[str] = None
    confidence: Optional[float] = Field(default=None, ge=0, le=1)
    frame_ref: Optional[str] = None
    calibration_ref: Optional[str] = None

    @field_validator("value")
    @classmethod
    def _nonneg(cls, v):
        if v is not None and v < 0:
            raise ValueError("measurement must be >= 0")
        return v


class Sweep(BaseModel):
    id: str = ""
    captured_at: str = ""
    device: str = ""
    duration_s: Optional[float] = None
    country: str = ""
    currency: str = ""


class Room(BaseModel):
    length_m: Optional[float] = None
    width_m: Optional[float] = None
    height_m: Optional[float] = None
    floor_area_m2: Optional[float] = None
    wall_area_m2: Optional[float] = None
    shelved_wall_area_m2: Optional[float] = None
    scale_method: str = ""
    confidence: Optional[float] = Field(default=None, ge=0, le=1)
    # additive
    floor_area_ft2: Optional[float] = None
    wall_area_ft2: Optional[float] = None
    shelved_wall_area_ft2: Optional[float] = None
    shape: str = ""
    assumptions: list[str] = []
    evidence_frames: list[str] = []
    status: str = "unavailable"  # measured | partial | unavailable


class ReplacementCost(BaseModel):
    amount: Optional[float] = None
    source: str = ""
    url: str = ""
    retrieved_at: str = ""
    converted: bool = False
    # additive
    currency: str = ""
    condition_assumed: str = ""
    original_amount: Optional[float] = None
    original_currency: str = ""
    fx_rate: Optional[float] = None
    fx_source: str = ""


class UsedValue(BaseModel):
    amount: Optional[float] = None
    source: str = ""
    url: str = ""
    retrieved_at: str = ""
    condition_assumed: str = ""
    # additive
    currency: str = ""
    converted: bool = False
    original_amount: Optional[float] = None
    original_currency: str = ""
    fx_rate: Optional[float] = None
    listings_used: int = 0


class Book(BaseModel):
    id: str
    shelf: str = ""
    position: int = 0
    frame_ref: str = ""
    status: BookStatus = "unidentified"
    title: str = ""
    author: str = ""
    edition: str = ""
    isbn: str = ""
    spine_height_cm: Optional[float] = None
    spine_thickness_cm: Optional[float] = None
    id_confidence: float = 0.0
    replacement_cost: ReplacementCost = ReplacementCost()
    used_value: UsedValue = UsedValue()
    # additive evidence
    publisher: str = ""
    orientation: str = "upright"  # upright | horizontal_stack | flat
    ocr_text: list[str] = []
    bbox: Optional[list[float]] = None  # x1,y1,x2,y2 in frame_ref pixels
    height_measurement: Measurement = Measurement()
    thickness_measurement: Measurement = Measurement()
    owner_note: str = ""
    flags: list[str] = []
    needs_appraisal: bool = False
    candidates: list[dict] = []
    detail_status: str = ""        # internal status before mapping to the contract's 3 values
    suggested_title: str = ""      # low-confidence proposal; NOT asserted as the title
    suggested_author: str = ""
    priced_isbn: str = ""      # edition actually priced (may differ from owned edition when edition unverified)
    pricing_note: str = ""
    verification: dict = {}
    track_id: str = ""
    _group_isbns: list = PrivateAttr(default_factory=list)


class Dim3(BaseModel):
    w: Optional[float] = None
    h: Optional[float] = None
    d: Optional[float] = None


class ItemCost(BaseModel):
    low: Optional[float] = None
    high: Optional[float] = None
    source: str = ""
    url: str = ""
    retrieved_at: str = ""
    currency: str = ""
    listings_used: int = 0
    sources: list[dict] = []   # every listing behind the range: {source,url,amount,currency}
    converted: bool = False


class Item(BaseModel):
    id: str
    category: str = ""
    description: str = ""
    brand_model: str = ""
    frame_ref: str = ""
    dimensions_cm: Dim3 = Dim3()
    status: ItemStatus = "unpriced"
    replacement_cost: ItemCost = ItemCost()
    confidence: float = 0.0
    # additive
    material: str = ""
    detail_status: str = ""
    dimension_method: str = ""
    dimension_confidence: Optional[float] = None
    bbox: Optional[list[float]] = None
    user_note: str = ""
    flags: list[str] = []


class Totals(BaseModel):
    book_count: int = 0
    books_identified: int = 0
    books_unidentified: int = 0
    shelf_run_m: Optional[float] = None
    books_replacement_cost: float = 0.0
    books_used_value: float = 0.0
    items_replacement_cost_low: float = 0.0
    items_replacement_cost_high: float = 0.0
    excluded_from_totals: int = 0
    # additive
    currency: str = ""
    books_priced_replacement: int = 0
    books_priced_used: int = 0
    items_priced: int = 0
    per_shelf: dict[str, dict] = {}


class ReviewEntry(BaseModel):
    ref_id: str
    reason: str
    category: str = ""
    severity: Literal["info", "warn", "blocker"] = "warn"
    frame_ref: str = ""
    resolved: bool = False


class SecondCountryRow(BaseModel):
    book_id: str
    title: str = ""
    isbn: str = ""
    replacement_cost: Optional[float] = None
    used_value: Optional[float] = None
    currency: str = ""
    source: str = ""
    url: str = ""
    retrieved_at: str = ""
    note: str = ""


class SecondCountryComparison(BaseModel):
    country: str = ""
    currency: str = ""
    rows: list[SecondCountryRow] = []


class ProcessingTiming(BaseModel):
    stage: str
    seconds: float
    status: str = "ok"


class ClaimPacket(BaseModel):
    model_config = ConfigDict(extra="allow")
    sweep: Sweep = Sweep()
    room: Room = Room()
    books: list[Book] = []
    items: list[Item] = []
    totals: Totals = Totals()
    review_queue: list[ReviewEntry] = []
    # additive
    second_country_pricing: Optional[SecondCountryComparison] = None
    excluded_books: list[Book] = []
    excluded_items: list[Item] = []
    shelves: list[dict] = []
    calibrations: list[dict] = []
    processing: list[ProcessingTiming] = []
    evidence_frames: list[dict] = []
    schema_version: str = "1.0"
    generated_at: str = ""
    disclaimer: str = (
        "Prepared by an automated agent. Totals are computed in Python from sourced prices only; "
        "excluded and review-queue entries require human confirmation."
    )
