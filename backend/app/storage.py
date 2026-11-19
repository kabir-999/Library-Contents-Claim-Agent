"""SQLite for structured data, filesystem for frames/evidence.

Layout: data/lca.sqlite3, data/sweeps/<sweep_id>/{frames,crops,exports}/
Entities are stored as JSON documents keyed (sweep_id, kind, id) - the schema
is validated by Pydantic at the edges, not duplicated in SQL.
"""
from __future__ import annotations

import json
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any, Iterable, Optional

from .config import get_settings

import re
SAFE_ID = re.compile(r"^[A-Za-z0-9_\-]{1,80}$")

SCHEMA = """
CREATE TABLE IF NOT EXISTS sweeps(id TEXT PRIMARY KEY, created_at REAL, state TEXT, country TEXT, currency TEXT, device TEXT, meta TEXT);
CREATE TABLE IF NOT EXISTS frames(id TEXT PRIMARY KEY, sweep_id TEXT, ts REAL, rel_ts REAL, path TEXT, w INT, h INT, quality TEXT, orient TEXT, keep INT DEFAULT 0);
CREATE INDEX IF NOT EXISTS ix_frames_sweep ON frames(sweep_id, ts);
CREATE TABLE IF NOT EXISTS entities(sweep_id TEXT, kind TEXT, id TEXT, body TEXT, PRIMARY KEY(sweep_id, kind, id));
CREATE TABLE IF NOT EXISTS events(n INTEGER PRIMARY KEY AUTOINCREMENT, sweep_id TEXT, ts REAL, kind TEXT, body TEXT);
CREATE INDEX IF NOT EXISTS ix_events_sweep ON events(sweep_id, n);
"""


class Storage:
    def __init__(self, base: Optional[Path] = None):
        self.base = Path(base) if base else get_settings().data_path
        self.base.mkdir(parents=True, exist_ok=True)
        self.db_path = self.base / "lca.sqlite3"
        self._lock = threading.RLock()
        self._con = sqlite3.connect(self.db_path, check_same_thread=False)
        self._con.row_factory = sqlite3.Row
        self._con.execute("PRAGMA journal_mode=WAL")
        self._con.executescript(SCHEMA)

    # ---------- paths ----------
    def sweep_dir(self, sweep_id: str, *sub: str) -> Path:
        if not SAFE_ID.match(sweep_id or ""):
            raise ValueError("invalid sweep id")
        p = self.base / "sweeps" / sweep_id
        for s in sub:
            p = p / s
        p.mkdir(parents=True, exist_ok=True)
        return p

    # ---------- sweeps ----------
    def create_sweep(self, sweep_id: str, country: str, currency: str, device: str, meta: dict) -> None:
        with self._lock:
            self._con.execute("INSERT INTO sweeps VALUES(?,?,?,?,?,?,?)",
                              (sweep_id, time.time(), "recording", country, currency, device, json.dumps(meta)))
            self._con.commit()
        self.sweep_dir(sweep_id, "frames"); self.sweep_dir(sweep_id, "crops"); self.sweep_dir(sweep_id, "exports")

    def get_sweep(self, sweep_id: str) -> Optional[dict]:
        with self._lock:
            r = self._con.execute("SELECT * FROM sweeps WHERE id=?", (sweep_id,)).fetchone()
        if not r:
            return None
        d = dict(r); d["meta"] = json.loads(d["meta"] or "{}"); return d

    def list_sweeps(self) -> list[dict]:
        with self._lock:
            rows = self._con.execute("SELECT id, created_at, state, country, currency FROM sweeps ORDER BY created_at DESC").fetchall()
        return [dict(r) for r in rows]

    def update_sweep(self, sweep_id: str, *, state: Optional[str] = None, meta: Optional[dict] = None) -> None:
        with self._lock:
            if state:
                self._con.execute("UPDATE sweeps SET state=? WHERE id=?", (state, sweep_id))
            if meta is not None:
                cur = self.get_sweep(sweep_id)["meta"]; cur.update(meta)
                self._con.execute("UPDATE sweeps SET meta=? WHERE id=?", (json.dumps(cur), sweep_id))
            self._con.commit()

    # ---------- frames ----------
    def save_frame(self, sweep_id: str, frame_id: str, jpeg: bytes, ts: float, rel_ts: float, w: int, h: int,
                   quality: dict, orient: Optional[dict]) -> Path:
        path = self.sweep_dir(sweep_id, "frames") / f"{frame_id}.jpg"
        path.write_bytes(jpeg)
        with self._lock:
            self._con.execute("INSERT OR REPLACE INTO frames VALUES(?,?,?,?,?,?,?,?,?,0)",
                              (frame_id, sweep_id, ts, rel_ts, str(path), w, h, json.dumps(quality), json.dumps(orient or {})))
            self._con.commit()
        return path

    def mark_keep(self, sweep_id: str, frame_ids: Iterable[str]) -> None:
        with self._lock:
            self._con.executemany("UPDATE frames SET keep=1 WHERE id=? AND sweep_id=?", [(f, sweep_id) for f in set(frame_ids) if f])
            self._con.commit()

    def get_frame(self, frame_id: str) -> Optional[dict]:
        with self._lock:
            r = self._con.execute("SELECT * FROM frames WHERE id=?", (frame_id,)).fetchone()
        return dict(r) if r else None

    def frame_ids(self, sweep_id: str) -> set[str]:
        with self._lock:
            return {r[0] for r in self._con.execute("SELECT id FROM frames WHERE sweep_id=?", (sweep_id,)).fetchall()}

    def frames_meta(self, sweep_id: str) -> list[dict]:
        with self._lock:
            rows = self._con.execute("SELECT id, ts, rel_ts, w, h, quality, keep FROM frames WHERE sweep_id=? ORDER BY ts", (sweep_id,)).fetchall()
        return [dict(r) for r in rows]

    def prune_frames(self, sweep_id: str, keep_every_s: float = 5.0) -> int:
        """Delete non-evidence frames, retaining a sparse timeline (one per `keep_every_s`)."""
        with self._lock:
            rows = self._con.execute("SELECT id, rel_ts, path, keep FROM frames WHERE sweep_id=? ORDER BY ts", (sweep_id,)).fetchall()
            last, removed = -1e9, 0
            for r in rows:
                if r["keep"]:
                    continue
                if r["rel_ts"] - last >= keep_every_s:
                    last = r["rel_ts"]; self._con.execute("UPDATE frames SET keep=1 WHERE id=?", (r["id"],)); continue
                Path(r["path"]).unlink(missing_ok=True)
                self._con.execute("DELETE FROM frames WHERE id=?", (r["id"],)); removed += 1
            self._con.commit()
        return removed

    # ---------- entities ----------
    def put(self, sweep_id: str, kind: str, id_: str, body: Any) -> None:
        with self._lock:
            self._con.execute("INSERT OR REPLACE INTO entities VALUES(?,?,?,?)", (sweep_id, kind, id_, json.dumps(body, default=str)))
            self._con.commit()

    def put_many(self, sweep_id: str, kind: str, rows: dict[str, Any]) -> None:
        with self._lock:
            self._con.executemany("INSERT OR REPLACE INTO entities VALUES(?,?,?,?)",
                                  [(sweep_id, kind, k, json.dumps(v, default=str)) for k, v in rows.items()])
            self._con.commit()

    def get(self, sweep_id: str, kind: str, id_: str) -> Optional[Any]:
        with self._lock:
            r = self._con.execute("SELECT body FROM entities WHERE sweep_id=? AND kind=? AND id=?", (sweep_id, kind, id_)).fetchone()
        return json.loads(r[0]) if r else None

    def list(self, sweep_id: str, kind: str) -> list[Any]:
        with self._lock:
            rows = self._con.execute("SELECT body FROM entities WHERE sweep_id=? AND kind=? ORDER BY id", (sweep_id, kind)).fetchall()
        return [json.loads(r[0]) for r in rows]

    def delete_kind(self, sweep_id: str, kind: str) -> None:
        with self._lock:
            self._con.execute("DELETE FROM entities WHERE sweep_id=? AND kind=?", (sweep_id, kind)); self._con.commit()

    # ---------- events ----------
    def add_event(self, sweep_id: str, kind: str, body: dict) -> int:
        with self._lock:
            c = self._con.execute("INSERT INTO events(sweep_id, ts, kind, body) VALUES(?,?,?,?)", (sweep_id, time.time(), kind, json.dumps(body, default=str)))
            self._con.commit(); return c.lastrowid

    def events(self, sweep_id: str, after: int = 0, limit: int = 500) -> list[dict]:
        with self._lock:
            rows = self._con.execute("SELECT n, ts, kind, body FROM events WHERE sweep_id=? AND n>? ORDER BY n LIMIT ?", (sweep_id, after, limit)).fetchall()
        return [{"n": r["n"], "ts": r["ts"], "kind": r["kind"], **json.loads(r["body"])} for r in rows]


_storage: Optional[Storage] = None


def storage() -> Storage:
    global _storage
    if _storage is None:
        _storage = Storage()
    return _storage
