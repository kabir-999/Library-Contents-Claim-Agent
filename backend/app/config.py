"""Environment-driven configuration. Nothing here is a secret default."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(ROOT / ".env"), env_file_encoding="utf-8", extra="ignore"
    )

    lca_data_dir: str = str(ROOT / "data")
    lca_cors_origins: str = "http://localhost:3000,http://127.0.0.1:3000"
    lca_log_level: str = "INFO"

    lca_local_llm_provider: str = "ollama"
    ollama_base_url: str = "http://localhost:11434"
    lca_local_llm_model: str = "qwen2.5:7b-instruct"
    lca_local_vlm_model: str = "qwen2.5vl:7b"
    groq_api_key: str = ""
    lca_groq_model: str = "openai/gpt-oss-120b"

    lca_yolo_object_weights: str = "yolov8s.pt"
    lca_yolo_spine_weights: str = ""
    lca_ocr_enabled: bool = True
    lca_book_conf: float = 0.05   # low on purpose: recall first; tracker hits + OCR/metadata verification filter false positives

    lca_whisper_model: str = "base.en"
    lca_tts_engine: str = "browser"

    google_books_api_key: str = ""
    ebay_client_id: str = ""
    ebay_client_secret: str = ""
    serpapi_api_key: str = ""
    serper_api_key: str = ""
    lca_fx_base_url: str = "https://api.frankfurter.app"

    lca_id_confidence_min: float = 0.80
    lca_appraisal_price_threshold: float = 250.0
    lca_min_listings: int = 2   # a price needs >=2 agreeing listings; a lone listing is too weak to put in a claim
    lca_http_timeout_s: float = 10.0
    lca_http_retries: int = 2

    @property
    def data_path(self) -> Path:
        p = Path(self.lca_data_dir)
        if not p.is_absolute():
            p = ROOT / p
        p.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def cors_list(self) -> list[str]:
        return [o.strip() for o in self.lca_cors_origins.split(",") if o.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()
