from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from dotenv import load_dotenv

import re
from .utils import DEFAULT_PUBLIC_URLS

load_dotenv()

@dataclass
class AppConfig:
    llm_provider: str = os.getenv("LLM_PROVIDER", "openai").strip().lower() or "openai"
    embedding_provider: str = os.getenv("EMBEDDING_PROVIDER", "").strip().lower()
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "").strip()
    openai_model: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini").strip() or "gpt-4o-mini"
    openai_embed_model: str = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small").strip() or "text-embedding-3-small"
    gemini_api_key: str = os.getenv("GEMINI_API_KEY", "").strip() or os.getenv("GOOGLE_API_KEY", "").strip()
    gemini_model: str = os.getenv("GEMINI_MODEL", "gemini-1.5-flash").strip() or "gemini-1.5-flash"
    gemini_embed_model: str = os.getenv("GEMINI_EMBED_MODEL", "embedding-001").strip() or "embedding-001"
    serpapi_api_key: str = os.getenv("SERPAPI_API_KEY", "").strip() or os.getenv("SERP_API_KEY", "").strip()
    request_timeout: int = int(os.getenv("REQUEST_TIMEOUT", "20"))
    llm_timeout: int = int(os.getenv("LLM_TIMEOUT", "60"))
    embed_timeout: int = int(os.getenv("EMBED_TIMEOUT", "30"))
    langfuse_timeout: int = int(os.getenv("LANGFUSE_TIMEOUT", "30"))
    milvus_timeout: float = float(os.getenv("MILVUS_TIMEOUT", "20.0"))
    rag_dir: str = os.getenv("RAG_DIR", "./data/rag").strip() or "./data/rag"
    milvus_db_path: str = os.getenv("MILVUS_DB_PATH", "./vector_db/agent_rag.db").strip() or "./vector_db/agent_rag.db"
    base_collection_name: str = os.getenv("MILVUS_COLLECTION", "booking_policy_chunks").strip() or "booking_policy_chunks"
    rag_top_k: int = int(os.getenv("RAG_TOP_K", "4"))
    web_top_k: int = int(os.getenv("WEB_TOP_K", "4"))
    langfuse_public_key: str = os.getenv("LANGFUSE_PUBLIC_KEY")
    langfuse_secret_key: str = os.getenv("LANGFUSE_SECRET_KEY")
    langfuse_host: str = os.getenv("LANGFUSE_HOST")
    default_user_id: str = os.getenv("DEFAULT_USER_ID", "demo_user").strip() or "demo_user"
    default_public_urls: List[str] = field(default_factory=lambda: list(DEFAULT_PUBLIC_URLS))

    @classmethod
    def from_env(cls) -> "AppConfig":
        return cls()

    @property
    def rag_path(self) -> Path:
        path = Path(self.rag_dir).expanduser()
        if not path.is_absolute():
            path = Path.cwd() / path
        path.mkdir(parents=True, exist_ok=True)
        return path.resolve()

    @property
    def db_path(self) -> Path:
        path = Path(self.milvus_db_path).expanduser()
        if not path.is_absolute():
            path = Path.cwd() / path
        path.parent.mkdir(parents=True, exist_ok=True)
        return path.resolve()

    @property
    def manifest_path(self) -> Path:
        return self.db_path.with_suffix(".manifest.json")

    def get_generation_model(self, provider: Optional[str] = None) -> str:
        provider = (provider or self.llm_provider or "openai").lower()
        if provider == "openai":
            return self.openai_model
        if provider == "gemini":
            return self.gemini_model
        raise ValueError(f"Unsupported LLM provider: {provider}")

    def get_embedding_provider(self, provider: Optional[str] = None) -> str:
        return (provider or self.embedding_provider or self.llm_provider or "openai").lower()

    def get_embedding_model(self, provider: Optional[str] = None) -> str:
        provider = self.get_embedding_provider(provider)
        if provider == "openai":
            return self.openai_embed_model
        if provider == "gemini":
            return self.gemini_embed_model
        raise ValueError(f"Unsupported embedding provider: {provider}")

    def get_collection_name(self, provider: Optional[str] = None, model: Optional[str] = None) -> str:
        provider = self.get_embedding_provider(provider)
        model = model or self.get_embedding_model(provider)
        raw = f"{self.base_collection_name}_{provider}_{model}"
        cleaned = re.sub(r"[^0-9a-zA-Z_]+", "_", raw)
        return cleaned[:180]

