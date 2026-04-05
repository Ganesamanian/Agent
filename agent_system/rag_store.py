from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from pymilvus import MilvusClient

from .config import AppConfig
from .model_provider import ModelProvider
from .models import Evidence
import re
from .utils import sanitize_text, shorten, hash_text, split_sentences


class RagStore:
    def __init__(self, config: AppConfig, provider: ModelProvider):
        self.config = config
        self.provider = provider

    def _client(self) -> Any:
        return MilvusClient(uri=str(self.config.db_path), timeout=self.config.milvus_timeout)

    def load_documents(self) -> List[Tuple[str, str]]:
        docs: List[Tuple[str, str]] = []
        for path in sorted(self.config.rag_path.rglob("*")):
            if path.is_file() and path.suffix.lower() in {".md", ".txt"}:
                docs.append((str(path), path.read_text(encoding="utf-8")))
        if not docs:
            raise FileNotFoundError(f"No .md or .txt files found in {self.config.rag_path}")
        return docs

    def chunk_documents(self, docs: List[Tuple[str, str]], max_chars: int = 700, overlap_sentences: int = 1) -> List[Dict[str, Any]]:
        chunks: List[Dict[str, Any]] = []
        next_id = 1
        for source_path, text in docs:
            paragraphs = [part.strip() for part in re.split(r"\n\s*\n", text) if part.strip()]
            doc_name = Path(source_path).name
            for para_index, paragraph in enumerate(paragraphs, start=1):
                sentences = split_sentences(paragraph)
                if not sentences:
                    continue
                current: List[str] = []
                i = 0
                while i < len(sentences):
                    sentence = sentences[i]
                    projected = len(" ".join(current + [sentence]))
                    if current and projected > max_chars:
                        chunk_text = sanitize_text(" ".join(current))
                        chunks.append(
                            {
                                "id": next_id,
                                "chunk_id": next_id,
                                "doc_name": doc_name,
                                "source_path": source_path,
                                "paragraph_index": para_index,
                                "text": chunk_text,
                            }
                        )
                        next_id += 1
                        overlap = current[-overlap_sentences:] if overlap_sentences > 0 else []
                        current = overlap[:]
                        continue
                    current.append(sentence)
                    i += 1
                if current:
                    chunk_text = sanitize_text(" ".join(current))
                    chunks.append(
                        {
                            "id": next_id,
                            "chunk_id": next_id,
                            "doc_name": doc_name,
                            "source_path": source_path,
                            "paragraph_index": para_index,
                            "text": chunk_text,
                        }
                    )
                    next_id += 1
        return chunks

    def _corpus_hash(self, docs: List[Tuple[str, str]]) -> str:
        joined = "\n".join(f"{path}:{hash_text(text)}" for path, text in docs)
        return hash_text(joined)

    def status(self, *, embedding_provider: Optional[str], embedding_model: Optional[str]) -> Dict[str, Any]:
        collection_name = self.config.get_collection_name(embedding_provider, embedding_model)
        client = self._client()
        try:
            exists = client.has_collection(collection_name=collection_name)
        finally:
            client.close()
        manifest = {}
        if self.config.manifest_path.exists():
            try:
                manifest = json.loads(self.config.manifest_path.read_text(encoding="utf-8"))
            except Exception:
                manifest = {}
        return {
            "db_path": str(self.config.db_path),
            "manifest_path": str(self.config.manifest_path),
            "collection_name": collection_name,
            "collection_exists": exists,
            "manifest": manifest,
        }

    def build_or_reuse(
        self,
        *,
        embedding_provider: Optional[str],
        embedding_model: Optional[str],
        force_rebuild: bool = False,
    ) -> Dict[str, Any]:
        docs = self.load_documents()
        chunks = self.chunk_documents(docs)
        corpus_hash = self._corpus_hash(docs)
        provider_name = self.config.get_embedding_provider(embedding_provider)
        model = embedding_model or self.config.get_embedding_model(provider_name)
        collection_name = self.config.get_collection_name(provider_name, model)

        manifest = {}
        if self.config.manifest_path.exists():
            try:
                manifest = json.loads(self.config.manifest_path.read_text(encoding="utf-8"))
            except Exception:
                manifest = {}

        client = self._client()
        try:
            exists = client.has_collection(collection_name=collection_name)
            reusable = (
                exists
                and not force_rebuild
                and manifest.get("collection_name") == collection_name
                and manifest.get("corpus_hash") == corpus_hash
                and manifest.get("chunk_count") == len(chunks)
            )
            if reusable:
                return {
                    "status": "reused",
                    "db_path": str(self.config.db_path),
                    "collection_name": collection_name,
                    "chunk_count": len(chunks),
                    "embedding_provider": provider_name,
                    "embedding_model": model,
                }

            if exists:
                client.drop_collection(collection_name=collection_name)

            vectors = self.provider.embed_texts([item["text"] for item in chunks], provider=provider_name)
            dimension = len(vectors[0])
            client.create_collection(
                collection_name=collection_name,
                dimension=dimension,
                metric_type="COSINE",
                auto_id=False,
            )
            rows = []
            for index, chunk in enumerate(chunks):
                rows.append(
                    {
                        "id": chunk["id"],
                        "vector": vectors[index],
                        "chunk_id": chunk["chunk_id"],
                        "doc_name": chunk["doc_name"],
                        "source_path": chunk["source_path"],
                        "text": chunk["text"],
                    }
                )
            client.insert(collection_name=collection_name, data=rows)
        finally:
            client.close()

        self.config.manifest_path.write_text(
            json.dumps(
                {
                    "collection_name": collection_name,
                    "corpus_hash": corpus_hash,
                    "chunk_count": len(chunks),
                    "embedding_provider": provider_name,
                    "embedding_model": model,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        return {
            "status": "created",
            "db_path": str(self.config.db_path),
            "collection_name": collection_name,
            "chunk_count": len(chunks),
            "embedding_provider": provider_name,
            "embedding_model": model,
        }

    def retrieve(
        self,
        query: str,
        *,
        top_k: int,
        embedding_provider: Optional[str],
        embedding_model: Optional[str] = None,
    ) -> List[Evidence]:
        provider_name = self.config.get_embedding_provider(embedding_provider)
        model = embedding_model or self.config.get_embedding_model(provider_name)
        collection_name = self.config.get_collection_name(provider_name, model)
        client = self._client()
        try:
            if not client.has_collection(collection_name=collection_name):
                raise FileNotFoundError(
                    f"RAG collection '{collection_name}' not found. Run `python run_milvius.py --embedding-provider {provider_name}` first."
                )
            query_vector = self.provider.embed_texts([query], provider=provider_name)[0]
            results = client.search(
                collection_name=collection_name,
                data=[query_vector],
                anns_field="vector",
                output_fields=["chunk_id", "doc_name", "source_path", "text"],
                limit=top_k,
            )
        finally:
            client.close()

        evidence: List[Evidence] = []
        for idx, item in enumerate(results[0], start=1):
            entity = item.get("entity", {})
            evidence.append(
                Evidence(
                    evidence_id=f"rag_{idx}",
                    title=entity.get("doc_name", "RAG chunk"),
                    excerpt=shorten(entity.get("text", ""), 320),
                    source_type="rag",
                    source_path=entity.get("source_path"),
                    score=float(item.get("distance", 0.0)),
                )
            )
        return evidence

