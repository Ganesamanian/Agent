from __future__ import annotations

import hashlib
import json
import math
import re
from typing import Any, Dict, Iterable, List, Optional

DEFAULT_PUBLIC_URLS = [
    "https://www.booking.com/content/terms.html",
    "https://secure.booking.com/faq.en-us.html?aid=330843",
    "https://www.booking.com/content/privacy.html",
]

STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "from", "has", "have",
    "if", "in", "into", "is", "it", "its", "of", "on", "or", "our", "so", "that", "the",
    "their", "there", "they", "this", "to", "was", "we", "were", "what", "when", "where",
    "which", "who", "will", "with", "within", "after", "before", "still", "then", "than",
    "them", "about", "like", "should", "would", "could", "can", "customer", "guest", "booking",
    "bookingcom", "booking.com", "user", "goal", "complaint", "refund", "support", "legal",
}


def sanitize_text(text: str) -> str:
    return re.sub(r"\\s+", " ", text or "").strip()


def shorten(text: str, max_chars: int = 280) -> str:
    cleaned = sanitize_text(text)
    if len(cleaned) <= max_chars:
        return cleaned
    return cleaned[: max_chars - 3].rstrip() + "..."


def safe_json_loads(text: str) -> Optional[Dict[str, Any]]:
    candidate = (text or "").strip()
    if not candidate:
        return None
    if candidate.startswith("```"):
        candidate = re.sub(r"^```(?:json)?", "", candidate).strip()
        candidate = re.sub(r"```$", "", candidate).strip()
    try:
        parsed = json.loads(candidate)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass
    return None


def hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def split_sentences(text: str) -> List[str]:
    parts = re.split(r"(?<=[.!?])\\s+", sanitize_text(text))
    return [part.strip() for part in parts if part.strip()]


def extract_keywords(text: str, max_keywords: int = 6) -> List[str]:
    tokens = re.findall(r"[a-zA-Z][a-zA-Z0-9_-]{2,}", (text or "").lower())
    counts: Dict[str, int] = {}
    for token in tokens:
        if token in STOPWORDS:
            continue
        counts[token] = counts.get(token, 0) + 1
    ranked = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    keywords = [token for token, _ in ranked[:max_keywords]]
    return keywords or ["cancellation", "refund", "payment"]


def lexical_overlap_score(text: str, keywords: Iterable[str]) -> float:
    lowered = (text or "").lower()
    return float(sum(1 for keyword in keywords if keyword.lower() in lowered))


def local_embed_texts(texts: List[str], dim: int = 256) -> List[List[float]]:
    vectors: List[List[float]] = []
    for text in texts:
        vec = [0.0] * dim
        for token in re.findall(r"\\w+", (text or "").lower()):
            index = hash(token) % dim
            vec[index] += 1.0
        norm = math.sqrt(sum(value * value for value in vec)) or 1.0
        vectors.append([value / norm for value in vec])
    return vectors

