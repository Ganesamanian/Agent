from __future__ import annotations

from openai import OpenAI
from google import genai
from typing import List, Optional, Dict, Any, Tuple

from .config import AppConfig
from .utils import safe_json_loads, local_embed_texts


class ModelProvider:
    def __init__(self, config: AppConfig):
        self.config = config
        self._openai = None
        self._gemini = None

    @property
    def openai(self):
        if self._openai is None and OpenAI and self.config.openai_api_key:
            self._openai = OpenAI(api_key=self.config.openai_api_key)
        return self._openai

    @property
    def gemini(self):
        if self._gemini is None and genai and self.config.gemini_api_key:
            self._gemini = genai.Client(api_key=self.config.gemini_api_key)
        return self._gemini

    def generate_text(self, *, provider: Optional[str], system_prompt: str, user_prompt: str) -> Tuple[str, int]:
        try:
            provider = (provider or self.config.llm_provider or "openai").lower()
            if provider == "openai" and self.openai is not None:
                response = self.openai.chat.completions.create(
                    model=self.config.get_generation_model("openai"),
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=0.2,
                )
                text = response.choices[0].message.content or ""
                total_tokens = getattr(response.usage, 'total_tokens', 0) if hasattr(response, 'usage') and response.usage else 0
                return text, total_tokens
            if provider == "gemini" and self.gemini is not None:
                response = self.gemini.models.generate_content(
                    model=self.config.get_generation_model("gemini"),
                    contents=f"System instructions:\n{system_prompt}\n\nUser prompt:\n{user_prompt}",
                )
                text = getattr(response, "text", "") or ""
                total_tokens = 0
                return text, total_tokens
        except Exception:
            # LLM API failure (network/auth/rate limit): return empty response
            pass
        return "", 0

    def generate_json(self, *, provider: Optional[str], system_prompt: str, user_prompt: str) -> Tuple[Optional[Dict[str, Any]], int]:
        try:
            text, tokens = self.generate_text(provider=provider, system_prompt=system_prompt, user_prompt=user_prompt)
            json_data = safe_json_loads(text)
            return json_data, tokens
        except Exception:
            # JSON parsing or generation failed: return None
            return None, 0

    def embed_texts(self, texts: List[str], *, provider: Optional[str]) -> List[List[float]]:
        try:
            provider = self.config.get_embedding_provider(provider)
            if provider == "openai" and self.openai is not None:
                response = self.openai.embeddings.create(
                    model=self.config.get_embedding_model("openai"),
                    input=texts,
                )
                return [item.embedding for item in response.data]
            if provider == "gemini" and self.gemini is not None:
                response = self.gemini.models.embed_content(
                    model=self.config.get_embedding_model("gemini"),
                    contents=texts,
                )
                embeddings = getattr(response, "embeddings", None) or []
                values: List[List[float]] = []
                for item in embeddings:
                    vector = getattr(item, "values", None)
                    if vector is not None:
                        values.append(list(vector))
                if values:
                    return values
        except Exception:
            # Embedding API failure: fallback to local embeddings
            pass
        return local_embed_texts(texts)

