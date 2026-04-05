from __future__ import annotations

from openai import OpenAI
import google as genai
from typing import List, Optional, Dict, Any

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
            self._openai = OpenAI(
                api_key=self.config.openai_api_key,
                timeout=self.config.llm_timeout,
            )
        return self._openai

    @property
    def gemini(self):
        if self._gemini is None and genai and self.config.gemini_api_key:
            genai.configure(api_key=self.config.gemini_api_key)
            self._gemini = True  # Flag configured; use genai.GenerativeModel directly
        return self._gemini

    def generate_text(self, *, provider: Optional[str], system_prompt: str, user_prompt: str) -> str:
        provider = (provider or self.config.llm_provider or "openai").lower()
        if provider == "openai" and self.openai is not None:
            response = self.openai.chat.completions.create(
                model=self.config.get_generation_model("openai"),
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.2,
                timeout=self.config.llm_timeout,
            )
            return response.choices[0].message.content or ""
        if provider == "gemini" and self.gemini:
            model_name = self.config.get_generation_model("gemini")
            model = genai.GenerativeModel(model_name)
            response = model.generate_content(
                f"System instructions:\n{system_prompt}\n\nUser prompt:\n{user_prompt}",
            )
            return getattr(response, "text", "") or ""
        return ""

    def generate_json(self, *, provider: Optional[str], system_prompt: str, user_prompt: str) -> Optional[Dict[str, Any]]:
        text = self.generate_text(provider=provider, system_prompt=system_prompt, user_prompt=user_prompt)
        return safe_json_loads(text)

    def embed_texts(self, texts: List[str], *, provider: Optional[str]) -> List[List[float]]:
        provider = self.config.get_embedding_provider(provider)
        if provider == "openai" and self.openai is not None:
            response = self.openai.embeddings.create(
                model=self.config.get_embedding_model("openai"),
                input=texts,
                timeout=self.config.embed_timeout,
            )
            return [item.embedding for item in response.data]
        if provider == "gemini" and self.gemini:
            model_name = self.config.get_embedding_model("gemini")
            result = genai.embed_content(
                model=model_name,
                content=texts,
            )
            embeddings = getattr(result, "embeddings", None) or []
            values: List[List[float]] = []
            for item in embeddings:
                vector = getattr(item, "values", None)
                if vector is not None:
                    values.append(list(vector))
            if values:
                return values
        return local_embed_texts(texts)

