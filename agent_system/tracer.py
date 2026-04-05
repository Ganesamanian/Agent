from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Dict, Optional

from langfuse import Langfuse
from typing import Tuple


class Tracer:
    def __init__(self, public_key: str, secret_key: str, host: str, timeout: float = 30.0):
        self.client = None
        self.timeout = timeout
        if public_key and secret_key:
            self.client = Langfuse(
                public_key=public_key,
                secret_key=secret_key,
                host=host,
                timeout=timeout,
            )

    @contextmanager
    def observation(
        self,
        *,
        name: str,
        as_type: str = "span",
        input: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        if not self.client:
            yield None
            return

        with self.client.start_as_current_observation(
            name=name,
            as_type=as_type,
            input=input,
            **kwargs,
        ) as obs:
            yield obs

    def update_current_trace(self, **kwargs):
        if self.client:
            self.client.update_current_trace(**kwargs)

    def flush(self):
        if self.client:
            self.client.flush()


    def get_prompt_from_langfuse(self, prompt_name: str, query: str, context: str)-> Tuple[str, str]:
    
        prompt = self.client.get_prompt(
            prompt_name,
            label="production",
            type="chat",
        )

        compiled_messages = prompt.compile(
                query=query,
                context=context,
            )

        system_prompt = "\n\n".join(
            msg["content"]
            for msg in compiled_messages
            if msg.get("role") == "system"
        ).strip()

        user_prompt = "\n\n".join(
            msg["content"]
            for msg in compiled_messages
            if msg.get("role") == "user"
        ).strip()

        return system_prompt, user_prompt

