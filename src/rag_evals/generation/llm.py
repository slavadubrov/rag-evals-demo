"""LLM façade: one ``LLM`` class wrapping ``litellm.completion`` + a
deterministic ``MockBackend`` keyed by SHA1 of the prompt.

The whole project goes through this class. To add a provider, extend the
``Model`` enum.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

from rag_evals.config import settings
from rag_evals.generation.models import Model, required_env_var

FIXTURES_DIR = Path(__file__).resolve().parents[3] / "tests" / "fixtures" / "llm"


def _hash_prompt(messages: list[dict[str, Any]], model: str) -> str:
    payload = json.dumps({"m": model, "msgs": messages}, sort_keys=True)
    return hashlib.sha1(payload.encode()).hexdigest()


class MockBackend:
    """Returns a canned response keyed by SHA1 of (model, messages).

    Looks for ``tests/fixtures/llm/<sha>.json`` and returns its ``content``.
    Falls back to a deterministic stub if no fixture is found, so tests can
    still run without authoring fixtures for every prompt.
    """

    def __init__(self, fixtures_dir: Path = FIXTURES_DIR) -> None:
        self.fixtures_dir = fixtures_dir

    def complete(self, model: str, messages: list[dict[str, Any]]) -> str:
        sha = _hash_prompt(messages, model)
        path = self.fixtures_dir / f"{sha}.json"
        if path.exists():
            return json.loads(path.read_text())["content"]
        last = messages[-1]["content"] if messages else ""
        return f"[mock:{model}:{sha[:6]}] echoes: {last[:200]}"


class LLM:
    def __init__(self, model: Model | str = Model.GPT_5_MINI) -> None:
        self.model = Model(model) if not isinstance(model, Model) else model
        self.mode = self._resolve_mode()
        self._mock = MockBackend()

    def _resolve_mode(self) -> str:
        backend = settings.rag_evals_backend.lower()
        if backend == "mock" or self.model is Model.MOCK:
            return "mock"
        if backend == "live":
            self._require_key()
            return "live"
        # auto: live if the matching API key is set, else mock
        env = required_env_var(self.model)
        if env and os.getenv(env):
            return "live"
        return "mock"

    def _require_key(self) -> None:
        env = required_env_var(self.model)
        if env and not os.getenv(env):
            raise RuntimeError(
                f"RAG_EVALS_BACKEND=live but ${env} is not set for model {self.model.value}"
            )

    def complete(
        self,
        messages: list[dict[str, Any]],
        *,
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> str:
        if self.mode == "mock":
            return self._mock.complete(self.model.value, messages)
        import litellm  # local import to keep CLI cold-start fast

        kwargs: dict[str, Any] = {
            "model": self.model.value,
            "messages": messages,
            "max_tokens": max_tokens,
        }
        # gpt-5 family is a reasoning model — only temperature=1 is allowed,
        # and reasoning tokens count against ``max_tokens`` (so we keep it
        # generous enough for the answer to fit alongside reasoning).
        if self.model.value.startswith("gpt-5"):
            kwargs["temperature"] = 1.0
        else:
            kwargs["temperature"] = temperature
        resp = litellm.completion(**kwargs)
        return resp.choices[0].message.content or ""

    def ask(self, prompt: str, *, system: str | None = None, **kw: Any) -> str:
        msgs: list[dict[str, Any]] = []
        if system:
            msgs.append({"role": "system", "content": system})
        msgs.append({"role": "user", "content": prompt})
        return self.complete(msgs, **kw)
