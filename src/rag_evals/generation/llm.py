"""Bounded provider calls and strict structured outputs. No live-to-mock fallback."""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from rag_evals.config import settings
from rag_evals.generation.models import Model, required_env_var

FIXTURES_DIR = Path(__file__).resolve().parents[3] / "tests" / "fixtures" / "llm"


class InvalidResponse(ValueError):
    """Refused, truncated, empty or schema-invalid provider response."""


def _hash_prompt(messages: list[dict[str, Any]], model: str) -> str:
    return hashlib.sha1(
        json.dumps({"m": model, "msgs": messages}, sort_keys=True).encode()
    ).hexdigest()


class MockBackend:
    """Explicit fixture replay; unrecorded structured judgments stay invalid."""

    def __init__(self, fixtures_dir: Path = FIXTURES_DIR) -> None:
        self.fixtures_dir = fixtures_dir

    def complete(self, model: str, messages: list[dict[str, Any]]) -> str:
        sha = _hash_prompt(messages, model)
        path = self.fixtures_dir / f"{sha}.json"
        if path.exists():
            return json.loads(path.read_text())["content"]
        return f"[mock:{model}:{sha[:6]}] unrecorded response"


class LLM:
    def __init__(self, model: Model | str | None = None, *, mode: str | None = None) -> None:
        self.model = str(model or settings.rag_evals_default_model)
        env = required_env_var(self.model)
        backend = mode or settings.rag_evals_backend
        if backend not in {"auto", "live", "mock"}:
            raise ValueError(f"Unknown backend: {backend}")
        self.mode = "mock" if self.model == "mock" else backend
        if self.mode == "auto":
            self.mode = "live" if env and os.getenv(env) else "mock"
        if self.mode == "live" and env and not os.getenv(env):
            raise RuntimeError(f"Missing {env} for {self.model}")
        self._mock = MockBackend()
        self.calls: list[dict[str, Any]] = []

    def complete(
        self,
        messages: list[dict[str, Any]],
        *,
        temperature: float = 0.0,
        max_tokens: int | None = None,
        response_format: dict[str, Any] | None = None,
    ) -> str:
        if len(self.calls) >= settings.llm_max_calls:
            raise RuntimeError("LLM call budget exhausted")
        cap = settings.llm_max_tokens if max_tokens is None else max_tokens
        if cap <= 0:
            raise ValueError("max_tokens must be positive")
        if temperature != 0.0:
            raise ValueError("Temperature is not configurable for the reasoning-model adapter")
        if self.mode == "mock":
            return self._mock.complete(self.model, messages)
        kwargs: dict[str, Any] = {"model": self.model, "messages": messages}
        if response_format is not None:
            kwargs["response_format"] = response_format
        kwargs["model"] = self.model.removeprefix("openai/")
        kwargs["max_completion_tokens"] = cap
        kwargs["reasoning_effort"] = settings.reasoning_effort
        event: dict[str, Any] = {
            "model": self.model,
            "status": "error",
            "prompt_tokens": None,
            "completion_tokens": None,
            "cost_usd": None,
        }
        started = time.perf_counter()
        self.calls.append(event)
        try:
            from openai import OpenAI

            with OpenAI(timeout=settings.llm_timeout, max_retries=0) as client:
                resp = client.chat.completions.create(**kwargs)
            choice = resp.choices[0]
            if resp.usage:
                event.update(
                    prompt_tokens=resp.usage.prompt_tokens,
                    completion_tokens=resp.usage.completion_tokens,
                )
            event["resolved_model"] = resp.model
            if choice.finish_reason != "stop" or getattr(choice.message, "refusal", None):
                raise InvalidResponse("Response refused or incomplete")
            content = choice.message.content
            if not content or not content.strip():
                raise InvalidResponse("Empty response")
            event["status"] = "ok"
            return str(content)
        finally:
            event["latency_ms"] = (time.perf_counter() - started) * 1000

    def ask(self, prompt: str, *, system: str | None = None, **kw: Any) -> str:
        msgs = []
        if system:
            msgs.append({"role": "system", "content": system})
        msgs.append({"role": "user", "content": prompt})
        return self.complete(msgs, **kw)

    def structured[T: BaseModel](self, prompt: str, schema: type[T], *, system: str) -> T:
        """Constrain the output at the provider and validate again at the boundary."""
        fmt = {
            "type": "json_schema",
            "json_schema": {
                "name": schema.__name__,
                "strict": True,
                "schema": schema.model_json_schema(),
            },
        }
        raw = self.ask(prompt, system=system, response_format=fmt)
        try:
            return schema.model_validate_json(raw, strict=True)
        except ValueError as exc:
            if self.calls:
                self.calls[-1]["status"] = "invalid_schema"
            raise InvalidResponse(f"Invalid {schema.__name__} output") from exc
