"""One bounded structured call per model; reports identifiers and status, never keys."""

from __future__ import annotations

import json
from pathlib import Path

import typer

from rag_evals.evaluation.schemas import JUDGE_SYSTEM, Support
from rag_evals.generation.llm import LLM

app = typer.Typer(add_completion=False)


@app.command()
def main(
    models: list[str] = typer.Argument(...),  # noqa: B008
    out: Path = Path("report/model-probe.json"),
) -> None:
    rows: list[dict] = []
    for model in models:
        try:
            llm = LLM(model, mode="live")
            result = llm.structured(
                "Context: Mars has two moons. Claim: Mars has two moons. Return SUPPORTED with exact evidence_quote.",
                Support,
                system=JUDGE_SYSTEM,
            )
            rows.append(
                {"model": model, "status": "ok", "verdict": result.verdict, "calls": llm.calls}
            )
        except Exception as exc:
            rows.append(
                {
                    "model": model,
                    "status": "error",
                    "error_type": type(exc).__name__,
                    "http_status": getattr(exc, "status_code", None),
                }
            )
        print(json.dumps(rows[-1]))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, indent=2))


if __name__ == "__main__":
    app()
