"""rag-evals CLI."""

from __future__ import annotations

import typer

from rag_evals.evaluation.runner import app as runner_app
from rag_evals.scripts import seed_index

app = typer.Typer(help="RAG evaluation harness", add_completion=False)


@app.command()
def seed() -> None:
    """Ingest scifact into Qdrant."""
    seed_index.main()


app.add_typer(runner_app, name="eval", help="Run eval suites")


if __name__ == "__main__":
    app()
