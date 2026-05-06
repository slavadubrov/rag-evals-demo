"""Ingest scifact into Qdrant + build the golden sets."""

from __future__ import annotations

from rich import print as rprint

from rag_evals.data import golden, scifact
from rag_evals.index.qdrant_store import QdrantStore
from rag_evals.ingest.pipeline import ingest


def main() -> None:
    store = QdrantStore()
    n_docs = ingest(scifact.documents(), store=store)
    rprint(f"[green]Ingested[/green] {n_docs} chunks into '{store.collection}'")

    paths = golden.build_all()
    for name, path in paths.items():
        n = sum(1 for _ in path.open())
        rprint(f"  [cyan]{name}[/cyan]: {n} rows -> {path}")


if __name__ == "__main__":
    main()
