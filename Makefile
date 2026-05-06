SHELL := /bin/bash
.DEFAULT_GOAL := help

UV ?= uv

define banner
	@printf "\n\033[1;34m▶ %s\033[0m\n" "$(1)"
endef

.PHONY: help install up down index golden eval eval-fast nb test lint fmt clean

help:  ## Show this help.
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)

install:  ## uv sync (creates .venv, installs all extras)
	$(call banner,uv sync)
	$(UV) sync --all-extras

up:  ## docker compose up -d qdrant; wait until healthy
	$(call banner,Qdrant up)
	docker compose up -d qdrant
	@for i in 1 2 3 4 5 6 7 8 9 10; do \
		curl -sf http://localhost:6333/readyz >/dev/null && echo "Qdrant ready" && exit 0; \
		sleep 1; \
	done; echo "Qdrant did not become ready" && exit 1

down:  ## docker compose down -v
	$(call banner,Qdrant down)
	docker compose down -v

index:  ## Ingest scifact into Qdrant
	$(call banner,Seeding Qdrant index)
	$(UV) run python -m rag_evals.scripts.seed_index

golden:  ## Rebuild golden sets from scifact qrels
	$(call banner,Building golden sets)
	$(UV) run python -m rag_evals.data.golden

eval:  ## Run full eval suite, write report.md, fail on regressions
	$(call banner,Eval suite)
	$(UV) run python -m rag_evals.evaluation.runner --suite all --report report.md

eval-fast:  ## 50-query smoke subset for CI
	$(call banner,Eval smoke)
	$(UV) run python -m rag_evals.evaluation.runner --suite all --limit 50 --report report.md

nb:  ## Execute all notebooks (uses MockBackend)
	$(call banner,Notebook execution)
	RAG_EVALS_BACKEND=mock $(UV) run jupyter nbconvert --to notebook --execute --inplace notebooks/*.ipynb

test:  ## pytest
	$(call banner,Tests)
	$(UV) run pytest

lint:  ## ruff + mypy
	$(call banner,Lint)
	$(UV) run ruff check .
	$(UV) run mypy src

fmt:  ## ruff format
	$(call banner,Format)
	$(UV) run ruff format .

clean:  ## Remove caches and artefacts
	$(call banner,Clean)
	rm -rf .venv .pytest_cache .mypy_cache .ruff_cache report.md report.json data/cache __pycache__
	find . -type d -name __pycache__ -exec rm -rf {} +
