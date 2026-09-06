"""Unit tests must never use real credentials from the developer's environment."""

import os

os.environ["RAG_EVALS_BACKEND"] = "mock"
