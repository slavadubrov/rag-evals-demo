"""Deterministic synthetic metadata layered on top of scifact.

The same doc_id always produces the same (tenant, locale, domain) triple.
Used by the filter false-exclusion demo: predicates that look reasonable
can silently zero out recall, and the harness must catch it.
"""

from __future__ import annotations

import hashlib
from typing import Any

TENANTS = ("acme", "globex", "initech")
LOCALES = ("en-US", "en-GB", "de-DE")
DOMAINS = ("biomed", "clinical", "public-health")


def _bucket(doc_id: str, salt: str, options: tuple[str, ...]) -> str:
    h = hashlib.sha1(f"{salt}:{doc_id}".encode()).digest()
    return options[h[0] % len(options)]


def synthesize(doc_id: str) -> dict[str, Any]:
    return {
        "tenant": _bucket(doc_id, "tenant", TENANTS),
        "locale": _bucket(doc_id, "locale", LOCALES),
        "domain": _bucket(doc_id, "domain", DOMAINS),
    }
