from __future__ import annotations

from rag_evals.data.metadata import DOMAINS, LOCALES, TENANTS, synthesize


def test_synthesize_is_deterministic() -> None:
    assert synthesize("doc-1") == synthesize("doc-1")


def test_synthesize_produces_valid_values() -> None:
    m = synthesize("doc-42")
    assert m["tenant"] in TENANTS
    assert m["locale"] in LOCALES
    assert m["domain"] in DOMAINS


def test_distribution_covers_all_buckets() -> None:
    seen = {"tenant": set(), "locale": set(), "domain": set()}
    for i in range(500):
        m = synthesize(f"doc-{i}")
        for k, v in m.items():
            seen[k].add(v)
    assert seen["tenant"] == set(TENANTS)
    assert seen["locale"] == set(LOCALES)
    assert seen["domain"] == set(DOMAINS)
