from __future__ import annotations

import time

from rag_evals.evaluation.latency import Tracer, quantiles, summarise


def test_tracer_records_stage_times() -> None:
    t = Tracer()
    with t.stage("retrieve"):
        time.sleep(0.001)
    with t.stage("generate"):
        time.sleep(0.001)
    by = t.by_stage()
    assert "retrieve" in by and "generate" in by


def test_quantiles_monotonic() -> None:
    q = quantiles([1.0, 2.0, 3.0, 4.0, 5.0, 10.0, 100.0])
    assert q.p50 <= q.p95 <= q.p99
    assert q.n == 7


def test_summarise_merges_tracers() -> None:
    a = Tracer()
    a.events.append(("retrieve", 5.0))
    b = Tracer()
    b.events.append(("retrieve", 10.0))
    s = summarise([a, b])
    assert s["retrieve"].n == 2
