"""Render a SuiteResult as Markdown."""

from __future__ import annotations

from typing import Any


def render_markdown(suite: dict[str, Any]) -> str:
    lines = ["# RAG eval report\n"]
    if (rs := suite.get("retrieval")):
        lines.append("## Retrieval\n")
        lines.append(f"_Queries: {rs['n_queries']}, k = {rs['k']}_\n")
        lines.append("| metric | value |")
        lines.append("| --- | --- |")
        for key in ("recall_at_k", "precision_at_k", "mrr", "ndcg_at_k", "map", "hit_rate_at_k", "coverage"):
            lines.append(f"| {key} | {rs[key]:.4f} |")
        lines.append("")
    if (fe := suite.get("filter_exclusion")):
        lines.append("## Filter false-exclusion\n")
        lines.append(f"- rate: **{fe['rate']:.2%}**")
        lines.append(f"- excluded: {fe['n_excluded']} of {fe['n_queries']}")
        lines.append("")
    if (lat := suite.get("latency")):
        lines.append("## Latency (ms)\n")
        lines.append("| stage | p50 | p95 | p99 | n |")
        lines.append("| --- | --- | --- | --- | --- |")
        for stage, stats in lat.items():
            lines.append(
                f"| {stage} | {stats['p50']:.1f} | {stats['p95']:.1f} | {stats['p99']:.1f} | {stats['n']} |"
            )
        lines.append("")
    if (gates := suite.get("gates")):
        lines.append("## Gates\n")
        lines.append("| metric | observed | threshold | pass |")
        lines.append("| --- | --- | --- | --- |")
        for g in gates:
            lines.append(
                f"| {g['name']} | {g['observed']:.4f} | {g['threshold']:.4f} | {'PASS' if g['pass'] else 'FAIL'} |"
            )
        lines.append("")
    return "\n".join(lines)
