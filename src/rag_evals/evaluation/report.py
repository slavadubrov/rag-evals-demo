"""Render a SuiteResult as Markdown."""

from __future__ import annotations

from typing import Any


def render_markdown(suite: dict[str, Any]) -> str:
    lines = ["# RAG eval report\n"]
    if rs := suite.get("retrieval"):
        lines.append("## Retrieval\n")
        lines.append(f"_Queries: {rs['n_queries']}, k = {rs['k']}_\n")
        lines.append("| metric | value |")
        lines.append("| --- | --- |")
        for key in (
            "recall_at_k",
            "precision_at_k",
            "mrr",
            "ndcg_at_k",
            "map",
            "hit_rate_at_k",
            "coverage",
        ):
            label = {
                "recall_at_k": f"Recall@{rs['k']}",
                "precision_at_k": f"Precision@{rs['k']}",
                "ndcg_at_k": f"nDCG@{rs['k']}",
                "hit_rate_at_k": f"Hit@{rs['k']}",
            }.get(key, key)
            lines.append(f"| {label} | {rs[key]:.4f} |")
        lines.append("")
    if fe := suite.get("filter_exclusion"):
        lines.append("## Filter false-exclusion\n")
        lines.append(f"- population: {fe.get('population', 'provided filter cases')}")
        lines.append(f"- rate: **{fe['rate']:.2%}**")
        lines.append(f"- excluded: {fe['n_excluded']} of {fe['n_queries']}")
        if clean := fe.get("clean"):
            lines.append(f"- clean subset: {clean['n_excluded']} excluded of {clean['n_queries']}")
        lines.append("")
    if lat := suite.get("latency"):
        lines.append("## Latency (ms)\n")
        lines.append("| stage | p50 | p95 | p99 | n |")
        lines.append("| --- | --- | --- | --- | --- |")
        for stage, stats in lat.items():
            lines.append(
                f"| {stage} | {stats['p50']:.1f} | {stats['p95']:.1f} | {stats['p99']:.1f} | {stats['n']} |"
            )
        lines.append("")
    if calibration := suite.get("calibration"):
        lines.extend(
            [
                "## Judge calibration",
                "",
                f"Attempts: {calibration['n_attempted']}; invalid: {calibration['n_invalid']}; agreement on all attempts: {calibration['agreement_all_attempts']:.3f}.",
                "",
            ]
        )
    if gen := suite.get("generation"):
        lines.extend(
            [
                "## Generation",
                "",
                f"Population: {gen['population']}; replay={gen['replay']}; attempts={gen['n_attempted']}; failures={gen['n_failures']}.",
                "",
                "Replay validates fixtures, not model quality. Same-provider judgments require human calibration.",
                "",
                "| metric | mean | scored | invalid |",
                "| --- | --- | --- | --- |",
            ]
        )
        for name, metric in gen["metrics"].items():
            lines.append(
                f"| {name} | {metric['mean']} | {metric['n_scored']} | {metric.get('n_invalid', 0)} |"
            )
    if gates := suite.get("gates"):
        lines.append("## Gates\n")
        lines.append("| metric | observed | threshold | pass |")
        lines.append("| --- | --- | --- | --- |")
        for g in gates:
            lines.append(
                f"| {g['name']} | {g['observed']} | {g['threshold']:.4f} | {'PASS' if g['pass'] else 'FAIL'} |"
            )
        lines.append("")
    return "\n".join(lines)
