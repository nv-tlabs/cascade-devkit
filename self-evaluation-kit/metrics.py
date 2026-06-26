"""Public retrieval metrics for local self-evaluation."""
from __future__ import annotations

from collections.abc import Mapping, Sequence


def dedupe_preserve_order(items: Sequence[str]) -> list[str]:
    seen = set()
    out = []
    for item in items:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out


def precision_at_k(ranked_ids: Sequence[str], relevant_ids: set[str], k: int) -> float:
    if k <= 0:
        return 0.0
    ranked = dedupe_preserve_order(ranked_ids)[:k]
    if not ranked:
        return 0.0
    hits = sum(1 for video_id in ranked if video_id in relevant_ids)
    return hits / k


def recall_at_k(ranked_ids: Sequence[str], relevant_ids: set[str], k: int) -> float:
    if k <= 0 or not relevant_ids:
        return 0.0
    ranked = dedupe_preserve_order(ranked_ids)[:k]
    hits = sum(1 for video_id in ranked if video_id in relevant_ids)
    return hits / len(relevant_ids)


def hit_at_k(ranked_ids: Sequence[str], relevant_ids: set[str], k: int) -> float:
    if k <= 0 or not relevant_ids:
        return 0.0
    ranked = dedupe_preserve_order(ranked_ids)[:k]
    return float(any(video_id in relevant_ids for video_id in ranked))


def average_precision(ranked_ids: Sequence[str], relevant_ids: set[str]) -> float:
    if not relevant_ids:
        return 0.0
    ranked = dedupe_preserve_order(ranked_ids)
    hits = 0
    precision_sum = 0.0
    for rank, video_id in enumerate(ranked, start=1):
        if video_id in relevant_ids:
            hits += 1
            precision_sum += hits / rank
    return precision_sum / len(relevant_ids)


def r_precision(ranked_ids: Sequence[str], relevant_ids: set[str]) -> float:
    if not relevant_ids:
        return 0.0
    r = len(relevant_ids)
    ranked = dedupe_preserve_order(ranked_ids)[:r]
    hits = sum(1 for video_id in ranked if video_id in relevant_ids)
    return hits / r


def evaluate_run(
    run: Mapping[str, Sequence[str]],
    qrels: Mapping[str, set[str]],
    k_values: Sequence[int] = (1, 3, 5, 10),
) -> dict:
    scored_query_ids = [qid for qid, relevant in qrels.items() if relevant]
    if not scored_query_ids:
        raise ValueError("No queries with at least one relevant video.")

    per_query_ap = []
    per_query_rprec = []
    p_at_k = {k: [] for k in k_values}
    r_at_k = {k: [] for k in k_values}
    h_at_k = {k: [] for k in k_values}

    for qid in scored_query_ids:
        ranked = run.get(qid, [])
        relevant = qrels[qid]
        per_query_ap.append(average_precision(ranked, relevant))
        per_query_rprec.append(r_precision(ranked, relevant))
        for k in k_values:
            p_at_k[k].append(precision_at_k(ranked, relevant, k))
            r_at_k[k].append(recall_at_k(ranked, relevant, k))
            h_at_k[k].append(hit_at_k(ranked, relevant, k))

    def mean(values: Sequence[float]) -> float:
        return sum(values) / len(values)

    return {
        "primary_metric": "average_r_precision",
        "average_r_precision": mean(per_query_rprec),
        "map": mean(per_query_ap),
        "precision_at_k": {str(k): mean(v) for k, v in p_at_k.items()},
        "recall_at_k": {str(k): mean(v) for k, v in r_at_k.items()},
        "hit_at_k": {str(k): mean(v) for k, v in h_at_k.items()},
        "num_queries": len(qrels),
        "num_scored_queries": len(scored_query_ids),
        "num_queries_without_relevance": len(qrels) - len(scored_query_ids),
    }
