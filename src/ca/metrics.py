"""Headline and auxiliary metrics computed from a run summary.

v7 headlines: latency (arrival -> delivery, in rounds), F1, coverage
(answered / arrived) and tokens_per_answer -- the proactive arm pre-pays
compute while idle, and these read off whether that bought anything.
"""
import statistics
from collections import defaultdict


def specialization(summary: dict) -> dict[str, float]:
    """Herfindahl index (Sum share^2) of each agent's delivered questions across
    TOPICS: 1.0 = every delivered question came from one topic (fully
    specialized), -> 1/n for an agent spread evenly over n topics (generalist).
    Topics are bank metadata the agents never see, which is exactly what makes
    the measure honest. Agents with zero deliveries are omitted -- there is
    nothing to compute a distribution over."""
    counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for d in summary.get("deliveries", []):
        counts[d["agent"]][d.get("topic") or ""] += 1
    result = {}
    for agent, buckets in counts.items():
        total = sum(buckets.values())
        result[agent] = sum((c / total) ** 2 for c in buckets.values()) if total else 0.0
    return result


def compute_metrics(summary: dict) -> dict:
    deliveries = summary.get("deliveries", [])
    n_answered = len(deliveries)
    total_f1 = sum(d["f1"] for d in deliveries)
    total_em = sum(d["em"] for d in deliveries)
    latencies = [d["latency"] for d in deliveries]
    solving = sum(t["solving"] for t in summary["tokens"].values())
    admin = sum(t["admin"] for t in summary["tokens"].values())
    all_tok = solving + admin
    arrived = summary.get("arrived_total", 0)
    n_messages = summary.get("n_messages", 0)
    turns = summary.get("turns", {})
    solving_turns = turns.get("solving", 0)
    agents = summary.get("agents", {})
    spec = specialization(summary)
    return {
        "n_answered": n_answered,
        "total_f1": total_f1,
        "total_em": total_em,
        "mean_f1": total_f1 / n_answered if n_answered else 0.0,
        "mean_em": total_em / n_answered if n_answered else 0.0,
        # headline: how much of what the WORLD actually asked got answered
        "coverage": n_answered / arrived if arrived else 0.0,
        # headline: rounds from arrival to delivery
        "mean_latency": sum(latencies) / n_answered if n_answered else 0.0,
        "median_latency": statistics.median(latencies) if latencies else 0.0,
        # the proactive arm's product, and how much of the solving effort it was
        "selfqa_total": summary.get("kb_selfqa", 0),
        "selfqa_per_agent": {a: v.get("selfqa", 0) for a, v in agents.items()},
        "proactive_ratio": (turns.get("selfqa", 0) / solving_turns
                            if solving_turns else 0.0),
        # headline efficiency metrics: tokens are measured, never charged
        "tokens_per_answer": all_tok / n_answered if n_answered else 0.0,
        "coordination_overhead": admin / all_tok if all_tok else 0.0,
        "rounds_used": summary["rounds_used"],
        "n_messages": n_messages,
        "messages_per_answer": n_messages / n_answered if n_answered else 0.0,
        "specialization": spec,
        "mean_specialization": (sum(spec.values()) / len(spec)) if spec else 0.0,
    }
