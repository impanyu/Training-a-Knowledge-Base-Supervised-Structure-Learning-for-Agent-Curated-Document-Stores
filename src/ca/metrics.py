"""Headline and auxiliary metrics computed from a run summary."""
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
    solving = sum(t["solving"] for t in summary["tokens"].values())
    admin = sum(t["admin"] for t in summary["tokens"].values())
    all_tok = solving + admin
    total_units = summary.get("total_units", 0)
    n_messages = summary.get("n_messages", 0)
    mem = summary.get("memory", {})
    n_claims = sum(v.get("n_claims", 0) for v in mem.values())
    n_memory_hits = sum(v.get("n_memory_hits", 0) for v in mem.values())
    n_repeats = sum(v.get("n_repeat_deliveries", 0) for v in mem.values())
    n_improved = sum(v.get("n_improved", 0) for v in mem.values())
    spec = specialization(summary)
    return {
        "n_answered": n_answered,
        "total_f1": total_f1,
        "total_em": total_em,
        "mean_f1": total_f1 / n_answered if n_answered else 0.0,
        "mean_em": total_em / n_answered if n_answered else 0.0,
        # headline: how much of the WORLD's posted demand -- the questions
        # themselves -- the system actually absorbed (closed / total)
        "demand_absorbed": n_answered / total_units if total_units else 0.0,
        "accuracy_per_ktok_solving": total_f1 / (solving / 1000) if solving else 0.0,
        "accuracy_per_ktok_all": total_f1 / (all_tok / 1000) if all_tok else 0.0,
        # headline efficiency metric: tokens are measured, never charged
        "coordination_overhead": admin / all_tok if all_tok else 0.0,
        "admin_solving_ratio": admin / solving if solving else 0.0,
        "rounds_used": summary["rounds_used"],
        "n_messages": n_messages,
        "messages_per_answer": n_messages / n_answered if n_answered else 0.0,
        "specialization": spec,
        "mean_specialization": (sum(spec.values()) / len(spec)) if spec else 0.0,
        # memory reuse: how often a claim came back with knowledge attached, and
        # how often a second attempt at a known question actually beat it
        "n_claims": n_claims,
        "memory_hit_rate": n_memory_hits / n_claims if n_claims else 0.0,
        "improvement_rate": n_improved / n_repeats if n_repeats else 0.0,
        "answers_in_memory_total": sum(v.get("answers", 0) for v in mem.values()),
    }
