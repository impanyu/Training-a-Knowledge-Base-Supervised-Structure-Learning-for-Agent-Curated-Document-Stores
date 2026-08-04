"""Headline and auxiliary metrics computed from a run summary."""
from collections import defaultdict


def gini(values: list[int | float]) -> float:
    vals = sorted(values)
    n = len(vals)
    total = sum(vals)
    if n == 0 or total == 0:
        return 0.0
    cum = 0.0
    for i, v in enumerate(vals, start=1):
        cum += i * v
    return (2 * cum) / (n * total) - (n + 1) / n


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


def delegation_rate(deliveries: list[dict], contracts: list[dict]) -> float:
    """Share of delivered (job, question) units whose answer was bought rather
    than produced: a qid-bound contract that some OTHER agent actually
    delivered. This is the v4.1 headline -- the market either exists or it does
    not, and this number says which."""
    sellers: dict[str, set[str]] = defaultdict(set)
    for c in contracts:
        if c.get("status") == "delivered" and c.get("qid"):
            sellers[c["qid"]].add(c["contractor"])
    if not deliveries:
        return 0.0
    bought = sum(1 for d in deliveries
                 if any(s != d["agent"] for s in sellers.get(d["qid"], ())))
    return bought / len(deliveries)


def compute_metrics(summary: dict) -> dict:
    deliveries = summary.get("deliveries", [])
    total_f1 = sum(d["f1"] for d in deliveries)
    total_em = sum(d["em"] for d in deliveries)
    solving = sum(t["solving"] for t in summary["tokens"].values())
    admin = sum(t["admin"] for t in summary["tokens"].values())
    all_tok = solving + admin
    prices = summary.get("contract_prices", [])
    total_units = summary.get("total_units", 0)
    jobs_posted = summary.get("jobs_posted", 0)
    loans = summary.get("loans", {})
    debtors = loans.get("debtors", {})
    bankrupt_with_debt = loans.get("bankrupt_with_debt", [])
    mem = summary.get("memory", {})
    n_claims = sum(v.get("n_claims", 0) for v in mem.values())
    n_memory_hits = sum(v.get("n_memory_hits", 0) for v in mem.values())
    n_repeats = sum(v.get("n_repeat_deliveries", 0) for v in mem.values())
    n_improved = sum(v.get("n_improved", 0) for v in mem.values())
    spec = specialization(summary)
    return {
        "total_f1": total_f1,
        "total_em": total_em,
        "n_answered": len(deliveries),
        # v4.1 headline: how much of the WORLD's posted demand -- the
        # (job, question) units -- the economy actually absorbed
        "demand_absorbed": len(deliveries) / total_units if total_units else 0.0,
        "job_completion_rate": (summary.get("jobs_closed", 0) / jobs_posted
                                if jobs_posted else 0.0),
        "delegation_rate": delegation_rate(deliveries, summary.get("contracts", [])),
        "accuracy_per_ktok_solving": total_f1 / (solving / 1000) if solving else 0.0,
        "accuracy_per_ktok_all": total_f1 / (all_tok / 1000) if all_tok else 0.0,
        "coordination_overhead": admin / all_tok if all_tok else 0.0,
        "admin_solving_ratio": admin / solving if solving else 0.0,
        "rounds_used": summary["rounds_used"],
        "bankrupt_rate": (len(summary["bankrupt"]) / len(summary["balances"])
                          if summary["balances"] else 0.0),
        "gini_final": gini([max(b, 0) for b in summary["balances"].values()]),
        "n_contracts": summary["n_contracts"],
        "mean_contract_price": sum(prices) / len(prices) if prices else 0.0,
        "n_loans": loans.get("n_proposed", 0),
        "loan_principal_outstanding": loans.get("total_principal_outstanding", 0),
        "interest_paid_total": loans.get("total_interest_paid", 0),
        "bad_debt": sum(debtors.get(a, 0) for a in bankrupt_with_debt),
        "specialization": spec,
        "mean_specialization": (sum(spec.values()) / len(spec)) if spec else 0.0,
        # memory reuse: how often a claim came back with knowledge attached, and
        # how often a second attempt at a known question actually beat it
        "n_claims": n_claims,
        "memory_hit_rate": n_memory_hits / n_claims if n_claims else 0.0,
        "improvement_rate": n_improved / n_repeats if n_repeats else 0.0,
        "answers_in_memory_total": sum(v.get("answers", 0) for v in mem.values()),
    }
