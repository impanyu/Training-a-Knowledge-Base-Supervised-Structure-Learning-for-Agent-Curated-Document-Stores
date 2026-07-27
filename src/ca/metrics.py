"""Headline and auxiliary metrics computed from a run summary."""


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


def compute_metrics(summary: dict) -> dict:
    qs = summary["questions"]
    total_f1 = sum(q["score"] for q in qs)
    total_em = sum(q["em"] for q in qs)
    n_answered = sum(1 for q in qs if q["status"] == "closed")
    billable = sum(t["billable"] for t in summary["tokens"].values())
    free = sum(t["free"] for t in summary["tokens"].values())
    all_tok = billable + free
    prices = summary.get("contract_prices", [])
    return {
        "total_f1": total_f1,
        "total_em": total_em,
        "n_answered": n_answered,
        "accuracy_per_ktok_billable": total_f1 / (billable / 1000) if billable else 0.0,
        "accuracy_per_ktok_all": total_f1 / (all_tok / 1000) if all_tok else 0.0,
        "coordination_overhead": free / all_tok if all_tok else 0.0,
        "rounds_used": summary["rounds_used"],
        "bankrupt_rate": (len(summary["bankrupt"]) / len(summary["balances"])
                          if summary["balances"] else 0.0),
        "gini_final": gini([max(b, 0) for b in summary["balances"].values()]),
        "n_contracts": summary["n_contracts"],
        "mean_contract_price": sum(prices) / len(prices) if prices else 0.0,
    }
