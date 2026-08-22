"""Emit the E2 / step-endpoint LaTeX table bodies from the v11 + pw1 logs.

Steps is the primary endpoint: rho = steps(arm) / steps(B1) on the same
questions, same reader, same budget. F1 is reported as a non-inferiority
check (delta = 0.03), not as the thing being maximised.
"""
import argparse, json, os

DELTA = 0.03
BUILD = {"b1": "---", "graphrag": "488k tok", "hipporag": r"$\sim$0",
         "oracle": r"$\sim$0", "trained": ""}   # trained filled in at runtime
LABEL = {"b1": "B1 flat", "graphrag": "B2 GraphRAG-style",
         "hipporag": "B3 HippoRAG2-style", "oracle": "Exhaustive index",
         "trained": "Ours (trained)"}


def load(run):
    p = f"runs/{run}/test_log.jsonl"
    if not os.path.exists(p):
        return {}
    return {json.loads(l)["qid"]: json.loads(l) for l in open(p)}


def agg(rows, qids):
    n = len(qids)
    return (sum(rows[q]["f1"] for q in qids) / n,
            sum(rows[q]["steps"] for q in qids) / n,
            sum(rows[q]["tokens_in"] + rows[q]["tokens_out"] for q in qids) / n)


def train_cost(run):
    p = f"runs/{run}/train_log.jsonl"
    if not os.path.exists(p):
        return 0
    return sum(json.loads(l)["tokens_in"] + json.loads(l)["tokens_out"]
               for l in open(p))


def seen_qids(run):
    p = f"runs/{run}/train_log.jsonl"
    return {json.loads(l)["qid"] for l in open(p)} if os.path.exists(p) else set()


def table(title, arms, subsets, cost):
    """arms: [(key, run)]; subsets: [(name, [qids])]."""
    print(f"\n%% {title}")
    for sub, qids in subsets:
        d = {k: load(r) for k, r in arms}
        common = [q for q in qids if all(q in d[k] for k, _ in arms if d[k])]
        if not common:
            print(f"%%   {sub}: no data")
            continue
        _, b_steps, _ = agg(d["b1"], common)
        b_f1, _, _ = agg(d["b1"], common)
        print(f"\\multicolumn{{6}}{{l}}{{\\emph{{{sub}}} ($n{{=}}{len(common)}$)}}\\\\")
        for k, _ in arms:
            if not d[k]:
                continue
            f1, st, tok = agg(d[k], common)
            rho = st / b_steps
            ni = "" if k == "b1" else (r"\checkmark" if f1 >= b_f1 - DELTA else r"$\times$")
            build = f"{cost/1e6:.1f}M tok" if k == "trained" else BUILD[k]
            print(f"{LABEL[k]} & {build} & {st:.1f} & {rho:.3f} & "
                  f"{f1:.3f}~{ni} & {tok/1000:.1f}k \\\\")
        print(r"\midrule")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", choices=["kbgym", "pw"], default="kbgym")
    a = ap.parse_args()

    if a.bench == "kbgym":
        main_run, pre, base = "v11_main", "v11", "v11base"
        keys = ["b1", "graphrag", "hipporag", "oracle", "trained"]
    else:
        main_run, pre, base = "pw1_main", "pw1", "pw1"
        keys = ["b1", "hipporag", "oracle", "trained"]

    cost = train_cost(main_run)
    seen = seen_qids(main_run)

    # held-out test split, both budgets
    for M in (15, 8):
        arms = [(k, (f"{pre}_trained_m{M}" if k == "trained"
                     else f"{base}_{k}_m{M}")) for k in keys]
        any_rows = load(arms[0][1])
        if any_rows:
            table(f"{a.bench} test split, M={M}", arms,
                  [(f"budget $M{{=}}{M}$", list(any_rows))], cost)

    # train split, sliced by whether training saw the question
    arms = [(k, (f"{pre}_trained_train" if k == "trained"
                 else f"{base}_{k}_train")) for k in keys]
    rows = load(arms[0][1])
    if rows:
        table(f"{a.bench} train split", arms,
              [("trained-on questions", [q for q in rows if q in seen]),
               ("held-out questions, same split",
                [q for q in rows if q not in seen])], cost)


if __name__ == "__main__":
    main()
