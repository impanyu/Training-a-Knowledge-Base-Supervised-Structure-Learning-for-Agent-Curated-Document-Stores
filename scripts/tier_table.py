"""Evaluation tables for the paper, with steps as the primary endpoint.

The evaluation set is the two question groups the protocol is about:

  SEEN  -- the question was used in training
  NEAR  -- near transfer: same template as a training question, and sharing
           at least one term with the training set (so the store has seen
           the relation type and at least one of the keys involved)
  FAR   -- everything else, reported only as a contrast

rho = steps(arm) / steps(B1) on the same questions, same reader, same
budget. F1 is a non-inferiority check at delta = 0.03, not the endpoint.
Bootstrap 95% CIs on rho (20k resamples, paired by question).
"""
import argparse, json, os, random, re

DELTA, B = 0.03, 20000
ARMS = [("b1", "B1 flat"), ("graphrag", "B2 GraphRAG-style"),
        ("hipporag", "B3 HippoRAG2-style"),
        ("trained", "Ours (trained)")]
BUILD = {"b1": "---", "graphrag": "488k tok", "hipporag": r"$\sim$0"}


def load(run):
    p = f"runs/{run}/test_log.jsonl"
    return ({json.loads(l)["qid"]: json.loads(l) for l in open(p)}
            if os.path.exists(p) else {})


def question_terms(universe):
    uni = json.load(open(universe))
    v = uni.get("vocab", {})
    vocab = set(v.get("jobs", [])) | set(v.get("hobbies", [])) | set(v.get("cities", []))
    name = re.compile(r"\b[A-Z][a-z]+ [A-Z][a-z]+\b")
    out = {}
    for q in uni["questions"]:
        t = q["text"]
        out[q["qid"]] = (q["template"],
                         set(name.findall(t)) |
                         {w for w in vocab if re.search(rf"\b{re.escape(w)}\b", t, re.I)})
    return out


def groups(universe, main_run, train_rows, test_rows):
    info = question_terms(universe)
    seen = {json.loads(l)["qid"] for l in open(f"runs/{main_run}/train_log.jsonl")}
    tmpl = {info[q][0] for q in seen if q in info}
    terms = {q: info[q][1] for q in seen if q in info}
    def near(q):
        t, tm = info[q]
        return t in tmpl and any(tm & s for s in terms.values())
    return [("seen in training", "train", [q for q in train_rows if q in seen]),
            ("near transfer", "test", [q for q in test_rows if near(q)]),
            ("far transfer (contrast)", "test",
             [q for q in test_rows if not near(q)])]


def ci(deltas, base):
    n = len(deltas)
    s = sorted(sum(random.choice(deltas) for _ in range(n)) / n for _ in range(B))
    return (base + s[int(.025 * B)]) / base, (base + s[int(.975 * B)]) / base


def emit(name, qids, runs, cost, latex):
    random.seed(0)
    d = {k: load(runs[k]) for k, _ in ARMS}
    qids = [q for q in qids if all(q in d[k] for k, _ in ARMS if d[k])]
    if not qids:
        return
    n = len(qids)
    b_steps = sum(d["b1"][q]["steps"] for q in qids) / n
    b_f1 = sum(d["b1"][q]["f1"] for q in qids) / n
    if latex:
        print(f"\\multicolumn{{7}}{{l}}{{\\emph{{{name}}} ($n{{=}}{n}$)}}\\\\")
    else:
        print(f"\n{name}   n={n}")
        print(f"  {'arm':<19}{'steps':>7}{'rho':>7}{'95% CI':>16}{'F1':>7}")
    for k, label in ARMS:
        if not d[k]:
            if not latex:
                print(f"  {label:<19}{'--':>7}{'pending':>7}")
            continue
        x = d[k]
        st = sum(x[q]["steps"] for q in qids) / n
        f1 = sum(x[q]["f1"] for q in qids) / n
        tok = sum(x[q]["tokens_in"] + x[q]["tokens_out"] for q in qids) / n
        rho = st / b_steps
        lo, hi = ((1.0, 1.0) if k == "b1" else
                  ci([x[q]["steps"] - d["b1"][q]["steps"] for q in qids], b_steps))
        ni = "" if k == "b1" else (r"\checkmark" if f1 >= b_f1 - DELTA else r"$\times$")
        if latex:
            build = f"{cost/1e6:.1f}M tok" if k == "trained" else BUILD[k]
            star = r"$^{*}$" if (k != "b1" and hi < 1.0) else ""
            print(f"{label} & {build} & {st:.1f} & {rho:.3f}{star} & "
                  f"[{lo:.2f}, {hi:.2f}] & {f1:.3f}~{ni} & {tok/1000:.1f}k \\\\")
        else:
            print(f"  {label:<19}{st:>7.1f}{rho:>7.3f}   [{lo:.2f}, {hi:.2f}]"
                  f"{f1:>8.3f}")
    if latex:
        print(r"\midrule")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", choices=["kbgym", "pw"], default="kbgym")
    ap.add_argument("--latex", action="store_true")
    a = ap.parse_args()
    if a.bench == "kbgym":
        uni, main_run = "data/v10L/universe.json", "v11_main"
        R = {"train": {k: ("v11_trained_train" if k == "trained"
                           else f"v11base_{k}_train") for k, _ in ARMS},
             "test": {k: ("v11_trained_m15" if k == "trained"
                          else f"v11base_{k}_m15") for k, _ in ARMS}}
    else:
        uni, main_run = "data/pw1/universe.json", "pw1_main"
        R = {"train": {k: f"pw1_{k}_train" for k, _ in ARMS},
             "test": {k: f"pw1_{k}_m15" for k, _ in ARMS}}
    cost = sum(json.loads(l)["tokens_in"] + json.loads(l)["tokens_out"]
               for l in open(f"runs/{main_run}/train_log.jsonl"))
    for name, which, qids in groups(uni, main_run,
                                    load(R["train"]["b1"]), load(R["test"]["b1"])):
        emit(name, qids, R[which], cost, a.latex)


if __name__ == "__main__":
    main()
