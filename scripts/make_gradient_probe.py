"""Build the controlled key-overlap probe.

The generalization question is how far the structure built during training
reaches. The universe's own test splits cannot answer it: they were drawn
disjoint from train, so almost every test question shares zero keys with
the training set. This script instead samples from the *full* instance
pool of the trained multi-slot templates and groups questions by how many
of their keys the training set touched:

  share 2 -- both keys appeared in some training question
  share 1 -- exactly one did
  share 0 -- neither did

Group "exact" (the training questions themselves) is already evaluated by
the train-split runs. Gold answers and support sets come from the same
regenerated universe, so every arm answers them against its own store.
"""
import argparse, json, random, re
from pathlib import Path

from kb import build as B

TWO_SLOT = ("qc6_job_city", "qc6_job_hobby", "qc7_common_friend",
            "qc8_friend_job", "qc9_older_pair", "qc10_spouse_job_city")


def term_fn(vocab):
    words = (set(vocab.get("jobs", [])) | set(vocab.get("hobbies", []))
             | set(vocab.get("cities", [])))
    name = re.compile(r"\b[A-Z][a-z]+ [A-Z][a-z]+\b")
    def terms(text):
        return (set(name.findall(text)) |
                {w for w in words if re.search(rf"\b{re.escape(w)}\b", text, re.I)})
    return terms


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--universe", default="data/v10L/universe.json")
    ap.add_argument("--train-log", default="runs/v11_main/train_log.jsonl")
    ap.add_argument("--per-group", type=int, default=100)
    ap.add_argument("--stratified", action="store_true",
                    help="equal per-template quotas in every group, so the "
                         "four groups share one template mix; quota per "
                         "template = number of trained questions of that "
                         "template (the binding constraint)")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--arms", nargs="+", required=True,
                    help="name=path/to/universe.json for each arm's store")
    ap.add_argument("--outdir", default="data")
    a = ap.parse_args()

    base = json.loads(Path(a.universe).read_text())
    meta, terms = base["meta"], term_fn(base["vocab"])
    Q = {q["qid"]: q for q in base["questions"]}

    trained = list(dict.fromkeys(
        json.loads(l)["qid"] for l in open(a.train_log)))
    keys, seen_text = set(), set()
    for q in trained:
        keys |= terms(Q[q]["text"])
        seen_text.add(Q[q]["text"])
    print(f"{len(trained)} training questions touched {len(keys)} distinct keys")

    rng = random.Random(meta["seed"])
    people = B._people(rng, meta["n_people"])
    B._families(rng, people)
    extras = B._distractors(rng, people, meta["distractors"])
    B._birthdates(rng, people, extras)
    _, idx = B._statement_nodes(people, extras)

    buckets = {0: [], 1: [], 2: []}
    for t in TWO_SLOT:
        for row in B._instances(t, people, extras, idx):
            if row["text"] in seen_text:
                continue
            tm = terms(row["text"])
            if len(tm) != 2:
                continue
            buckets[len(tm & keys)].append((t, row))

    pick = random.Random(a.seed)
    splits, questions, n = {}, [], 0

    def add(name, src_rows):
        nonlocal n
        splits.setdefault(name, [])
        for item in src_rows:
            n += 1
            qid = f"g{n:04d}"
            if isinstance(item, dict):          # a training question, verbatim
                questions.append({**item, "qid": qid})
            else:                               # (template, instance row)
                t, row = item
                questions.append({
                    "qid": qid, "template": t, "category": B.TEMPLATES[t][0],
                    "hops": B.TEMPLATES[t][1], "text": row["text"],
                    "golds": row["golds"], "support": row["support"],
                    "unanswerable": bool(row.get("unanswerable", False)),
                    "eval_flavor": None})
            splits[name].append(qid)

    if a.stratified:
        # One template mix for all four groups. The quota for template t is
        # the number of trained questions of t (every bucket pool is far
        # larger), so "exact" is ALL trained two-slot questions and each
        # share group mirrors its composition exactly.
        ex = {}
        for qid in trained:
            t = Q[qid]["template"]
            if t in TWO_SLOT:
                ex.setdefault(t, []).append(Q[qid])
        by_tb = {}
        for k in (2, 1, 0):
            for t, row in buckets[k]:
                by_tb.setdefault((t, k), []).append((t, row))
        quota = {t: min([len(ex[t])] + [len(by_tb.get((t, k), []))
                                        for k in (2, 1, 0)])
                 for t in ex}
        print("  per-template quota:",
              {t: q for t, q in sorted(quota.items())})
        add("exact", [q for t in sorted(quota) for q in ex[t][:quota[t]]])
        for k in (2, 1, 0):
            rows = []
            for t in sorted(quota):
                pool = by_tb.get((t, k), [])
                pick.shuffle(pool)
                rows += pool[:quota[t]]
            add(f"share{k}", rows)
        for name in splits:
            print(f"  {name}: {len(splits[name])} questions")
    else:
        # Group "exact": the training questions themselves, the anchor of
        # the gradient. Same size as the other groups so the four points
        # are directly comparable.
        anchor = list(trained)
        pick.shuffle(anchor)
        add("exact", [Q[qid] for qid in anchor[:a.per_group]])
        print(f"  exact: {len(splits['exact'])} questions (the training set)")
        for k in (2, 1, 0):
            pool = buckets[k]
            pick.shuffle(pool)
            add(f"share{k}", pool[:a.per_group])
            print(f"  share{k}: {len(splits[f'share{k}'])} questions "
                  f"(pool {len(pool)})")

    for spec in a.arms:
        name, path = spec.split("=", 1)
        u = json.loads(Path(path).read_text())
        u["questions"] = questions
        u["splits"] = splits
        out = Path(a.outdir) / f"grad_{name}"
        out.mkdir(parents=True, exist_ok=True)
        (out / "universe.json").write_text(json.dumps(u, ensure_ascii=False))
        print(f"  wrote {out/'universe.json'}  ({len(u['nodes'])} nodes)")


if __name__ == "__main__":
    main()
