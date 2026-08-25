"""Stratified key-overlap probe for the PhantomWiki arm.

PW ships questions, not instance pools, so the probe is drawn from the 330
official questions. Keys are detected as person names; only questions
carrying exactly one detected key, in templates where BOTH overlap buckets
are populated, participate. Groups (equal per-template quotas, so the three
groups share one template mix):

  exact  -- training questions themselves
  share1 -- unseen question whose key appeared in some training question
  share0 -- unseen question whose key did not
"""
import argparse, json, random, re
from pathlib import Path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--universe", default="data/pw1/universe.json")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--arms", nargs="+", required=True)
    ap.add_argument("--outdir", default="data")
    a = ap.parse_args()

    u = json.loads(Path(a.universe).read_text())
    qs = {q["qid"]: q for q in u["questions"]}
    train = list(u["splits"]["train"])
    name = re.compile(r"\b[A-Z][a-z]+(?: [A-Z][a-z]+)+\b")
    terms = lambda t: set(name.findall(t))

    keys = set()
    for q in train:
        keys |= terms(qs[q]["text"])
    print(f"{len(train)} training questions carry {len(keys)} distinct name keys")

    tr_pool, s1_pool, s0_pool = {}, {}, {}
    trset = set(train)
    for q in u["questions"]:
        tm = terms(q["text"])
        if len(tm) != 1:
            continue
        t = q["template"]
        if q["qid"] in trset:
            tr_pool.setdefault(t, []).append(q)
        elif tm & keys:
            s1_pool.setdefault(t, []).append(q)
        else:
            s0_pool.setdefault(t, []).append(q)

    quota = {t: min(len(tr_pool.get(t, [])), len(s1_pool.get(t, [])),
                    len(s0_pool.get(t, [])))
             for t in tr_pool}
    quota = {t: n for t, n in quota.items() if n > 0}
    print("per-template quota:", dict(sorted(quota.items())))

    pick = random.Random(a.seed)
    splits, questions, n = {}, [], 0
    for gname, pool in [("exact", tr_pool), ("share1", s1_pool),
                        ("share0", s0_pool)]:
        splits[gname] = []
        for t in sorted(quota):
            rows = pool[t][:]
            pick.shuffle(rows)
            for q in rows[:quota[t]]:
                n += 1
                qid = f"p{n:04d}"
                questions.append({**q, "qid": qid})
                splits[gname].append(qid)
        print(f"  {gname}: {len(splits[gname])}")

    for spec in a.arms:
        aname, path = spec.split("=", 1)
        v = json.loads(Path(path).read_text())
        v["questions"] = questions
        v["splits"] = splits
        out = Path(a.outdir) / f"pwgrad_{aname}"
        out.mkdir(parents=True, exist_ok=True)
        (out / "universe.json").write_text(json.dumps(v, ensure_ascii=False))
        print(f"  wrote {out/'universe.json'} ({len(v['nodes'])} nodes)")

if __name__ == "__main__":
    main()
