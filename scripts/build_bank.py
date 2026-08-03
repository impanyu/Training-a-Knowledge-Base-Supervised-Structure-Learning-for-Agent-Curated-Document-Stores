"""Build the flat question bank (v4).

v4 drops the hierarchical task trees: a *task* is one question, and the WORLD
posts each question with a QUOTA -- how many times it may be claimed and paid
for. Total system demand is therefore sum(quota) claimable units, and a
question that several agents answer over the run pays each of them (the repeat
that makes knowledge reuse and specialization measurable).

Each question also carries a `topic` id from a light clustering of question
embeddings. Topics are metadata only -- the agents never see them -- and exist
so per-agent specialization stays measurable without a task tree.
"""
import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))


def assign_quotas(questions: list[dict], lo: int, hi: int, rng: random.Random) -> None:
    for q in questions:
        q["quota"] = rng.randint(lo, hi)


def kmeans_topics(vectors: np.ndarray, k: int, rng: random.Random, iters: int = 25) -> list[int]:
    """k-means++ seeding + Lloyd iterations on L2-normalized vectors (so the
    Euclidean objective is equivalent to cosine). Hand-rolled to avoid a
    scikit-learn dependency for what is a one-off metadata pass."""
    v = np.asarray(vectors, dtype=float)
    v = v / np.clip(np.linalg.norm(v, axis=1, keepdims=True), 1e-12, None)
    n = len(v)
    k = min(k, n)

    centers = [v[rng.randrange(n)]]
    for _ in range(k - 1):
        d2 = np.min(((v[:, None, :] - np.array(centers)[None, :, :]) ** 2).sum(-1), axis=1)
        total = d2.sum()
        if total <= 0:
            centers.append(v[rng.randrange(n)])
            continue
        pick = rng.random() * total
        centers.append(v[int(np.searchsorted(np.cumsum(d2), pick))])
    c = np.array(centers)

    labels = np.zeros(n, dtype=int)
    for _ in range(iters):
        d = ((v[:, None, :] - c[None, :, :]) ** 2).sum(-1)
        new = d.argmin(axis=1)
        if (new == labels).all():
            break
        labels = new
        for j in range(k):
            members = v[labels == j]
            if len(members):
                c[j] = members.mean(axis=0)
    return [int(x) for x in labels]


def build_bank(pool: list[dict], n_topics: int, quota_lo: int, quota_hi: int,
               seed: int, embed=None) -> dict:
    rng = random.Random(seed)
    questions = [dict(q) for q in pool]
    assign_quotas(questions, quota_lo, quota_hi, rng)
    if embed is None:
        from build_tasks import embed_texts as embed
    labels = kmeans_topics(embed([q["text"] for q in questions]), n_topics, rng)
    for q, t in zip(questions, labels):
        q["topic"] = f"k{t:02d}"
    return {"questions": questions,
            "total_units": sum(q["quota"] for q in questions),
            "n_topics": len(set(labels))}


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default="data/v4/pool.jsonl")
    ap.add_argument("--out", default="data/v4/bank.json")
    ap.add_argument("--topics", type=int, default=25)
    ap.add_argument("--quota-lo", type=int, default=1)
    ap.add_argument("--quota-hi", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    with open(args.pool) as f:
        pool = [json.loads(line) for line in f]
    bank = build_bank(pool, args.topics, args.quota_lo, args.quota_hi, args.seed)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(bank, f, ensure_ascii=False)

    from collections import Counter
    qs = bank["questions"]
    print(f"bank saved to {args.out}: {len(qs)} questions, "
          f"{bank['total_units']} claimable units, {bank['n_topics']} topics")
    print("quota mix:", dict(sorted(Counter(q["quota"] for q in qs).items())))
    print("difficulty:", dict(Counter(q["difficulty"] for q in qs)))
    print("topic sizes: min %d max %d" % (
        min(Counter(q["topic"] for q in qs).values()),
        max(Counter(q["topic"] for q in qs).values())))
    print("total posted value:", sum(q["price"] * q["quota"] for q in qs))


if __name__ == "__main__":
    main()
