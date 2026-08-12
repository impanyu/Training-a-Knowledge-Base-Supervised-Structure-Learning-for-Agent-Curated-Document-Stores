"""Build the question bank (v5): the pool, topic-labeled.

A task is one question; the bank IS the posted demand. Each question carries a
`topic` id from a light clustering of question embeddings -- metadata for the
specialization metric only, never shown to agents. Run prepare_data.py first.
"""
import argparse
import json
import random
from pathlib import Path

import numpy as np


def embed_texts(texts: list[str]) -> np.ndarray:
    """Chroma's local ONNX embedder -- the same model the corpus is embedded
    with, so topic clusters live in the corpus's own semantic space."""
    from chromadb.utils.embedding_functions import DefaultEmbeddingFunction
    ef = DefaultEmbeddingFunction()
    return np.asarray(ef(texts), dtype=float)


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


def build_bank(pool: list[dict], n_topics: int, seed: int, embed=None) -> dict:
    rng = random.Random(seed)
    questions = [dict(q) for q in pool]
    if embed is None:
        embed = embed_texts
    labels = kmeans_topics(embed([q["text"] for q in questions]), n_topics, rng)
    for q, t in zip(questions, labels):
        q["topic"] = f"k{t:02d}"
    return {"questions": questions, "n_topics": len(set(labels))}


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default="data/v5/pool.jsonl")
    ap.add_argument("--out", default="data/v5/bank.json")
    ap.add_argument("--topics", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    with open(args.pool) as f:
        pool = [json.loads(line) for line in f]
    bank = build_bank(pool, args.topics, args.seed)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(bank, f, ensure_ascii=False)

    from collections import Counter
    qs = bank["questions"]
    print(f"bank saved to {args.out}: {len(qs)} questions, {bank['n_topics']} topics")
    print("difficulty:", dict(Counter(q["difficulty"] for q in qs)))
    print("topic sizes:", sorted(Counter(q["topic"] for q in qs).values()))
    print("total question value:", sum(q["price"] for q in qs))


if __name__ == "__main__":
    main()
