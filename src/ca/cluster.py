"""Deterministic domain clustering: Lloyd's k-means + the qclusters cache.

The N agents' domains are the N k-means clusters of the bank's question
embeddings (same ONNX space the corpus is embedded in). Hand-rolled to avoid
a scikit-learn dependency: seeded init by choice-without-replacement, fixed
iteration cap, ties and empty clusters resolved deterministically. The
runner caches the result at data/<bank-dir>/qclusters_{N}.json so the 1000
questions are embedded once per N.
"""
import json
from pathlib import Path

import numpy as np

MAX_ITER = 50
N_EXEMPLARS = 5


def kmeans(X, k: int, seed: int = 0, max_iter: int = MAX_ITER):
    """(centroids [k,d], assignment [n]) for row vectors X."""
    X = np.asarray(X, dtype=float)
    if not 0 < k <= len(X):
        raise ValueError(f"k={k} needs 1..{len(X)} rows")
    rng = np.random.RandomState(seed)
    centroids = X[rng.choice(len(X), size=k, replace=False)].copy()
    assign = None
    for _ in range(max_iter):
        d = ((X[:, None, :] - centroids[None, :, :]) ** 2).sum(-1)
        new = d.argmin(axis=1)
        if assign is not None and (new == assign).all():
            break
        assign = new
        for j in range(k):
            members = X[assign == j]
            if len(members):                # empty cluster keeps its centroid
                centroids[j] = members.mean(axis=0)
    return centroids, assign


def exemplar_qids(qids: list[str], X, centroids, assign,
                  n: int = N_EXEMPLARS) -> dict[int, list[str]]:
    """Per cluster, the n member qids nearest its centroid -- the questions
    that describe the domain in the owner's system prompt."""
    X = np.asarray(X, dtype=float)
    out = {}
    for j in range(len(centroids)):
        members = [i for i in range(len(qids)) if assign[i] == j]
        members.sort(key=lambda i: float(((X[i] - centroids[j]) ** 2).sum()))
        out[j] = [qids[i] for i in members[:n]]
    return out


def build_cache(path, bank, embedding_function, n_clusters: int,
                seed: int = 0) -> dict:
    """Embed every bank question, cluster, and write the cache. The
    embedding_function must be the same model the corpus was embedded with
    (chroma's default ONNX in production; tests pass the hash stub)."""
    qids = sorted(bank.questions)
    X = np.asarray(embedding_function([bank.questions[q].text for q in qids]),
                   dtype=float)
    centroids, assign = kmeans(X, n_clusters, seed=seed)
    cache = {
        "seed": seed,
        "centroids": [[float(x) for x in c] for c in centroids],
        "assignment": {qid: int(a) for qid, a in zip(qids, assign)},
        "exemplars": {str(j): v for j, v in
                      exemplar_qids(qids, X, centroids, assign).items()},
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(cache, f)
    return cache


def load_cache(path) -> dict:
    with open(path) as f:
        return json.load(f)


def exemplar_texts(cache: dict, bank) -> dict[str, list[str]]:
    """agent_id -> its domain's exemplar question texts (cluster i is owned
    by agent_{i+1})."""
    return {f"agent_{int(j) + 1}": [bank.questions[qid].text for qid in qids]
            for j, qids in cache["exemplars"].items()}
