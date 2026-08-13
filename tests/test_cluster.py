import json

import numpy as np
import pytest
from fixtures import HashEmbedding, demo_bank

from ca.cluster import (build_cache, exemplar_qids, exemplar_texts, kmeans,
                        load_cache)


def blobs():
    """Two well-separated 2D blobs, 4 points each."""
    a = [[0.0, 0.0], [0.1, 0.0], [0.0, 0.1], [0.1, 0.1]]
    b = [[5.0, 5.0], [5.1, 5.0], [5.0, 5.1], [5.1, 5.1]]
    return np.array(a + b)


def test_kmeans_recovers_obvious_blobs():
    centroids, assign = kmeans(blobs(), 2, seed=0)
    assert len(set(assign[:4])) == 1 and len(set(assign[4:])) == 1
    assert assign[0] != assign[4]
    lo, hi = sorted(np.asarray(centroids)[:, 0])
    assert lo == pytest.approx(0.05) and hi == pytest.approx(5.05)


def test_kmeans_is_deterministic_per_seed():
    c1, a1 = kmeans(blobs(), 2, seed=0)
    c2, a2 = kmeans(blobs(), 2, seed=0)
    assert (np.asarray(c1) == np.asarray(c2)).all() and (a1 == a2).all()


def test_kmeans_rejects_a_bad_k():
    with pytest.raises(ValueError):
        kmeans(blobs(), 0)
    with pytest.raises(ValueError):
        kmeans(blobs(), 9)


def test_exemplars_are_the_members_nearest_the_centroid_in_order():
    X = np.array([[0.0], [1.0], [2.0], [10.0]])
    centroids = np.array([[0.5], [10.0]])
    assign = np.array([0, 0, 0, 1])
    ex = exemplar_qids(["a", "b", "c", "d"], X, centroids, assign, n=2)
    assert ex == {0: ["a", "b"], 1: ["d"]}     # a and b tie-broken by distance


def test_build_cache_writes_and_load_cache_reads(tmp_path):
    path = tmp_path / "qclusters_2.json"
    cache = build_cache(path, demo_bank(), HashEmbedding(), 2)
    assert load_cache(path) == cache
    assert cache["seed"] == 0
    assert set(cache["assignment"]) == {f"q{i:04d}" for i in range(1, 9)}
    assert set(cache["assignment"].values()) == {0, 1}
    assert len(cache["centroids"]) == 2
    # the demo bank's two lexical families come out as the two domains
    france = {cache["assignment"][q] for q in ("q0001", "q0002", "q0003", "q0004")}
    sums = {cache["assignment"][q] for q in ("q0005", "q0006", "q0007", "q0008")}
    assert len(france) == 1 and len(sums) == 1 and france != sums
    json.loads(path.read_text())               # plain JSON on disk


def test_exemplars_cap_at_five_and_map_to_agent_texts(tmp_path):
    bank = demo_bank()
    cache = build_cache(tmp_path / "qclusters_1.json", bank, HashEmbedding(), 1)
    assert len(cache["exemplars"]["0"]) == 5   # 8 members, capped at 5
    texts = exemplar_texts(cache, bank)
    assert set(texts) == {"agent_1"}
    assert all(t in {q.text for q in bank.questions.values()}
               for t in texts["agent_1"])


def test_cluster_i_belongs_to_agent_i_plus_one(tmp_path):
    bank = demo_bank()
    cache = build_cache(tmp_path / "qclusters_2.json", bank, HashEmbedding(), 2)
    texts = exemplar_texts(cache, bank)
    assert set(texts) == {"agent_1", "agent_2"}
    for j, qids in cache["exemplars"].items():
        assert texts[f"agent_{int(j) + 1}"] == [bank.questions[q].text for q in qids]
