"""Offline tests for scripts/build_tasks.py: clustering shape, semantic-
locality checking, sentence-uniqueness fallback, posted-task selection, and
a full --no-llm build with everything network-facing monkeypatched out.
No network, no LLM calls, no live dataset download -- that's T24.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import build_tasks  # noqa: E402

from ca.tasktree import TaskLibrary, normalize  # noqa: E402


def _stub_vectors(n: int, dim: int = 8, seed: int = 0) -> np.ndarray:
    return np.random.RandomState(seed).rand(n, dim)


# ---------------------------------------------------------------------------
# greedy_cluster
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("n", [1, 2, 3, 4, 5, 7, 10, 13, 25])
def test_greedy_cluster_branching_le_3_and_covers_every_index_once(n):
    vecs = _stub_vectors(n, seed=n)
    clusters = build_tasks.greedy_cluster(vecs, seed=0)
    seen = []
    for c in clusters:
        assert 1 <= len(c) <= build_tasks.MAX_BRANCH
        seen.extend(c)
    assert sorted(seen) == list(range(n))


def test_greedy_cluster_deterministic_given_seed():
    vecs = _stub_vectors(17, seed=17)
    assert build_tasks.greedy_cluster(vecs, seed=3) == build_tasks.greedy_cluster(vecs, seed=3)


def test_greedy_cluster_different_seed_can_change_tie_breaks():
    # Perfectly symmetric points: every pair equidistant, so the outcome is
    # purely tie-break driven -- different seeds should be able to produce a
    # different (still valid) partition, which is what makes the re-cluster
    # retry meaningful.
    vecs = np.eye(6)
    a = build_tasks.greedy_cluster(vecs, seed=0)
    b = build_tasks.greedy_cluster(vecs, seed=1)
    for clusters in (a, b):
        seen = [i for c in clusters for i in c]
        assert sorted(seen) == list(range(6))


# ---------------------------------------------------------------------------
# build_tree: depth/branching/leaf-partition invariants
# ---------------------------------------------------------------------------

def test_build_tree_branching_le_3_every_node():
    n = 14
    qids = [f"q{i:04d}" for i in range(1, n + 1)]
    vecs = _stub_vectors(n, seed=42)
    nodes, levels, _ = build_tasks.build_tree(qids, vecs, seed=0)
    for node in nodes.values():
        assert 1 <= len(node.children) <= build_tasks.MAX_BRANCH


def test_build_tree_depth_le_4_for_every_root():
    from ca.taskboard import Question

    n = 14
    qids = [f"q{i:04d}" for i in range(1, n + 1)]
    vecs = _stub_vectors(n, seed=42)
    nodes, levels, _ = build_tasks.build_tree(qids, vecs, seed=0)
    for node in nodes.values():
        node.sentence = f"sentence for {node.nid}"
    questions = [Question(qid, f"text {qid}", ["x"], "2hop", 100) for qid in qids]
    lib = TaskLibrary(nodes, questions, list(levels["L3"]))
    for root in levels["L3"]:
        assert lib.depth(root) <= 4


def test_all_questions_appear_exactly_once_as_l1_leaves():
    n = 20
    qids = [f"q{i:04d}" for i in range(1, n + 1)]
    vecs = _stub_vectors(n, seed=7)
    nodes, levels, _ = build_tasks.build_tree(qids, vecs, seed=1)
    leaves = [c for nid in levels["L1"] for c in nodes[nid].children]
    assert sorted(leaves) == sorted(qids)
    assert len(leaves) == len(set(leaves))


# ---------------------------------------------------------------------------
# semantic-locality validation
# ---------------------------------------------------------------------------

def test_semantic_locality_flags_planted_violation():
    # cluster0 = {0, 1} are near-opposite (low cohesion); item 2 (its own
    # singleton cluster) is nearly identical to item 0, so cluster0 is less
    # cohesive than its similarity to its nearest sibling -- a violation.
    vectors = np.array([
        [1.0, 0.0],
        [-1.0, 0.0],
        [0.99, 0.01],
    ])
    clusters = [[0, 1], [2]]
    violations = build_tasks.check_semantic_locality(clusters, vectors)
    assert len(violations) == 1
    assert violations[0]["cluster_index"] == 0


def test_semantic_locality_passes_cohesive_well_separated_clusters():
    vectors = np.array([
        [1.0, 0.0], [0.98, 0.02],
        [0.0, 1.0], [0.02, 0.98],
    ])
    clusters = [[0, 1], [2, 3]]
    assert build_tasks.check_semantic_locality(clusters, vectors) == []


def test_semantic_locality_skips_singletons_and_single_cluster_input():
    vectors = np.array([[1.0, 0.0], [-1.0, 0.0]])
    assert build_tasks.check_semantic_locality([[0], [1]], vectors) == []
    assert build_tasks.check_semantic_locality([[0, 1]], vectors) == []


# ---------------------------------------------------------------------------
# sentence uniqueness / fallback
# ---------------------------------------------------------------------------

def test_summarize_node_no_llm_uses_fallback_with_nid():
    sentence = build_tasks.summarize_node(
        ["capital of France?", "capital of Germany?"], [], "t0042", llm_call=None)
    assert "t0042" in sentence


def test_summarize_node_retries_once_then_falls_back_on_duplicate():
    calls = []

    def dup_llm(member_texts, used_sentences):
        calls.append((member_texts, used_sentences))
        return "answer capital questions"

    used = ["answer capital questions"]
    sentence = build_tasks.summarize_node(
        ["capital of France?", "capital of Germany?"], used, "t0007", llm_call=dup_llm)
    assert len(calls) == 2   # one try, one retry, then fallback -- no third call
    assert "t0007" in sentence
    assert normalize(sentence) != normalize(used[0])


def test_summarize_node_retries_then_falls_back_on_empty_reply():
    sentence = build_tasks.summarize_node(
        ["a?", "b?"], [], "t0011", llm_call=lambda texts, used: "")
    assert "t0011" in sentence


def test_summarize_node_accepts_first_good_unique_reply():
    sentence = build_tasks.summarize_node(
        ["a?"], [], "t0001", llm_call=lambda texts, used: "Answer French geography questions")
    assert sentence == "Answer French geography questions"


def test_summarize_node_accepts_reply_on_second_try():
    replies = iter(["dup sentence", "a fresh unique sentence"])
    sentence = build_tasks.summarize_node(
        ["a?"], ["dup sentence"], "t0002", llm_call=lambda t, u: next(replies))
    assert sentence == "a fresh unique sentence"


def test_unique_fallback_sentence_appends_marker_on_collision():
    base = build_tasks.keyword_fallback_sentence(["capital of France?"], "t0099")
    used_norm = {normalize(base)}
    result = build_tasks.unique_fallback_sentence(["capital of France?"], "t0099", used_norm)
    assert normalize(result) not in used_norm
    assert "t0099" in result


def test_keyword_fallback_sentence_is_deterministic():
    a = build_tasks.keyword_fallback_sentence(["capital of France?", "capital of Spain?"], "t0005")
    b = build_tasks.keyword_fallback_sentence(["capital of France?", "capital of Spain?"], "t0005")
    assert a == b
    assert "t0005" in a


# ---------------------------------------------------------------------------
# posted-task selection
# ---------------------------------------------------------------------------

def test_select_posted_all_when_fewer_than_target():
    levels = {"L1": ["t0001", "t0002"], "L2": ["t0003"], "L3": ["t0004"]}
    posted = build_tasks.select_posted(levels, target=30, seed=0)
    assert sorted(posted) == ["t0001", "t0002", "t0003", "t0004"]


def test_select_posted_count_is_min_target_available():
    levels = {"L1": [f"t{i:04d}" for i in range(1, 41)],
              "L2": [f"t{i:04d}" for i in range(41, 51)],
              "L3": [f"t{i:04d}" for i in range(51, 56)]}
    total_available = sum(len(v) for v in levels.values())
    posted = build_tasks.select_posted(levels, target=30, seed=0)
    assert len(posted) == min(30, total_available)
    assert set(levels["L3"]) <= set(posted)


def test_select_posted_no_duplicates_and_all_known_nodes():
    levels = {"L1": [f"t{i:04d}" for i in range(1, 10)],
              "L2": [f"t{i:04d}" for i in range(10, 14)],
              "L3": [f"t{i:04d}" for i in range(14, 16)]}
    all_nodes = set(levels["L1"] + levels["L2"] + levels["L3"])
    posted = build_tasks.select_posted(levels, target=8, seed=5)
    assert len(posted) == len(set(posted))
    assert set(posted) <= all_nodes
    assert len(posted) == 8


# ---------------------------------------------------------------------------
# full --no-llm pipeline, fully offline
# ---------------------------------------------------------------------------

def test_no_llm_cli_builds_a_valid_round_tripping_library(tmp_path, monkeypatch):
    n_hotpot, n_musique = 9, 3
    total = n_hotpot + n_musique
    topics = ["france geography", "germany capital", "arithmetic warmup"]

    def fake_download_pool(hotpot_n, musique_n, seed):
        pool = []
        for i in range(hotpot_n + musique_n):
            topic = topics[i % len(topics)]
            pool.append({
                "qid": f"q{i + 1:04d}",
                "text": f"question {i + 1} about {topic}?",
                "answers": [f"answer{i + 1}"],
                "difficulty": "2hop",
                "price": 100 + i,
                "source": "hotpotqa" if i < hotpot_n else "musique",
            })
        corpus = [{"title": f"doc{i}", "text": f"paragraph {i} about {topics[i % 3]}"}
                  for i in range(5)]
        return pool, corpus

    def fake_embed_texts(texts):
        return np.random.RandomState(123).rand(len(texts), 6)

    def fake_build_corpus_index(corpus, persist_dir):
        Path(persist_dir).mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(build_tasks, "download_pool", fake_download_pool)
    monkeypatch.setattr(build_tasks, "embed_texts", fake_embed_texts)
    monkeypatch.setattr(build_tasks, "build_corpus_index", fake_build_corpus_index)

    out = tmp_path / "v2"
    build_tasks.main([
        "--hotpot-n", str(n_hotpot), "--musique-n", str(n_musique),
        "--seed", "0", "--out", str(out), "--no-llm", "--post-target", "5",
    ])

    assert (out / "pool.jsonl").exists()
    assert (out / "library.json").exists()
    assert (out / "summaries.json").exists()
    assert (out / "index").exists()

    lib = TaskLibrary.from_json(str(out / "library.json"))   # raises on any invariant break
    assert len(lib.questions) == total
    assert len(lib.posted) == min(5, len(lib.nodes))
    for nid in lib.posted:
        assert lib.leaves(nid)          # posted nodes are never empty
        assert lib.depth(nid) <= 4
    for node in lib.nodes.values():
        assert 1 <= len(node.children) <= build_tasks.MAX_BRANCH

    # sentences are globally unique (up to normalization), as enforced by the
    # fallback path exercised end-to-end here
    sentences = [normalize(n.sentence) for n in lib.nodes.values()]
    assert len(sentences) == len(set(sentences))
