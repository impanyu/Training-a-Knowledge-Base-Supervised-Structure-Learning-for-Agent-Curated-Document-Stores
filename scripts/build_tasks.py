"""Task library builder (T21).

Pipeline: download a pool of hotpot+musique questions and their paragraph
corpus (reusing prepare_data's logic) -> embed question texts -> bottom-up
greedy clustering into a 3-level tree with branching <= 3 (leaves are
questions, so depth <= 4 including root) -> one-sentence LLM summary per node
(cached, globally unique) -> select ~30 posted tasks -> write
out/{pool.jsonl, library.json, summaries.json, index/}.

Structured so the clustering/uniqueness/fallback pieces are importable and
unit-testable with stub embeddings/summarizers, with no network access:
`python -m pytest tests/test_build_tasks.py`. The live download + LLM build
is a separate step (T24), triggered only via `if __name__ == "__main__"`.
"""
import argparse
import functools
import itertools
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from ca.taskboard import Question              # noqa: E402
from ca.tasktree import TaskLibrary, TaskNode, normalize  # noqa: E402

MAX_BRANCH = 3
LEVEL_NAMES = ("L1", "L2", "L3")   # L3 nodes are the roots

_STOPWORDS = {
    "the", "a", "an", "and", "or", "of", "in", "on", "to", "is", "are", "was",
    "were", "what", "which", "who", "whom", "whose", "where", "when", "how",
    "does", "did", "do", "for", "with", "by", "at", "from", "this", "that",
    "these", "those", "it", "its", "as", "be", "been", "has", "have", "had",
    "he", "she", "they", "you", "i", "we", "many", "much", "also", "into",
}


# ---------------------------------------------------------------------------
# Clustering (pure numpy, deterministic given seed)
# ---------------------------------------------------------------------------

def cosine_sim_matrix(vectors) -> np.ndarray:
    v = np.asarray(vectors, dtype=float)
    norms = np.linalg.norm(v, axis=1, keepdims=True)
    norms[norms == 0] = 1e-9
    u = v / norms
    return u @ u.T


def greedy_cluster(vectors, seed: int = 0, max_size: int = MAX_BRANCH) -> list[list[int]]:
    """Greedy agglomerative clustering: repeatedly merge the two closest
    items/clusters by average-linkage cosine similarity, never letting a
    cluster exceed `max_size` members. Deterministic given seed (seed only
    breaks ties, reproducibly but differently across attempts, so a
    re-cluster retry can escape a bad local tie-break). Returns a partition
    of range(len(vectors)) as a list of sorted index lists; every input index
    appears in exactly one cluster."""
    vectors = np.asarray(vectors, dtype=float)
    n = len(vectors)
    if n == 0:
        return []
    if n == 1:
        return [[0]]
    sim = cosine_sim_matrix(vectors)
    rng = random.Random(seed)
    order = list(range(n))
    rng.shuffle(order)
    rank = {idx: r for r, idx in enumerate(order)}

    clusters: dict[int, list[int]] = {i: [i] for i in range(n)}
    active = list(range(n))
    next_id = n

    def avg_linkage(a: int, b: int) -> float:
        A, B = clusters[a], clusters[b]
        return float(sim[np.ix_(A, B)].mean())

    while True:
        best = None   # (sort_key, ca, cb)
        for i in range(len(active)):
            for j in range(i + 1, len(active)):
                ca, cb = active[i], active[j]
                if len(clusters[ca]) + len(clusters[cb]) > max_size:
                    continue
                s = avg_linkage(ca, cb)
                tie = tuple(sorted(rank[m] for m in clusters[ca] + clusters[cb]))
                key = (-s, tie)
                if best is None or key < best[0]:
                    best = (key, ca, cb)
        if best is None:
            break
        _, ca, cb = best
        merged = clusters[ca] + clusters[cb]
        next_active = [c for c in active if c not in (ca, cb)]
        next_active.append(next_id)
        del clusters[ca]
        del clusters[cb]
        clusters[next_id] = merged
        active = next_active
        next_id += 1

    return [sorted(clusters[c]) for c in active]


def _mean_pairwise(sim: np.ndarray, a: list[int], b: list[int], exclude_diag: bool) -> float:
    vals = []
    for i in a:
        for j in b:
            if exclude_diag and i == j:
                continue
            vals.append(sim[i, j])
    return sum(vals) / len(vals) if vals else 0.0


def check_semantic_locality(clusters: list[list[int]], vectors) -> list[dict]:
    """For every cluster with >=2 members (an 'internal node with siblings'
    needs at least one other cluster to compare against, and needs >=2
    members to have an intra-cluster cohesion to speak of): mean pairwise
    cosine sim within the cluster must exceed the mean cosine sim to its
    nearest sibling cluster's members. Returns one violation record per
    cluster that fails this; singletons and single-cluster inputs (nothing
    to compare against) are skipped, not flagged."""
    if len(clusters) < 2:
        return []
    sim = cosine_sim_matrix(vectors)
    violations = []
    for idx, cluster in enumerate(clusters):
        if len(cluster) < 2:
            continue
        intra = _mean_pairwise(sim, cluster, cluster, exclude_diag=True)
        best_inter, best_other = None, None
        for jdx, other in enumerate(clusters):
            if jdx == idx:
                continue
            inter = _mean_pairwise(sim, cluster, other, exclude_diag=False)
            if best_inter is None or inter > best_inter:
                best_inter, best_other = inter, jdx
        if best_inter is not None and intra <= best_inter:
            violations.append({
                "cluster_index": idx, "members": list(cluster),
                "intra_sim": intra, "nearest_sibling_index": best_other,
                "sibling_sim": best_inter,
            })
    return violations


def cluster_level_with_retry(vectors, seed: int, level_name: str,
                              max_size: int = MAX_BRANCH) -> tuple[list[list[int]], list[dict]]:
    """One clustering level with the spec's re-cluster-once-on-violation
    policy: cluster, validate; if any violations, retry once with a
    different merge-order seed and keep whichever attempt has fewer
    violations; any violations still standing are returned (and logged) as
    warnings rather than failing the build."""
    clusters = greedy_cluster(vectors, seed=seed, max_size=max_size)
    violations = check_semantic_locality(clusters, vectors)
    if violations:
        retry_seed = seed * 7919 + 97   # arbitrary but deterministic perturbation
        retry_clusters = greedy_cluster(vectors, seed=retry_seed, max_size=max_size)
        retry_violations = check_semantic_locality(retry_clusters, vectors)
        if len(retry_violations) < len(violations):
            clusters, violations = retry_clusters, retry_violations
        for v in violations:
            print(f"[warn] semantic-locality violation at {level_name}: "
                  f"cluster {v['members']} (intra={v['intra_sim']:.3f}) is not more "
                  f"cohesive than sibling {v['nearest_sibling_index']} "
                  f"(sim={v['sibling_sim']:.3f})", file=sys.stderr)
    return clusters, violations


# ---------------------------------------------------------------------------
# Sentence summaries
# ---------------------------------------------------------------------------

def keyword_fallback_sentence(member_texts: list[str], nid: str) -> str:
    """Deterministic fallback sentence, ALWAYS suffixed with the node id --
    since node ids are unique, this is guaranteed globally unique even
    without checking against what's already used."""
    words: list[str] = []
    for text in member_texts:
        for tok in re.findall(r"[A-Za-z0-9]+", text.lower()):
            if tok not in _STOPWORDS and len(tok) > 2:
                words.append(tok)
    top = [w for w, _ in Counter(words).most_common(5)]
    if not top:
        top = ["items"]
    return f"Handle questions about {', '.join(top)} ({nid})"


def unique_fallback_sentence(member_texts: list[str], nid: str, used_norm: set[str]) -> str:
    """Deterministic fallback that is unique against `used_norm` (normalized
    sentences already assigned to other nodes). The nid-suffixed keyword
    sentence is unique by construction, but guard anyway in case some other
    node's fallback happens to collide on keywords+nid formatting."""
    base = keyword_fallback_sentence(member_texts, nid)
    if normalize(base) not in used_norm:
        return base
    return f"{base} [{nid}]"


def summarize_node(member_texts: list[str], used_sentences: list[str], nid: str,
                    llm_call=None) -> str:
    """Produce a one-sentence summary for a node's members. `llm_call`, if
    given, is `(member_texts, used_sentences) -> str`; it is tried up to
    twice (spec: retry once on duplicate/empty) before falling back to the
    deterministic nid-suffixed keyword sentence. `llm_call=None` (the
    --no-llm path) skips straight to the fallback."""
    used_norm = {normalize(s) for s in used_sentences}
    if llm_call is not None:
        for _ in range(2):
            try:
                candidate = llm_call(member_texts, used_sentences)
            except Exception as e:
                print(f"[warn] summary LLM call failed: {e}", file=sys.stderr)
                candidate = ""
            candidate = (candidate or "").strip()
            if candidate and normalize(candidate) not in used_norm:
                return candidate
    return unique_fallback_sentence(member_texts, nid, used_norm)


def call_openai_summary(client, model: str, member_texts: list[str],
                         used_sentences: list[str]) -> str:
    members = "\n".join(f"- {t}" for t in member_texts)
    used = "; ".join(used_sentences) if used_sentences else "(none yet)"
    user = (
        "Write ONE short imperative sentence (max 20 words) describing the "
        "common theme/work of these items.\n\n"
        f"{members}\n\n"
        f"It must be UNIQUE among: [{used}]. Reply with the sentence only."
    )
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You summarize question groups for a task marketplace."},
            {"role": "user", "content": user},
        ],
        max_completion_tokens=200,
    )
    return resp.choices[0].message.content or ""


def cache_key(children: list[str]) -> str:
    return ",".join(sorted(children))


def load_summary_cache(path: Path) -> dict:
    if path.exists():
        return json.loads(path.read_text())
    return {}


def save_summary_cache(path: Path, cache: dict) -> None:
    path.write_text(json.dumps(cache, indent=2, ensure_ascii=False, sort_keys=True))


def summarize_tree(nodes: dict[str, TaskNode], levels: dict[str, list[str]],
                    questions_by_id: dict[str, dict], cache: dict, llm_call=None) -> None:
    """Fill in `.sentence` for every node, bottom-up (L1 -> L2 -> L3) so a
    parent's summary can draw on its already-summarized children. Reuses
    `cache` (keyed by sorted child-id list) and keeps sentences globally
    unique across the whole tree, cache hits included."""
    used_sentences: list[str] = list(cache.values())
    for level_name in LEVEL_NAMES:
        for nid in levels.get(level_name, []):
            node = nodes[nid]
            key = cache_key(node.children)
            if key in cache:
                node.sentence = cache[key]
                continue
            if level_name == "L1":
                member_texts = [questions_by_id[c]["text"] for c in node.children]
            else:
                member_texts = [nodes[c].sentence for c in node.children]
            sentence = summarize_node(member_texts, used_sentences, nid, llm_call=llm_call)
            node.sentence = sentence
            cache[key] = sentence
            used_sentences.append(sentence)


# ---------------------------------------------------------------------------
# Tree construction
# ---------------------------------------------------------------------------

def build_tree(qids: list[str], vectors, seed: int):
    """Bottom-up cluster `qids` (paired with `vectors`, same order) into a
    3-level tree (L1 over questions, L2 over L1 centroids, L3/roots over L2
    centroids), each level branching <= MAX_BRANCH. Returns
    (nodes: dict[nid, TaskNode] with sentence=="" placeholders,
     levels: dict[level_name, list[nid]],
     violations: list[dict] with a 'level' key added)."""
    nid_counter = itertools.count(1)
    nodes: dict[str, TaskNode] = {}
    levels: dict[str, list[str]] = {}
    violations: list[dict] = []

    level_ids: list[str] = list(qids)
    level_vecs = np.asarray(vectors, dtype=float)

    for level_name in LEVEL_NAMES:
        clusters, viol = cluster_level_with_retry(level_vecs, seed, level_name)
        violations.extend({"level": level_name, **v} for v in viol)
        new_ids, new_vecs = [], []
        for cluster in clusters:
            nid = f"t{next(nid_counter):04d}"
            children = [level_ids[i] for i in cluster]
            nodes[nid] = TaskNode(nid, sentence="", children=children)
            new_ids.append(nid)
            new_vecs.append(level_vecs[cluster].mean(axis=0))
        levels[level_name] = new_ids
        level_ids = new_ids
        level_vecs = np.array(new_vecs)

    return nodes, levels, violations


# ---------------------------------------------------------------------------
# Posted-task selection
# ---------------------------------------------------------------------------

def select_posted(levels: dict[str, list[str]], target: int = 30, seed: int = 0) -> list[str]:
    """All roots (L3), plus a seeded-random sample of L1/L2 nodes, up to
    `target` total (or all available nodes if fewer than `target` exist).
    Ancestors and descendants may both be posted -- overlap is allowed."""
    roots = list(dict.fromkeys(levels.get("L3", [])))
    others = [nid for nid in levels.get("L1", []) + levels.get("L2", []) if nid not in roots]
    total_available = len(roots) + len(others)
    want = min(target, total_available)

    if len(roots) >= want:
        return roots[:want]

    posted = list(roots)
    remaining = want - len(posted)
    rng = random.Random(seed)
    pool = list(others)
    rng.shuffle(pool)
    posted.extend(pool[:remaining])
    return posted


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_report(library: TaskLibrary, levels: dict[str, list[str]],
                  posted: list[str], violations: list[dict]) -> None:
    print("=== nodes per level ===")
    for level_name in LEVEL_NAMES:
        print(f"  {level_name}: {len(levels.get(level_name, []))} nodes")
    print(f"  questions (leaves): {len(library.questions)}")

    print("=== leaves per posted task ===")
    hist: Counter = Counter()
    prices = []
    for nid in posted:
        n_leaves = len(library.leaves(nid))
        hist[n_leaves] += 1
        prices.append(library.price(nid))
    for n_leaves in sorted(hist):
        print(f"  {n_leaves} leaves: {hist[n_leaves]} tasks")

    print("=== price distribution (posted) ===")
    if prices:
        print(f"  min={min(prices)} max={max(prices)} "
              f"mean={sum(prices)/len(prices):.0f} total={sum(prices)}")

    print(f"semantic-locality violations: {len(violations)}")
    print(f"total posted value: {sum(prices)}")


# ---------------------------------------------------------------------------
# Embedding + CLI
# ---------------------------------------------------------------------------

def embed_texts(texts: list[str]) -> np.ndarray:
    from chromadb.utils.embedding_functions import DefaultEmbeddingFunction
    ef = DefaultEmbeddingFunction()
    return np.asarray(ef(texts), dtype=float)


def download_pool(hotpot_n: int, musique_n: int, seed: int) -> tuple[list[dict], list[dict]]:
    """Thin wrapper around prepare_data's downloader, kept as its own
    function so tests can monkeypatch it out entirely (no network)."""
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import prepare_data  # noqa: E402
    return prepare_data.build_pool_and_corpus(hotpot_n, musique_n, seed)


def build_corpus_index(corpus: list[dict], persist_dir: str) -> None:
    from ca.retrieval import ChromaBackend
    ChromaBackend(corpus, persist_dir=persist_dir)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hotpot-n", type=int, default=90)
    ap.add_argument("--musique-n", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="data/v2")
    ap.add_argument("--post-target", type=int, default=30)
    ap.add_argument("--model", default="gpt-5-mini")
    ap.add_argument("--no-llm", action="store_true",
                     help="skip the LLM summarizer; use the deterministic keyword "
                          "fallback for every node (offline/CI builds)")
    args = ap.parse_args(argv)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    pool, corpus = download_pool(args.hotpot_n, args.musique_n, args.seed)
    with open(out / "pool.jsonl", "w") as f:
        for r in pool:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"pool: {len(pool)} questions; corpus: {len(corpus)} paragraphs")

    build_corpus_index(corpus, str(out / "index"))
    print(f"chroma index saved to {out}/index")

    vectors = embed_texts([q["text"] for q in pool])
    qids = [q["qid"] for q in pool]
    nodes, levels, violations = build_tree(qids, vectors, args.seed)

    cache_path = out / "summaries.json"
    cache = load_summary_cache(cache_path)

    llm_call = None
    if not args.no_llm:
        from openai import OpenAI
        client = OpenAI()
        llm_call = functools.partial(call_openai_summary, client, args.model)

    questions_by_id = {q["qid"]: q for q in pool}
    summarize_tree(nodes, levels, questions_by_id, cache, llm_call=llm_call)
    save_summary_cache(cache_path, cache)

    posted = select_posted(levels, target=args.post_target, seed=args.seed)

    questions = [Question(q["qid"], q["text"], q["answers"], q["difficulty"], q["price"])
                 for q in pool]
    library = TaskLibrary(nodes, questions, posted)
    library.to_json(str(out / "library.json"))
    print(f"library saved to {out}/library.json ({len(nodes)} nodes, {len(posted)} posted)")

    print_report(library, levels, posted, violations)


if __name__ == "__main__":
    main()
