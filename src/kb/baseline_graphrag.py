"""B3 — GraphRAG-style offline structuring baseline (T40).

Follows the GraphRAG recipe in spirit, adapted to OUR store format so the
fixed reader (kb.test) runs on the output unchanged:

1. entity extraction per statement — LEXICAL (see baseline_common: the
   statements are template-shaped, a regex extractor is exact and free;
   documented deviation from the original's LLM extraction step);
2. entity communities over the person co-occurrence graph — deterministic
   size-capped greedy agglomeration: edges sorted by descending co-occurrence
   weight (ties lexicographic), merged while the union stays within
   --max-size. A Leiden-free stand-in; persons with no relations (e.g. name
   distractors) stay singleton communities;
3. ONE summary node per community (LLM, default gpt-5-mini; temperature 0
   where the model accepts it), flagged "authored" with origin None, linked
   to every statement that mentions a member person. A statement naming
   persons from two communities is linked from both summaries, intentionally.

The original universe file is never touched; the output directory gets a full
universe copy (questions / splits intact) plus build_meta.json with node /
link / token tallies.

    python -m kb.baseline_graphrag --universe data/v10L/universe.json \
        --out data/v10L_graphrag [--model gpt-5-mini] [--max-size 10] [--stub]
"""
import argparse
import json
import sys
import time
from collections import Counter

from kb.baseline_common import (PERSON_RELATIONS, copy_universe,
                                index_statements, next_sid, write_output)

MAX_COMMUNITY = 10


class StubSummarizer:
    """Deterministic offline stand-in (tests / --stub): no LLM, no tokens."""

    def __init__(self):
        self.tokens_in = self.tokens_out = self.calls = self.empty = 0

    def __call__(self, members: list[str], texts: list[str]) -> str:
        self.calls += 1
        return (f"Summary of {len(texts)} notes about "
                + ", ".join(members) + ".")


class LLMSummarizer:
    """One chat call per community (default gpt-5-mini). gpt-5*/o* models
    reject non-default temperature and want max_completion_tokens (same
    handling as kb.policy); reasoning models may burn the whole completion
    budget on reasoning and return empty content — the budget is sized for
    the largest communities' fact lists, and a still-empty reply falls back
    to the stub text so the build never emits an empty node."""

    def __init__(self, model: str = "gpt-5-mini", max_tokens: int = 6000):
        import os

        from openai import OpenAI
        self.client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"),
                             max_retries=5)
        self.model = model
        self.max_tokens = max_tokens
        self.tokens_in = self.tokens_out = self.calls = self.empty = 0

    def __call__(self, members: list[str], texts: list[str]) -> str:
        prompt = (
            "Summarize the following facts about a group of related people "
            "in ONE paragraph of at most 120 words. Use full names only, "
            "never pronouns; mention every listed person at least once; "
            "cover the family ties, marriages, friendships, jobs, hobbies, "
            "home cities and birth dates that appear in the facts.\n"
            f"People: {', '.join(members)}\n"
            "Facts:\n" + "\n".join(f"- {t}" for t in texts))
        kwargs = dict(model=self.model,
                      messages=[{"role": "user", "content": prompt}])
        if self.model.startswith("gpt-5") or self.model.startswith("o"):
            kwargs["max_completion_tokens"] = self.max_tokens
        else:
            kwargs["max_tokens"] = self.max_tokens
            kwargs["temperature"] = 0
        resp = self.client.chat.completions.create(**kwargs)
        self.calls += 1
        usage = resp.usage
        self.tokens_in += getattr(usage, "prompt_tokens", 0) or 0
        self.tokens_out += getattr(usage, "completion_tokens", 0) or 0
        text = " ".join((resp.choices[0].message.content or "").split())
        if not text:
            self.empty += 1
            text = (f"Summary of {len(texts)} notes about "
                    + ", ".join(members) + ".")
        return text


def _communities(persons: list[str], weights: Counter,
                 max_size: int) -> list[list[str]]:
    """Deterministic size-capped greedy agglomeration over the co-occurrence
    graph: strongest edges first, merge while the union fits max_size."""
    parent = {p: p for p in persons}
    size = {p: 1 for p in persons}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for (a, b), _w in sorted(weights.items(), key=lambda kv: (-kv[1], kv[0])):
        ra, rb = find(a), find(b)
        if ra != rb and size[ra] + size[rb] <= max_size:
            root, other = min(ra, rb), max(ra, rb)
            parent[other] = root
            size[root] = size[ra] + size[rb]
    groups: dict[str, list[str]] = {}
    for p in persons:
        groups.setdefault(find(p), []).append(p)
    return sorted((sorted(g) for g in groups.values()), key=lambda g: g[0])


def build_graphrag(universe: dict, summarize,
                   max_size: int = MAX_COMMUNITY) -> tuple[dict, dict]:
    """(universe copy with summary nodes + links added, build meta)."""
    t0 = time.perf_counter()
    out = copy_universe(universe)
    by_entity, triples = index_statements(out["nodes"])
    person_sids = {name: sids for (kind, name), sids in by_entity.items()
                   if kind == "person"}
    texts = {n["id"]: n["text"] for n in out["nodes"]}

    weights: Counter = Counter()
    for subj, rel, obj in triples.values():
        if (rel in PERSON_RELATIONS and subj != obj
                and subj in person_sids and obj in person_sids):
            weights[tuple(sorted((subj, obj)))] += 1

    comms = _communities(sorted(person_sids), weights, max_size)
    sid_n = next_sid(out["nodes"])
    nodes_added = links_added = 0
    for members in comms:
        member_sids = sorted({s for m in members for s in person_sids[m]})
        summary = summarize(members, [texts[s] for s in member_sids])
        nid = f"s{sid_n:04d}"
        sid_n += 1
        out["nodes"].append({"id": nid, "text": f"Community summary: {summary}",
                             "origin": None, "flag": "authored",
                             "links": member_sids})
        nodes_added += 1
        links_added += len(member_sids)

    sizes = [len(c) for c in comms]
    meta = {"baseline": "graphrag",
            "n_person_entities": len(person_sids),
            "n_communities": len(comms),
            "community_size_min": min(sizes), "community_size_max": max(sizes),
            "singleton_communities": sum(1 for s in sizes if s == 1),
            "nodes_added": nodes_added, "links_added": links_added,
            "llm_calls": summarize.calls,
            "empty_summaries": summarize.empty,
            "build_tokens_in": summarize.tokens_in,
            "build_tokens_out": summarize.tokens_out,
            "seconds": time.perf_counter() - t0}
    out["meta"] = {**out.get("meta", {}), "baseline": meta}
    return out, meta


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--universe", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="gpt-5-mini")
    ap.add_argument("--max-size", type=int, default=MAX_COMMUNITY)
    ap.add_argument("--stub", action="store_true",
                    help="offline stub summarizer (tests / dry runs)")
    args = ap.parse_args(argv)
    with open(args.universe) as f:
        universe = json.load(f)
    summarize = StubSummarizer() if args.stub else LLMSummarizer(args.model)
    built, meta = build_graphrag(universe, summarize, args.max_size)
    meta["universe"] = args.universe
    meta["model"] = "stub" if args.stub else args.model
    write_output(args.out, built, meta)
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    sys.exit(main())
