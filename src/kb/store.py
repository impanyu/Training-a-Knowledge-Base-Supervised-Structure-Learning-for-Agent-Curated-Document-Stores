"""The statement-node graph (v9.6): the doc/statement two-level structure is
collapsed. The store is a graph of statement NODES:

    node { id ("s0421"), text (one self-contained sentence), links [id...] }

- No summaries and no summarizer machinery: a node's text IS its display.
  Embeddings live at node level (the text is embedded); an add or edit marks
  the node dirty and `refresh()` — called at iteration end only, keeping the
  batch semantics — (re)embeds every dirty node; delete drops the row.
- Provenance (T39.5 rule carried over): initial nodes carry `origin` (their
  own id at build time); `edit` keeps id AND origin and flags "edited";
  `add` mints origin None, flag "authored". Coverage / duplication range
  over origin-preserving UNEDITED nodes only — integer-exact. Live flagged
  nodes are tallied as `authored_statements` / `edited_statements`.
- Links are stored per node (ids only, no copies); `delete` cascades inbound
  link removal from every other node. Link changes never mark dirty (the
  embedded text did not change).
- Infrastructure dedup (T42, active iff a `judge` is injected): after add()
  or edit() the STORE embeds the candidate text on the fly, queries the
  top-3 similar nodes (excluding the edited node itself), and if the best
  cosine >= dedup_threshold (default 0.90) asks the duplicate judge —
  callable(text_a, text_b) -> bool; LLMDuplicateJudge live, a stub in
  tests. Judged duplicates merge: an add creates nothing; an edit merges
  the edited node into the survivor (inbound links rewired with dedupe,
  outbound links unioned, node deleted). The survivor's `absorbed` list
  carries the merged-away node's origin and everything it had absorbed.
  Coverage rule: an origin is alive iff some node CARRIES it unedited
  (flag != "edited", the T39.5 rule) OR some node lists it in `absorbed`;
  absorbed entries persist for the survivor's lifetime (later edits of the
  survivor keep them — the merge certified the fact identical, and edit
  keeps origins by the same logic) and die with its deletion. Both
  representations count toward duplication."""
import json
import uuid
from dataclasses import dataclass, field

import chromadb

DEDUP_THRESHOLD = 0.90


class StoreError(Exception):
    pass


@dataclass
class Node:
    nid: str
    text: str
    origin: str | None                 # None = agent-authored (no provenance)
    flag: str | None = None            # None | "authored" | "edited"
    links: list[str] = field(default_factory=list)
    absorbed: list[str] = field(default_factory=list)   # origins merged in


class LLMDuplicateJudge:
    """Live duplicate judge (T42): gpt-5-mini, temperature 0, minimal
    prompt. Fails OPEN (not a duplicate) — a lost judge call must not kill
    a multi-hour run, and accepting a near-duplicate is the recoverable
    mistake."""

    PROMPT = ("Do these two sentences state the same fact? "
              "Answer yes or no.")

    def __init__(self, model: str = "gpt-5-mini", max_tokens: int = 500,
                 reasoning_effort: str = "minimal"):
        import os

        from openai import OpenAI
        self.client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"),
                             max_retries=5)
        self.model = model
        self.max_tokens = max_tokens
        self.reasoning_effort = reasoning_effort
        self.tokens_in = 0
        self.tokens_out = 0

    def __call__(self, text_a: str, text_b: str) -> bool:
        import sys
        kwargs = dict(model=self.model,
                      messages=[{"role": "system", "content": self.PROMPT},
                                {"role": "user",
                                 "content": f"1. {text_a}\n2. {text_b}"}])
        if self.model.startswith("gpt-5") or self.model.startswith("o"):
            # A reasoning model spends its completion budget thinking before
            # it emits anything: at the 4 tokens this judge used to allow it
            # returned empty every single time, which the caller read as "not
            # a duplicate". The guard was wired up and inert. Measured on
            # gpt-5-mini: 4 and 64 tokens both yield empty; 500 with minimal
            # effort answers in about 10.
            kwargs["max_completion_tokens"] = self.max_tokens
            if self.reasoning_effort:
                kwargs["reasoning_effort"] = self.reasoning_effort
        else:
            kwargs["max_tokens"] = self.max_tokens
            kwargs["temperature"] = 0
        try:
            resp = self.client.chat.completions.create(**kwargs)
        except Exception as e:
            print(f"[judge-error] {e}", file=sys.stderr)
            return False
        usage = resp.usage
        self.tokens_in += getattr(usage, "prompt_tokens", 0) or 0
        self.tokens_out += getattr(usage, "completion_tokens", 0) or 0
        return (resp.choices[0].message.content or "").strip().lower().startswith("yes")


class Store:
    def __init__(self, embedding_function=None, judge=None,
                 dedup_threshold: float = DEDUP_THRESHOLD):
        self.nodes: dict[str, Node] = {}
        self.origins: list[str] = []       # build-time origin ids, fixed
        self.dirty: set[str] = set()       # node ids awaiting (re)embedding
        self.judge = judge                 # None -> dedup off
        self.dedup_threshold = dedup_threshold
        self.merges = 0                    # cumulative, add- and edit-merges
        self._next_id = 1
        self._ef = embedding_function      # None -> chroma default ONNX
        self._client = chromadb.EphemeralClient()
        # chroma hands every in-process EphemeralClient the SAME in-memory db,
        # so an instance-unique prefix is what actually isolates two stores.
        self._prefix = f"kb{uuid.uuid4().hex[:8]}"
        self._collection = None

    def _col(self):
        if self._collection is None:
            kw = {"embedding_function": self._ef} if self._ef is not None else {}
            # cosine space: dedup thresholds are cosine similarities
            # (1 - distance); ranking is unchanged for normalized embeddings
            self._collection = self._client.get_or_create_collection(
                f"{self._prefix}-nodes", metadata={"hnsw:space": "cosine"},
                **kw)
        return self._collection

    # ---------- loading (build output or snapshot) ----------

    @classmethod
    def from_nodes(cls, nodes: list[dict], embedding_function=None,
                   judge=None, dedup_threshold: float = DEDUP_THRESHOLD) -> "Store":
        s = cls(embedding_function, judge, dedup_threshold)
        for n in nodes:
            node = Node(n["id"], n["text"], n["origin"], n.get("flag"),
                        list(n.get("links", [])),
                        list(n.get("absorbed", [])))
            s.nodes[node.nid] = node
        s.origins = sorted({n.origin for n in s.nodes.values()
                            if n.origin is not None})
        nums = ([int(i[1:]) for i in s.nodes]
                + [int(o[1:]) for o in s.origins])
        s._next_id = max(nums, default=0) + 1
        s._embed_all()
        return s

    @classmethod
    def from_json(cls, state: dict, embedding_function=None,
                  judge=None, dedup_threshold: float = DEDUP_THRESHOLD) -> "Store":
        s = cls.from_nodes(state["nodes"], embedding_function, judge,
                           dedup_threshold)
        s.origins = list(state["origins"])
        s._next_id = max(s._next_id, state["next_id"])
        s.merges = state.get("merges", 0)
        return s

    def to_json(self) -> dict:
        return {"nodes": [{"id": n.nid, "text": n.text, "origin": n.origin,
                           "flag": n.flag, "links": list(n.links),
                           "absorbed": list(n.absorbed)}
                          for n in self.nodes.values()],
                "origins": list(self.origins),
                "next_id": self._next_id,
                "merges": self.merges}

    def _embed_all(self) -> None:
        """Embeddings are rebuilt on load: the snapshot carries only text;
        whatever EF this store was given re-embeds every node once."""
        ids = list(self.nodes)
        if not ids:
            return
        col = self._col()
        for i in range(0, len(ids), 500):
            batch = ids[i:i + 500]
            col.upsert(ids=batch,
                       documents=[self.nodes[i].text for i in batch])

    # ---------- lookups ----------

    def _node(self, nid: str) -> Node:
        if nid not in self.nodes:
            raise StoreError(f"no such note {nid}")
        return self.nodes[nid]

    def _new_id(self) -> str:
        nid = f"s{self._next_id:04d}"
        self._next_id += 1
        return nid

    # ---------- edits (agents call these through actions.dispatch) ----------

    def _near_duplicate(self, text: str, exclude: str | None = None) -> str | None:
        """Dedup gate (T42): embed `text` on the fly, take the BEST similar
        node (top-3 queried so `exclude` and dead ids can be skipped). Below
        dedup_threshold cosine -> None without a judge call; at or above ->
        one judge call on that best pair, returning its id iff judged the
        same fact. No-op when no judge is configured."""
        if self.judge is None:
            return None
        col = self._col()
        n = min(3 + (1 if exclude else 0), col.count())
        if n <= 0:
            return None
        res = col.query(query_texts=[text], n_results=n)
        for nid, dist in zip(res["ids"][0], res["distances"][0]):
            if nid == exclude or nid not in self.nodes:
                continue
            if 1.0 - dist < self.dedup_threshold:
                return None                 # best is below: no judge call
            return nid if self.judge(text, self.nodes[nid].text) else None
        return None

    def add(self, text: str) -> tuple[str | None, str | None]:
        """Author a NEW note -> (new_id, None): new id, origin None (no
        build-time provenance to descend from), flag "authored"; never
        counts toward coverage or duplication. If the text is judged a
        duplicate of an existing note -> (None, survivor_id): NOTHING is
        created (the candidate carries no provenance to absorb) and the
        merge counter ticks."""
        text = str(text).strip()
        if not text:
            raise StoreError("note text must not be empty")
        dup = self._near_duplicate(text)
        if dup is not None:
            self.merges += 1
            return None, dup
        node = Node(self._new_id(), text, None, "authored")
        self.nodes[node.nid] = node
        self.dirty.add(node.nid)
        return node.nid, None

    def edit(self, nid: str, text: str) -> str | None:
        """Rewrite a note's text in place -> None: id AND origin are kept,
        flag becomes "edited" — the node no longer carries its origin's
        text, so it stops counting toward coverage / duplication. If the
        new text is judged a duplicate of ANOTHER note -> survivor_id: the
        edited node is merged into it instead (see merge)."""
        self._node(nid)
        text = str(text).strip()
        if not text:
            raise StoreError("note text must not be empty")
        dup = self._near_duplicate(text, exclude=nid)
        if dup is not None:
            self.merge(nid, dup)
            return dup
        node = self.nodes[nid]
        node.text = text
        node.flag = "edited"
        self.dirty.add(nid)
        return None

    def merge(self, loser_id: str, survivor_id: str) -> None:
        """Merge `loser` into `survivor` (T42): the survivor absorbs the
        loser's origin and everything the loser had absorbed; every inbound
        link to the loser is rewired to the survivor (deduped, never
        self-linking); the loser's outbound links are unioned into the
        survivor's; the loser is deleted (embedding row dropped). Also the
        deterministic replay primitive for recorded edit-merges."""
        x = self._node(loser_id)
        y = self._node(survivor_id)
        if loser_id == survivor_id:
            raise StoreError("cannot merge a note into itself")
        for o in ([x.origin] if x.origin is not None else []) + x.absorbed:
            if o not in y.absorbed:
                y.absorbed.append(o)
        for lid in x.links:                       # outbound union
            if lid != survivor_id and lid not in y.links:
                y.links.append(lid)
        for n in self.nodes.values():             # inbound rewire, deduped
            if n.nid == loser_id or loser_id not in n.links:
                continue
            n.links.remove(loser_id)
            if n.nid != survivor_id and survivor_id not in n.links:
                n.links.append(survivor_id)
        del self.nodes[loser_id]
        self.dirty.discard(loser_id)
        try:
            self._col().delete(ids=[loser_id])
        except Exception:
            pass
        self.merges += 1

    def delete(self, nid: str) -> None:
        """Delete a note: inbound links to it are removed from every other
        node (cascade), its embedding row is dropped. An origin's last
        unedited node dying = coverage loss, allowed."""
        self._node(nid)
        del self.nodes[nid]
        for other in self.nodes.values():
            if nid in other.links:
                other.links.remove(nid)
        self.dirty.discard(nid)
        try:
            self._col().delete(ids=[nid])
        except Exception:
            pass

    def link(self, a: str, b: str) -> None:
        na = self._node(a)
        self._node(b)
        if a == b:
            raise StoreError("cannot link a note to itself")
        if b in na.links:
            raise StoreError(f"{a} already links to {b}")
        na.links.append(b)          # link changes never mark dirty

    def unlink(self, a: str, b: str) -> None:
        na = self._node(a)
        if b not in na.links:
            raise StoreError(f"{a} does not link to {b}")
        na.links.remove(b)

    # ---------- sense ----------

    def search(self, query: str, k: int = 5) -> list[tuple[str, str]]:
        """Top-k (id, full text) by embedding similarity."""
        col = self._col()
        n = min(k, col.count())
        if n <= 0:
            return []
        res = col.query(query_texts=[str(query)], n_results=n)
        return [(nid, self.nodes[nid].text)
                for nid in res["ids"][0] if nid in self.nodes]

    def read(self, nid: str) -> Node:
        return self._node(nid)

    # ---------- maintenance (iteration end only) ----------

    def refresh(self) -> int:
        """(Re)embed every dirty node in one batch; returns how many. With
        summaries gone this is ALL the environment maintains."""
        ids = [nid for nid in sorted(self.dirty) if nid in self.nodes]
        if ids:
            self._col().upsert(ids=ids,
                               documents=[self.nodes[i].text for i in ids])
        self.dirty.clear()
        return len(ids)

    # ---------- stats (kb_stats.jsonl row) ----------

    def origin_counts(self) -> dict[str, int]:
        """Live representations per origin (T42 rule): an origin counts once
        for every node that CARRIES it unedited (origin == o, flag !=
        "edited" — authored nodes have no origin, edited ones no longer
        carry their origin's text) plus once for every node listing it in
        `absorbed` (the merge certified the survivor's text states the same
        fact; the entry lives as long as the survivor does). Coverage =
        origins with >= 1 representation; duplication = origins with > 1."""
        counts = {o: 0 for o in self.origins}
        for n in self.nodes.values():
            if n.origin is not None and n.flag != "edited":
                counts[n.origin] = counts.get(n.origin, 0) + 1
            for o in n.absorbed:
                counts[o] = counts.get(o, 0) + 1
        return counts

    def stats(self) -> dict:
        counts = self.origin_counts()
        alive = sum(1 for o in self.origins if counts.get(o, 0) >= 1)
        inbound = {i: 0 for i in self.nodes}
        for n in self.nodes.values():
            for lid in n.links:
                if lid in inbound:
                    inbound[lid] += 1
        orphans = sum(1 for i, n in self.nodes.items()
                      if not n.links and inbound[i] == 0)
        return {
            "n_nodes": len(self.nodes),
            "n_links": sum(len(n.links) for n in self.nodes.values()),
            "origins_total": len(self.origins),
            "origins_alive": alive,
            "coverage": alive / len(self.origins) if self.origins else 0.0,
            "dup_origins": sum(1 for o in self.origins if counts.get(o, 0) > 1),
            # live flagged nodes (nodes ARE statements), not cumulative
            "authored_statements": sum(1 for n in self.nodes.values()
                                       if n.flag == "authored"),
            "edited_statements": sum(1 for n in self.nodes.values()
                                     if n.flag == "edited"),
            # approximate token count of all live node text: total chars // 4
            # (the usual ~4-chars-per-token heuristic; no tiktoken)
            "statement_tokens": sum(len(n.text)
                                    for n in self.nodes.values()) // 4,
            "orphan_nodes": orphans,
            "merges": self.merges,         # cumulative add- and edit-merges
        }


def save_snapshot(store: Store, path, universe_path=None) -> None:
    """Per-epoch KB snapshot: node JSON + the universe it belongs to, so
    kb.test can find the question file without extra flags."""
    with open(path, "w") as f:
        json.dump({"universe": str(universe_path) if universe_path else None,
                   "store": store.to_json()}, f, ensure_ascii=False)


def load_snapshot(path, embedding_function=None):
    """(store, universe_path_or_None) from a save_snapshot file."""
    with open(path) as f:
        state = json.load(f)
    return (Store.from_json(state["store"], embedding_function),
            state.get("universe"))
