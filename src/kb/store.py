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
  embedded text did not change)."""
import json
import uuid
from dataclasses import dataclass, field

import chromadb


class StoreError(Exception):
    pass


@dataclass
class Node:
    nid: str
    text: str
    origin: str | None                 # None = agent-authored (no provenance)
    flag: str | None = None            # None | "authored" | "edited"
    links: list[str] = field(default_factory=list)


class Store:
    def __init__(self, embedding_function=None):
        self.nodes: dict[str, Node] = {}
        self.origins: list[str] = []       # build-time origin ids, fixed
        self.dirty: set[str] = set()       # node ids awaiting (re)embedding
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
            self._collection = self._client.get_or_create_collection(
                f"{self._prefix}-nodes", **kw)
        return self._collection

    # ---------- loading (build output or snapshot) ----------

    @classmethod
    def from_nodes(cls, nodes: list[dict], embedding_function=None) -> "Store":
        s = cls(embedding_function)
        for n in nodes:
            node = Node(n["id"], n["text"], n["origin"], n.get("flag"),
                        list(n.get("links", [])))
            s.nodes[node.nid] = node
        s.origins = sorted({n.origin for n in s.nodes.values()
                            if n.origin is not None})
        nums = ([int(i[1:]) for i in s.nodes]
                + [int(o[1:]) for o in s.origins])
        s._next_id = max(nums, default=0) + 1
        s._embed_all()
        return s

    @classmethod
    def from_json(cls, state: dict, embedding_function=None) -> "Store":
        s = cls.from_nodes(state["nodes"], embedding_function)
        s.origins = list(state["origins"])
        s._next_id = max(s._next_id, state["next_id"])
        return s

    def to_json(self) -> dict:
        return {"nodes": [{"id": n.nid, "text": n.text, "origin": n.origin,
                           "flag": n.flag, "links": list(n.links)}
                          for n in self.nodes.values()],
                "origins": list(self.origins),
                "next_id": self._next_id}

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

    def add(self, text: str) -> str:
        """Author a NEW note: new id, origin None (no build-time provenance
        to descend from), flag "authored". Never counts toward coverage or
        duplication."""
        text = str(text).strip()
        if not text:
            raise StoreError("note text must not be empty")
        node = Node(self._new_id(), text, None, "authored")
        self.nodes[node.nid] = node
        self.dirty.add(node.nid)
        return node.nid

    def edit(self, nid: str, text: str) -> None:
        """Rewrite a note's text in place: id AND origin are kept, flag
        becomes "edited" — the node no longer carries its origin's text, so
        it stops counting toward coverage / duplication."""
        node = self._node(nid)
        text = str(text).strip()
        if not text:
            raise StoreError("note text must not be empty")
        node.text = text
        node.flag = "edited"
        self.dirty.add(nid)

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
        """Live nodes per origin, over origin-preserving UNEDITED nodes only:
        authored ones have no origin, edited ones no longer carry their
        origin's text."""
        counts = {o: 0 for o in self.origins}
        for n in self.nodes.values():
            if n.origin is not None and n.flag != "edited":
                counts[n.origin] = counts.get(n.origin, 0) + 1
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
