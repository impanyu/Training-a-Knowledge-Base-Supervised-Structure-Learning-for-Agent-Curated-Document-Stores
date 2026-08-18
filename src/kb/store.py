"""The document store (spec §1): docs of immutable statement instances with
origin bookkeeping, links by id, environment-maintained summaries and
embeddings.

- Statements are moved/copied/deleted by reference; build-time instances
  descend from an original via `origin` (copies share origin). T39.5 adds
  two authoring edits: `add_statement` mints a NEW statement (origin None,
  flag "authored") and `edit_statement` rewrites an instance's text in place
  (sid and origin kept, flag "edited"). Coverage / duplication stay defined
  over origin-preserving UNEDITED instances only — an authored instance has
  no origin, an edited one no longer carries its origin's text — so both
  metrics remain integer-exact; flags survive copy/move and snapshots.
  Live flagged instances are tallied as `authored_statements` /
  `edited_statements` in stats().
- Any statement-set change marks the doc dirty; `refresh()` (called at
  iteration end only) regenerates dirty docs' summaries via the injectable
  Summarizer and their embeddings via the chroma collection (default ONNX EF
  at runtime, HashEmbedding in tests). Link changes trigger neither.
- `delete_doc` cascades: instances die, inbound links are removed from all
  other docs, the embedding row is dropped."""
import json
import uuid
from dataclasses import dataclass, field

import chromadb

from kb.grader import normalize

SNIPPET_CHARS = 120


class StoreError(Exception):
    pass


@dataclass
class Statement:
    sid: str
    text: str
    origin: str | None                 # None = agent-authored (no provenance)
    flag: str | None = None            # None | "authored" | "edited"


@dataclass
class Doc:
    doc_id: str
    summary: str = "(pending)"
    statements: list[Statement] = field(default_factory=list)
    links: list[str] = field(default_factory=list)


# ---------------- summarizers (injectable) ----------------

class TemplateSummarizer:
    """Deterministic no-LLM summary: "Facts about <name>." from the first
    statement's leading name tokens. The build-time default and the test stub."""

    def __init__(self):
        self.tokens_in = 0
        self.tokens_out = 0

    def summarize(self, texts: list[str]) -> str:
        words = str(texts[0]).split()[:2]
        name = " ".join(w[:-2] if w.endswith("'s") else w for w in words).strip(".,")
        return f"Facts about {name}."


class LLMSummarizer:
    """Fixed-prompt summarizer (spec §1: configurable model, default
    gpt-5-mini, temp 0, <=60 output tokens). Falls back to the template on
    API failure — a lost summary must not kill a multi-hour run."""

    PROMPT = ("Summarize the following statements in one sentence of at most "
              "25 words. Output only the summary, nothing else.")

    def __init__(self, model: str = "gpt-5-mini", max_tokens: int = 60):
        import os

        from openai import OpenAI
        self.client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"), max_retries=5)
        self.model = model
        self.max_tokens = max_tokens
        self.tokens_in = 0
        self.tokens_out = 0
        self._fallback = TemplateSummarizer()

    def summarize(self, texts: list[str]) -> str:
        import sys
        kwargs = dict(model=self.model,
                      messages=[{"role": "system", "content": self.PROMPT},
                                {"role": "user", "content": "\n".join(texts)}])
        if self.model.startswith("gpt-5") or self.model.startswith("o"):
            kwargs["max_completion_tokens"] = self.max_tokens
        else:
            kwargs["max_tokens"] = self.max_tokens
            kwargs["temperature"] = 0
        try:
            resp = self.client.chat.completions.create(**kwargs)
        except Exception as e:
            print(f"[summarizer-error] {e}", file=sys.stderr)
            return self._fallback.summarize(texts)
        usage = resp.usage
        self.tokens_in += getattr(usage, "prompt_tokens", 0) or 0
        self.tokens_out += getattr(usage, "completion_tokens", 0) or 0
        out = (resp.choices[0].message.content or "").strip()
        return out or self._fallback.summarize(texts)


# ---------------- the store ----------------

class Store:
    def __init__(self, summarizer=None, embedding_function=None):
        self.summarizer = summarizer or TemplateSummarizer()
        self.docs: dict[str, Doc] = {}
        self.origins: list[str] = []       # build-time origin sids, fixed
        self.dirty: set[str] = set()
        self.created_docs = 0              # agent-era counters
        self.deleted_docs = 0
        self._home: dict[str, str] = {}    # live sid -> doc_id
        self._next_doc = 1
        self._next_sid = 1
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
                f"{self._prefix}-docs", **kw)
        return self._collection

    # ---------- loading (build output or snapshot) ----------

    @classmethod
    def from_docs(cls, docs: list[dict], summarizer=None, embedding_function=None) -> "Store":
        s = cls(summarizer, embedding_function)
        for d in docs:
            doc = Doc(d["id"], d["summary"],
                      [Statement(st["sid"], st["text"], st["origin"],
                                 st.get("flag"))
                       for st in d["statements"]],
                      list(d.get("links", [])))
            s.docs[doc.doc_id] = doc
            for st in doc.statements:
                s._home[st.sid] = doc.doc_id
        s.origins = sorted({st.origin for doc in s.docs.values()
                            for st in doc.statements
                            if st.origin is not None})
        nums = [int(i[1:]) for i in s.docs]
        s._next_doc = max(nums, default=0) + 1
        sids = ([int(i[1:]) for i in s._home]
                + [int(o[1:]) for o in s.origins])
        s._next_sid = max(sids, default=0) + 1
        s._embed_all()
        return s

    @classmethod
    def from_json(cls, state: dict, summarizer=None, embedding_function=None) -> "Store":
        s = cls.from_docs(state["docs"], summarizer, embedding_function)
        s.origins = list(state["origins"])
        s._next_doc = max(s._next_doc, state["next_doc"])
        s._next_sid = max(s._next_sid, state["next_sid"])
        s.created_docs = state.get("created_docs", 0)
        s.deleted_docs = state.get("deleted_docs", 0)
        return s

    def to_json(self) -> dict:
        return {"docs": [{"id": d.doc_id, "summary": d.summary,
                          "statements": [{"sid": st.sid, "text": st.text,
                                          "origin": st.origin,
                                          "flag": st.flag}
                                         for st in d.statements],
                          "links": list(d.links)}
                         for d in self.docs.values()],
                "origins": list(self.origins),
                "next_doc": self._next_doc, "next_sid": self._next_sid,
                "created_docs": self.created_docs,
                "deleted_docs": self.deleted_docs}

    def _embed_all(self) -> None:
        """Embeddings are rebuilt on load (spec §4): the snapshot carries only
        text; whatever EF this store was given re-embeds every doc once."""
        ids = list(self.docs)
        if not ids:
            return
        col = self._col()
        for i in range(0, len(ids), 500):
            batch = ids[i:i + 500]
            col.upsert(ids=batch, documents=[self._doc_text(d) for d in batch])

    def _doc_text(self, doc_id: str) -> str:
        doc = self.docs[doc_id]
        return " ".join(st.text for st in doc.statements) or "(empty)"

    # ---------- lookups ----------

    def _doc(self, doc_id: str) -> Doc:
        if doc_id not in self.docs:
            raise StoreError(f"no such doc {doc_id}")
        return self.docs[doc_id]

    def _live(self, sid: str) -> str:
        if sid not in self._home:
            raise StoreError(f"no live statement {sid}")
        return self._home[sid]

    def _new_sid(self) -> str:
        sid = f"s{self._next_sid:04d}"
        self._next_sid += 1
        return sid

    # ---------- edits (agents call these through actions.dispatch) ----------

    def copy_statement(self, sid: str, to_doc: str) -> str:
        src = self.docs[self._live(sid)]
        dst = self._doc(to_doc)
        st = next(x for x in src.statements if x.sid == sid)
        new = Statement(self._new_sid(), st.text, st.origin, st.flag)
        dst.statements.append(new)
        self._home[new.sid] = dst.doc_id
        self.dirty.add(dst.doc_id)
        return new.sid

    def move_statement(self, sid: str, to_doc: str) -> None:
        home = self._live(sid)
        dst = self._doc(to_doc)
        if home == to_doc:
            raise StoreError(f"{sid} is already in {to_doc}")
        src = self.docs[home]
        st = next(x for x in src.statements if x.sid == sid)
        src.statements.remove(st)
        dst.statements.append(st)
        self._home[sid] = to_doc
        self.dirty.add(home)
        self.dirty.add(to_doc)

    def delete_statement(self, sid: str) -> None:
        home = self._live(sid)
        doc = self.docs[home]
        doc.statements = [x for x in doc.statements if x.sid != sid]
        del self._home[sid]
        self.dirty.add(home)

    def add_statement(self, doc_id: str, text: str) -> str:
        """Author a NEW statement (T39.5): new sid, origin None (no build-time
        provenance to descend from), flag "authored". Never counts toward
        coverage or duplication."""
        doc = self._doc(doc_id)
        text = str(text).strip()
        if not text:
            raise StoreError("statement text must not be empty")
        st = Statement(self._new_sid(), text, None, "authored")
        doc.statements.append(st)
        self._home[st.sid] = doc_id
        self.dirty.add(doc_id)
        return st.sid

    def edit_statement(self, sid: str, text: str) -> None:
        """Rewrite an existing instance's text in place (T39.5): sid and
        origin are kept, flag becomes "edited" — the instance no longer
        carries its origin's text, so it stops counting toward coverage /
        duplication (an authored instance stays outside them either way)."""
        home = self._live(sid)
        text = str(text).strip()
        if not text:
            raise StoreError("statement text must not be empty")
        st = next(x for x in self.docs[home].statements if x.sid == sid)
        st.text = text
        st.flag = "edited"
        self.dirty.add(home)

    def link(self, a: str, b: str) -> None:
        da = self._doc(a)
        self._doc(b)
        if a == b:
            raise StoreError("cannot link a doc to itself")
        if b in da.links:
            raise StoreError(f"{a} already links to {b}")
        da.links.append(b)          # link changes never mark dirty (spec §1)

    def unlink(self, a: str, b: str) -> None:
        da = self._doc(a)
        if b not in da.links:
            raise StoreError(f"{a} does not link to {b}")
        da.links.remove(b)

    def create_doc(self, sids: list[str] = (), link_ids: list[str] = ()) -> str:
        for sid in sids:                      # validate everything first: atomic
            self._live(sid)
        for lid in link_ids:
            self._doc(lid)
        doc_id = f"d{self._next_doc:03d}"
        self._next_doc += 1
        doc = Doc(doc_id)
        self.docs[doc_id] = doc
        self.created_docs += 1
        self.dirty.add(doc_id)
        for sid in sids:
            self.copy_statement(sid, doc_id)
        doc.links = list(dict.fromkeys(link_ids))
        return doc_id

    def delete_doc(self, doc_id: str) -> int:
        doc = self._doc(doc_id)
        n = len(doc.statements)
        for st in doc.statements:             # instances die with the doc
            del self._home[st.sid]
        del self.docs[doc_id]
        for other in self.docs.values():      # inbound links removed everywhere
            if doc_id in other.links:
                other.links.remove(doc_id)
        self.dirty.discard(doc_id)
        self.deleted_docs += 1
        try:
            self._col().delete(ids=[doc_id])  # embedding row dropped
        except Exception:
            pass
        return n

    # ---------- sense ----------

    def _snippet(self, doc: Doc, query: str) -> str:
        """Best-matching single statement for the query — the 'matched chunk'
        a real vector DB would return. Scored by LEXICAL token overlap
        (normalized-token set intersection, first statement wins ties), not
        embeddings: doc-level ranking already lives in the embedding space,
        and a statement-level embedding index would have to track every
        copy/move/delete live, while this picks among the dozen statements
        of an already-retrieved doc. Truncated to ~120 chars."""
        q = set(normalize(query).split())
        best, best_score = "", -1
        for st in doc.statements:
            score = len(q & set(normalize(st.text).split()))
            if score > best_score:
                best, best_score = st.text, score
        if len(best) > SNIPPET_CHARS:
            best = best[:SNIPPET_CHARS - 3] + "..."
        return best

    def search(self, query: str, k: int = 5) -> list[tuple[str, str, str]]:
        """Top-k (doc_id, summary, snippet); ranking stays whole-doc."""
        col = self._col()
        n = min(k, col.count())
        if n <= 0:
            return []
        res = col.query(query_texts=[str(query)], n_results=n)
        return [(doc_id, self.docs[doc_id].summary,
                 self._snippet(self.docs[doc_id], query))
                for doc_id in res["ids"][0] if doc_id in self.docs]

    def read(self, doc_id: str) -> Doc:
        return self._doc(doc_id)

    # ---------- maintenance (iteration end only) ----------

    def refresh(self) -> int:
        """Regenerate every dirty doc's summary + embedding; returns how many."""
        n = 0
        for doc_id in sorted(self.dirty):
            doc = self.docs.get(doc_id)
            if doc is None:
                continue
            texts = [st.text for st in doc.statements]
            doc.summary = "(empty)" if not texts else self.summarizer.summarize(texts)
            self._col().upsert(ids=[doc_id], documents=[self._doc_text(doc_id)])
            n += 1
        self.dirty.clear()
        return n

    # ---------- stats (kb_stats.jsonl row, spec §8) ----------

    def origin_counts(self) -> dict[str, int]:
        """Live instances per origin, over origin-preserving UNEDITED
        instances only: authored ones have no origin, edited ones no longer
        carry their origin's text (T39.5)."""
        counts = {o: 0 for o in self.origins}
        for doc in self.docs.values():
            for st in doc.statements:
                if st.origin is not None and st.flag != "edited":
                    counts[st.origin] = counts.get(st.origin, 0) + 1
        return counts

    def stats(self) -> dict:
        counts = self.origin_counts()
        alive = sum(1 for o in self.origins if counts.get(o, 0) >= 1)
        inbound = {d: 0 for d in self.docs}
        for doc in self.docs.values():
            for lid in doc.links:
                if lid in inbound:
                    inbound[lid] += 1
        orphans = sum(1 for d, doc in self.docs.items()
                      if not doc.links and inbound[d] == 0)
        return {
            "docs": len(self.docs),
            "statements": sum(len(d.statements) for d in self.docs.values()),
            "origins_total": len(self.origins),
            "origins_alive": alive,
            "coverage": alive / len(self.origins) if self.origins else 0.0,
            "dup_origins": sum(1 for o in self.origins if counts.get(o, 0) > 1),
            "links": sum(len(d.links) for d in self.docs.values()),
            # approximate token count of all live statement text: total
            # chars // 4 (the usual ~4-chars-per-token heuristic; no tiktoken)
            "statement_tokens": sum(len(st.text) for d in self.docs.values()
                                    for st in d.statements) // 4,
            "orphan_docs": orphans,
            "created_docs": self.created_docs,
            "deleted_docs": self.deleted_docs,
            # live flagged instances (copies keep their flag), not cumulative
            "authored_statements": sum(1 for d in self.docs.values()
                                       for st in d.statements
                                       if st.flag == "authored"),
            "edited_statements": sum(1 for d in self.docs.values()
                                     for st in d.statements
                                     if st.flag == "edited"),
        }


def save_snapshot(store: Store, path, universe_path=None) -> None:
    """Per-epoch KB snapshot: docs JSON + the universe it belongs to, so
    kb.test can find the question file without extra flags."""
    with open(path, "w") as f:
        json.dump({"universe": str(universe_path) if universe_path else None,
                   "store": store.to_json()}, f, ensure_ascii=False)


def load_snapshot(path, summarizer=None, embedding_function=None):
    """(store, universe_path_or_None) from a save_snapshot file."""
    with open(path) as f:
        state = json.load(f)
    return (Store.from_json(state["store"], summarizer, embedding_function),
            state.get("universe"))
