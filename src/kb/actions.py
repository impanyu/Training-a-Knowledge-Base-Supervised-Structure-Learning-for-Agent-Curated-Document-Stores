"""The 11 actions (v9.6 node graph): ONE static tool schema, no per-node
variation, no dynamic schema. Phase gating is by schema subset per mode
(MODE_TOOLS), never by runtime rejection; the loop keeps a runtime error
string only as a safety net for scripted policies that go off-mode.

flow:  answer(text) . done()            -- handled by the loop, not dispatch
sense: search(query) -> top-k (id, full text)     [k=5]
       read(id)      -> text + links (target id, target full text)
edit:  add(text) / edit(id, text) / delete(id) / link(a, b)
       link_many(a, targets) / unlink(a, b)
       (train Phase 2 ONLY; Phase 1 and test keep search/read/answer)

Invalid / dead ids -> "ERROR: ..." result, recorded like any result."""
from kb.store import Store, StoreError

SEARCH_K = 5
KEYWORD_CAP = 200        # curation only: guard against a keyword matching the store


def _schema(props: dict, required: list[str]) -> dict:
    return {"type": "object", "properties": props, "required": required}


_S = {"type": "string"}
_SLIST = {"type": "array", "items": {"type": "string"}}

ACTION_SPECS: dict[str, dict] = {
    # -------- flow --------
    "answer": {
        "description": ("Submit your ONE final answer to the question: a short "
                        "answer (a name / phrase / number), never a sentence. "
                        'Say "unknown" if the notes cannot determine it. '
                        "Submitting ends this phase immediately."),
        "input_schema": _schema({"text": _S}, ["text"]),
    },
    "done": {
        "description": "Declare consolidation finished; ends this phase.",
        "input_schema": _schema({}, []),
    },
    # -------- sense --------
    "search": {
        "description": ("Search the knowledge base by meaning; returns the "
                        f"top-{SEARCH_K} notes as (id, text)."),
        "input_schema": _schema({"query": _S}, ["query"]),
    },
    "search_keyword": {
        "description": ("EVERY note whose text contains all of the given "
                        "keywords, matched literally and case-insensitively "
                        "- not a similarity ranking and not a top-k. This is "
                        "how you enumerate a set completely: one call returns "
                        "all of it, and the count tells you the set is "
                        "closed. Available while curating only; the reader "
                        "you are building for has similarity search alone, "
                        "which is why it needs the structure you leave."),
        "input_schema": _schema({"keywords": _SLIST}, ["keywords"]),
    },
    "read": {
        "description": ("Read one note: its text and its links, each rendered "
                        "as (target id, target text)."),
        "input_schema": _schema({"id": _S}, ["id"]),
    },
    # -------- edit --------
    "add": {
        "description": ("Author a NEW note: one short self-contained sentence "
                        "(full names, no pronouns). It gets a new id and is "
                        "flagged as authored. A note stating the same fact as "
                        "an existing note is merged instead of created."),
        "input_schema": _schema({"text": _S}, ["text"]),
    },
    "edit": {
        "description": ("Rewrite an existing note's text in place (it keeps "
                        "its id and is flagged as edited). An edit that "
                        "duplicates another note merges into it."),
        "input_schema": _schema({"id": _S, "text": _S}, ["id", "text"]),
    },
    "delete": {
        "description": ("Delete a note; links to it are removed from all "
                        "other notes."),
        "input_schema": _schema({"id": _S}, ["id"]),
    },
    "link": {
        "description": "Add a directed link from note a to note b (ids only).",
        "input_schema": _schema({"a": _S, "b": _S}, ["a", "b"]),
    },
    "link_many": {
        "description": ("Link one note to MANY notes in a single action: the "
                        "cheap way to build or extend an index, since "
                        "populating one costs a single step rather than one "
                        "per member. Ids that do not exist are reported and "
                        "skipped; the rest are linked."),
        "input_schema": _schema({"a": _S, "targets": _SLIST}, ["a", "targets"]),
    },
    "unlink": {
        "description": "Remove the link from note a to note b.",
        "input_schema": _schema({"a": _S, "b": _S}, ["a", "b"]),
    },
}

EDIT_ACTIONS = ("add", "edit", "delete", "link", "link_many", "unlink")

# Phase/tool gating by schema subset: the model never sees an out-of-phase
# tool, so gating needs no runtime rejection.
MODE_TOOLS: dict[str, tuple[str, ...]] = {
    "train_forward": ("search", "read", "answer"),
    "train_backward": ("search", "search_keyword", "read") + EDIT_ACTIONS + ("done",),
    "test": ("search", "read", "answer"),
}


def tools_for(mode: str) -> list[dict]:
    return [{"name": n, **ACTION_SPECS[n]} for n in MODE_TOOLS[mode]]


def dispatch(store: Store, name: str, inp: dict) -> str:
    """Sense + edit actions only; answer/done are flow and belong to the loop."""
    if name not in _HANDLERS:
        return f"ERROR: unknown action {name}"
    try:
        return _HANDLERS[name](store, inp)
    except StoreError as e:
        return f"ERROR: {e}"
    except (KeyError, TypeError, ValueError) as e:
        return f"ERROR: bad input: {e}"


# ---------------- handlers ----------------

def _h_search(store, inp):
    hits = store.search(str(inp["query"]), k=SEARCH_K)
    return "\n".join(f"- {nid}: {text}" for nid, text in hits) or "(no results)"


def _h_search_keyword(store, inp):
    """Exact substring match over every note, so a set can be closed in one
    call. Similarity search cannot do this: the same query returns the same
    five notes however often it is asked."""
    kws = [str(k).strip().lower() for k in (inp.get("keywords") or []) if str(k).strip()]
    if not kws:
        return "ERROR: keywords is empty"
    hits = [(nid, n.text) for nid, n in store.nodes.items()
            if all(k in n.text.lower() for k in kws)]
    if not hits:
        return "(no notes contain all of: " + ", ".join(kws) + ")"
    head = sorted(hits)[:KEYWORD_CAP]
    body = "\n".join(f"- {nid}: {text}" for nid, text in head)
    if len(hits) > KEYWORD_CAP:
        return (f"{len(hits)} notes match; showing the first {KEYWORD_CAP}. "
                f"Narrow the keywords.\n{body}")
    return f"{len(hits)} notes match, all shown.\n{body}"


def _h_read(store, inp):
    node = store.read(str(inp["id"]).strip())
    lines = [f"{node.nid}: {node.text}", "links:"]
    lines += [f"- {lid}: {store.nodes[lid].text}"
              for lid in node.links] or ["(none)"]
    return "\n".join(lines)


def _h_add(store, inp):
    nid, merged = store.add(str(inp["text"]))
    if merged is not None:      # informative, deliberately NOT an ERROR
        return f"duplicate of {merged} - merged, no new note created"
    return f"added {nid}"


def _h_edit(store, inp):
    nid = str(inp["id"]).strip()
    merged = store.edit(nid, str(inp["text"]))
    if merged is not None:
        return f"merged into {merged}"
    return f"edited {nid}"


def _h_delete(store, inp):
    nid = str(inp["id"]).strip()
    store.delete(nid)
    return f"deleted {nid}"


def _h_link(store, inp):
    a, b = str(inp["a"]).strip(), str(inp["b"]).strip()
    store.link(a, b)
    return f"linked {a} -> {b}"


def _h_link_many(store, inp):
    """One action, many edges. Partial failure is reported, not fatal: an
    index built from search results will sometimes name an id that has since
    been merged or deleted, and losing the whole batch for that would make
    the cheap path the risky one."""
    a = str(inp["a"]).strip()
    targets = [str(t).strip() for t in (inp.get("targets") or [])]
    if not targets:
        return "ERROR: targets is empty"
    ok, bad = [], []
    for t in targets:
        try:
            store.link(a, t)
            ok.append(t)
        except StoreError as e:
            bad.append(f"{t} ({e})")
    out = f"linked {a} -> {len(ok)} notes: {', '.join(ok)}" if ok else \
        f"no links made from {a}"
    return out + (f" | skipped: {'; '.join(bad)}" if bad else "")


def _h_unlink(store, inp):
    a, b = str(inp["a"]).strip(), str(inp["b"]).strip()
    store.unlink(a, b)
    return f"unlinked {a} -> {b}"


_HANDLERS = {
    "search": _h_search, "search_keyword": _h_search_keyword, "read": _h_read,
    "add": _h_add, "edit": _h_edit, "delete": _h_delete,
    "link": _h_link, "link_many": _h_link_many, "unlink": _h_unlink,
}
