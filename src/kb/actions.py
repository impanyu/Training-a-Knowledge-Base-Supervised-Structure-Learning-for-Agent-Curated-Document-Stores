"""The 11 actions (spec §2): ONE static tool schema, no per-doc variation,
no dynamic schema. Phase gating is by schema subset per mode (MODE_TOOLS),
never by runtime rejection; the loop keeps a runtime error string only as a
safety net for scripted policies that go off-mode.

flow:  answer(text) . done()            -- handled by the loop, not dispatch
sense: search(query) -> top-k (doc_id, summary)   [k=5]
       read(doc_id)  -> statements (sid: text) + links (doc_id, live summary)
edit:  copy_statement / move_statement / delete_statement / link / unlink /
       create_doc / delete_doc

Invalid ids / dead sids -> "ERROR: ..." result, recorded like any result."""
from kb.store import Store, StoreError

SEARCH_K = 5


def _schema(props: dict, required: list[str]) -> dict:
    return {"type": "object", "properties": props, "required": required}


_S = {"type": "string"}
_SL = {"type": "array", "items": {"type": "string"}}

ACTION_SPECS: dict[str, dict] = {
    # -------- flow --------
    "answer": {
        "description": ("Submit your ONE final answer to the question: a short "
                        "answer (a name / phrase / number), never a sentence. "
                        'Say "unknown" if the documents cannot determine it. '
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
                        f"top-{SEARCH_K} (doc_id, summary) pairs."),
        "input_schema": _schema({"query": _S}, ["query"]),
    },
    "read": {
        "description": ("Read one doc in full: its statements (sid: text) and "
                        "its links (doc_id, current summary)."),
        "input_schema": _schema({"doc_id": _S}, ["doc_id"]),
    },
    # -------- edit --------
    "copy_statement": {
        "description": ("Copy a statement instance into another doc (new sid, "
                        "same origin; the source doc keeps its copy)."),
        "input_schema": _schema({"sid": _S, "to_doc": _S}, ["sid", "to_doc"]),
    },
    "move_statement": {
        "description": ("Move a statement instance to another doc (atomic "
                        "copy+delete; it keeps its sid)."),
        "input_schema": _schema({"sid": _S, "to_doc": _S}, ["sid", "to_doc"]),
    },
    "delete_statement": {
        "description": ("Delete this statement instance (copies of it in other "
                        "docs survive)."),
        "input_schema": _schema({"sid": _S}, ["sid"]),
    },
    "link": {
        "description": "Add a directed link from doc a to doc b (ids only).",
        "input_schema": _schema({"a": _S, "b": _S}, ["a", "b"]),
    },
    "unlink": {
        "description": "Remove the link from doc a to doc b.",
        "input_schema": _schema({"a": _S, "b": _S}, ["a", "b"]),
    },
    "create_doc": {
        "description": ("Create a new doc; optionally bulk-copy statements into "
                        "it (sids) and link it to docs (link_ids). Its summary "
                        "and embedding are generated automatically."),
        "input_schema": _schema({"sids": _SL, "link_ids": _SL}, []),
    },
    "delete_doc": {
        "description": ("Delete a doc: its statement instances die with it and "
                        "inbound links to it are removed from all other docs."),
        "input_schema": _schema({"doc_id": _S}, ["doc_id"]),
    },
}

EDIT_ACTIONS = ("copy_statement", "move_statement", "delete_statement",
                "link", "unlink", "create_doc", "delete_doc")

# Phase/tool gating by schema subset (spec §4): the model never sees an
# out-of-phase tool, so gating needs no runtime rejection.
MODE_TOOLS: dict[str, tuple[str, ...]] = {
    "train_forward": ("search", "read", "answer"),
    "train_backprop": ("search", "read") + EDIT_ACTIONS + ("done",),
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
    return "\n".join(f"- {d}: {s}" for d, s in hits) or "(no results)"


def _h_read(store, inp):
    doc = store.read(str(inp["doc_id"]).strip())
    lines = [f"{doc.doc_id} | summary: {doc.summary}", "statements:"]
    lines += [f"- {st.sid}: {st.text}" for st in doc.statements] or ["(none)"]
    lines.append("links:")
    lines += [f"- {lid}: {store.docs[lid].summary}" for lid in doc.links] or ["(none)"]
    return "\n".join(lines)


def _h_copy(store, inp):
    sid, to_doc = str(inp["sid"]).strip(), str(inp["to_doc"]).strip()
    new = store.copy_statement(sid, to_doc)
    return f"copied {sid} -> {new} in {to_doc}"


def _h_move(store, inp):
    sid, to_doc = str(inp["sid"]).strip(), str(inp["to_doc"]).strip()
    store.move_statement(sid, to_doc)
    return f"moved {sid} to {to_doc}"


def _h_delete_statement(store, inp):
    sid = str(inp["sid"]).strip()
    store.delete_statement(sid)
    return f"deleted {sid}"


def _h_link(store, inp):
    a, b = str(inp["a"]).strip(), str(inp["b"]).strip()
    store.link(a, b)
    return f"linked {a} -> {b}"


def _h_unlink(store, inp):
    a, b = str(inp["a"]).strip(), str(inp["b"]).strip()
    store.unlink(a, b)
    return f"unlinked {a} -> {b}"


def _h_create_doc(store, inp):
    sids = [str(s).strip() for s in (inp.get("sids") or [])]
    link_ids = [str(x).strip() for x in (inp.get("link_ids") or [])]
    doc_id = store.create_doc(sids, link_ids)
    doc = store.docs[doc_id]
    return f"created {doc_id} ({len(doc.statements)} statements, {len(doc.links)} links)"


def _h_delete_doc(store, inp):
    doc_id = str(inp["doc_id"]).strip()
    n = store.delete_doc(doc_id)
    return f"deleted {doc_id} ({n} statement instances died)"


_HANDLERS = {
    "search": _h_search, "read": _h_read,
    "copy_statement": _h_copy, "move_statement": _h_move,
    "delete_statement": _h_delete_statement,
    "link": _h_link, "unlink": _h_unlink,
    "create_doc": _h_create_doc, "delete_doc": _h_delete_doc,
}
