"""Prepared, NOT applied: search returns five results per page.

Applied only once the current chains have finished, because the exam runs as
a fresh process and would otherwise pick up an action set the training never
saw.

Why: with k up to sixty a reader can enumerate a whole set in one action,
which quietly removes the very cost an index exists to avoid. Paging keeps
enumeration possible but charges a step per page - six steps to walk
twenty-nine residents, against one read of a complete index - so the
asymmetry the paper measures is the real one.

    python3 patches/paged_search.py --apply
"""
import argparse
import re
from pathlib import Path

ACTIONS = Path("src/kb/actions.py")
LOOPS = Path("src/kb/loops.py")

OLD_SPEC = '''    "search": {
        "description": ("Search the knowledge base by meaning. Returns the "
                        f"k most similar notes as (id, text); k defaults to "
                        f"{SEARCH_K} and may be raised to {TOPK_MAX} when "
                        "five is plainly not the whole set - a city's "
                        "residents come back as the same five however often "
                        "you ask. A set is closed when raising k further "
                        "returns nothing new."),
        "input_schema": {"type": "object",
                         "properties": {"query": _S,
                                        "k": {"type": "integer"}},
                         "required": ["query"]},
    },'''

NEW_SPEC = '''    "search": {
        "description": ("Search the knowledge base by meaning. Returns "
                        f"{SEARCH_K} notes per page, most similar first. "
                        "page defaults to 1; ask for page 2, 3, ... to see "
                        "the next five. A set is closed when a page comes "
                        "back empty. Each page costs one action, so walking "
                        "a large set this way is expensive - that is the "
                        "cost an index exists to remove."),
        "input_schema": {"type": "object",
                         "properties": {"query": _S,
                                        "page": {"type": "integer"}},
                         "required": ["query"]},
    },'''

OLD_HANDLER = '''def _h_search(store, inp):
    """One search for reader and trainer alike. k is optional so the reader's
    default behaviour is unchanged, and raising it is what makes a set
    recoverable at all: five notes cannot enumerate twenty-nine residents."""
    try:
        k = int(inp.get("k", SEARCH_K))
    except (TypeError, ValueError):
        return "ERROR: k must be an integer"
    k = max(1, min(k, TOPK_MAX))
    hits = store.search(str(inp["query"]), k=k)
    return "\\n".join(f"- {nid}: {text}" for nid, text in hits) or "(no hits)"'''

NEW_HANDLER = '''def _h_search(store, inp):
    """One search for reader and trainer alike, five results per page.

    Enumerating a set is possible but priced: a page is an action, so the
    twenty-nine residents of a city cost six of them, while one read of a
    complete index costs one. That gap is what the trained store is supposed
    to close, and handing the reader sixty results at once would erase it."""
    try:
        page = int(inp.get("page", 1))
    except (TypeError, ValueError):
        return "ERROR: page must be an integer"
    page = max(1, min(page, TOPK_MAX // SEARCH_K))
    hits = store.search(str(inp["query"]), k=SEARCH_K * page)[SEARCH_K * (page - 1):]
    if not hits:
        return f"(page {page} is empty - no further matches)"
    body = "\\n".join(f"- {nid}: {text}" for nid, text in hits)
    return f"page {page}:\\n{body}"'''

OLD_RETRIEVAL = '''    "  search(query, k) - the notes whose text is most similar to your "
    "query; k defaults to five and may be raised to sixty. It is "
    "similarity, not understanding: it finds notes that SOUND like the "
    "query, so query with the words the note itself would use (a full "
    "name, an attribute word), not with the question. Five is enough to "
    "find one fact; when a question needs a whole set - everyone in a "
    "city, all of someone's friends - five is not all of them, so raise "
    "k until raising it further returns nothing new.\\n"'''

NEW_RETRIEVAL = '''    "  search(query, page) - five notes per page, most similar first; "
    "page defaults to 1 and each further page costs another action. It is "
    "similarity, not understanding: it finds notes that SOUND like the "
    "query, so query with the words the note itself would use (a full "
    "name, an attribute word), not with the question. Five is enough to "
    "find one fact. A whole set - everyone in a city, all of someone's "
    "friends - takes a page at a time, which is slow; if an index for "
    "that set already exists, one read of it is cheaper than paging.\\n"'''

OLD_CURATION = '''    "  - search(query, k) returns the k most similar notes, k defaulting "
    "to five and going up to sixty. Five cannot enumerate a set - the "
    "residents of a city come back as the same five however often you "
    "ask - so raise k when you need all of them, and treat the set as "
    "closed when raising it further returns nobody new.\\n"'''

NEW_CURATION = '''    "  - search(query, page) returns five notes per page, and a page is "
    "one action. Five cannot enumerate a set, so walk the pages until one "
    "comes back empty; that is what closing a set costs you, and it is "
    "exactly the cost a future reader will not have to pay once the index "
    "exists.\\n"'''

OLD_EX = '''search(\\"lives in the city of "
    "Fenmarch\\", k=40) returns twenty-nine residents and nothing else, "
    "and k=60 returns the same twenty-nine, so the set is closed; "'''

NEW_EX = '''search(\\"lives in the city of "
    "Fenmarch\\") page by page returns twenty-nine residents over six "
    "pages and the seventh comes back empty, so the set is closed; "'''

OLD_EX2 = '''search(\\"job is cooper\\", k=40) -> add(\\"People whose job is "'''
NEW_EX2 = '''search(\\"job is cooper\\") -> add(\\"People whose job is "'''


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    a = ap.parse_args()
    edits = [(ACTIONS, OLD_SPEC, NEW_SPEC), (ACTIONS, OLD_HANDLER, NEW_HANDLER),
             (LOOPS, OLD_RETRIEVAL, NEW_RETRIEVAL), (LOOPS, OLD_CURATION, NEW_CURATION),
             (LOOPS, OLD_EX, NEW_EX), (LOOPS, OLD_EX2, NEW_EX2)]
    missing = [old[:48] for path, old, _ in edits if old not in path.read_text()]
    if missing:
        raise SystemExit("anchors not found:\\n  " + "\\n  ".join(missing))
    print(f"all {len(edits)} anchors present")
    if not a.apply:
        print("dry run; pass --apply to write")
        return
    for path, old, new in edits:
        path.write_text(path.read_text().replace(old, new, 1))
    print("applied")


if __name__ == "__main__":
    main()
