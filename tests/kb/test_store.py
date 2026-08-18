import pytest

from kb.store import Store, StoreError, TemplateSummarizer

from .fixtures import CountingSummarizer, HashEmbedding


def two_docs(summarizer=None):
    docs = [{"id": "d001", "summary": "about alpha",
             "statements": [{"sid": "s0001", "text": "Alpha beta.", "origin": "s0001"},
                            {"sid": "s0002", "text": "Gamma delta.", "origin": "s0002"}],
             "links": []},
            {"id": "d002", "summary": "about epsilon",
             "statements": [{"sid": "s0003", "text": "Epsilon zeta.", "origin": "s0003"}],
             "links": ["d001"]}]
    return Store.from_docs(docs, summarizer or CountingSummarizer(), HashEmbedding())


# ---------------- copy / move / delete + origin bookkeeping ----------------

def test_copy_new_sid_same_origin_source_keeps_instance():
    s = two_docs()
    new = s.copy_statement("s0001", "d002")
    assert new == "s0004"                      # globally unique, next in line
    assert [st.sid for st in s.docs["d001"].statements] == ["s0001", "s0002"]
    copied = next(st for st in s.docs["d002"].statements if st.sid == new)
    assert copied.origin == "s0001" and copied.text == "Alpha beta."
    assert s.dirty == {"d002"}
    assert s.stats()["dup_origins"] == 1       # >1 live instance of s0001


def test_move_keeps_sid_marks_both_dirty():
    s = two_docs()
    s.move_statement("s0001", "d002")
    assert all(st.sid != "s0001" for st in s.docs["d001"].statements)
    assert any(st.sid == "s0001" for st in s.docs["d002"].statements)
    assert s.dirty == {"d001", "d002"}
    assert s.stats()["dup_origins"] == 0       # atomic copy+delete, no dup


def test_move_to_own_doc_errors():
    s = two_docs()
    with pytest.raises(StoreError):
        s.move_statement("s0001", "d001")


def test_delete_statement_last_instance_is_coverage_loss():
    s = two_docs()
    assert s.stats()["origins_alive"] == 3
    s.delete_statement("s0002")
    assert s.stats()["origins_alive"] == 2     # allowed, integer-exact
    assert s.dirty == {"d001"}
    with pytest.raises(StoreError):
        s.delete_statement("s0002")            # dead sid


def test_copy_of_a_copy_still_descends_from_the_original():
    s = two_docs()
    first = s.copy_statement("s0001", "d002")
    s.delete_statement("s0001")                # original instance dies
    second = s.copy_statement(first, "d001")
    st = next(st for st in s.docs["d001"].statements if st.sid == second)
    assert st.origin == "s0001"
    assert s.stats()["origins_alive"] == 3


def test_sids_never_reused_after_death():
    s = two_docs()
    s.delete_statement("s0003")
    assert s.copy_statement("s0001", "d002") == "s0004"


# ---------------- links ----------------

def test_link_unlink_and_errors():
    s = two_docs()
    s.link("d001", "d002")
    assert s.docs["d001"].links == ["d002"]
    with pytest.raises(StoreError):
        s.link("d001", "d002")                 # duplicate
    with pytest.raises(StoreError):
        s.link("d001", "d001")                 # self
    with pytest.raises(StoreError):
        s.link("d001", "d999")                 # unknown target
    s.unlink("d001", "d002")
    assert s.docs["d001"].links == []
    with pytest.raises(StoreError):
        s.unlink("d001", "d002")


def test_link_changes_never_mark_dirty():
    s = two_docs()
    s.link("d001", "d002")
    s.unlink("d002", "d001")
    assert s.dirty == set()


# ---------------- create_doc / delete_doc ----------------

def test_create_doc_empty_gets_empty_summary_on_refresh():
    s = two_docs()
    did = s.create_doc()
    assert did == "d003" and s.docs[did].summary == "(pending)"
    assert did in s.dirty
    s.refresh()
    assert s.docs[did].summary == "(empty)"
    assert s.summarizer.calls == 0             # no summarizer call for empty


def test_create_doc_bulk_copies_and_links():
    s = two_docs()
    did = s.create_doc(sids=["s0001", "s0003"], link_ids=["d001", "d001"])
    doc = s.docs[did]
    assert [st.origin for st in doc.statements] == ["s0001", "s0003"]
    assert [st.sid for st in doc.statements] == ["s0004", "s0005"]
    assert doc.links == ["d001"]               # deduped
    assert s.created_docs == 1


def test_create_doc_invalid_sid_is_atomic():
    s = two_docs()
    with pytest.raises(StoreError):
        s.create_doc(sids=["s0001", "s9999"])
    assert "d003" not in s.docs
    assert s.created_docs == 0
    assert s.create_doc() == "d003"            # id was not burned


def test_delete_doc_cascade():
    s = two_docs()
    s.link("d001", "d002")
    n = s.delete_doc("d002")
    assert n == 1
    assert "d002" not in s.docs
    assert s.docs["d001"].links == []          # inbound entry removed
    assert s.stats()["origins_alive"] == 2     # s0003's last instance died
    assert s.deleted_docs == 1
    with pytest.raises(StoreError):
        s.delete_statement("s0003")            # instances died with the doc
    assert all(d != "d002" for d, _ in s.search("Epsilon zeta.", 5))


def test_doc_ids_never_reused_after_delete():
    s = two_docs()
    s.create_doc()
    s.delete_doc("d003")
    assert s.create_doc() == "d004"


# ---------------- dirty marking + batch regeneration ----------------

def test_summaries_regenerate_only_at_refresh_and_only_dirty():
    s = two_docs()
    s.copy_statement("s0003", "d001")
    assert s.docs["d001"].summary == "about alpha"   # unchanged until refresh
    assert s.summarizer.calls == 0
    n = s.refresh()
    assert n == 1 and s.summarizer.calls == 1        # d001 only, batch at end
    assert s.docs["d001"].summary.startswith("[v1] 3 statements")
    assert s.docs["d002"].summary == "about epsilon"
    assert s.dirty == set()
    assert s.refresh() == 0                          # nothing dirty, no calls


def test_refresh_updates_embedding_and_search_sees_live_summary():
    s = two_docs()
    s.move_statement("s0003", "d001")
    s.refresh()
    hits = dict(s.search("Epsilon zeta.", 5))
    assert hits["d001"].startswith("[v")             # live summary returned


def test_summary_is_never_agent_settable():
    s = two_docs()
    # No edit path takes summary text: only refresh() writes it, via the
    # injectable summarizer. The public surface exposes no setter.
    assert not hasattr(s, "set_summary")
    before = s.docs["d001"].summary
    s.link("d001", "d002")
    assert s.docs["d001"].summary == before


def test_search_top_k_pairs():
    s = two_docs()
    hits = s.search("Alpha beta.", 1)
    assert hits == [("d001", "about alpha")]


# ---------------- snapshot roundtrip ----------------

def test_json_roundtrip_preserves_everything_and_rebuilds_embeddings():
    s = two_docs()
    s.copy_statement("s0001", "d002")
    s.delete_statement("s0002")
    s.create_doc(sids=["s0003"])
    s.link("d003", "d001")
    s.refresh()
    state = s.to_json()
    s2 = Store.from_json(state, CountingSummarizer(), HashEmbedding())
    assert s2.to_json() == state
    assert s2.stats() == s.stats()
    assert s2.copy_statement("s0001", "d003") == "s0006"   # counters restored
    assert s2.create_doc() == "d004"
    assert dict(s2.search("Alpha beta.", 5))               # embeddings rebuilt


def test_template_summarizer_names_the_subject():
    t = TemplateSummarizer()
    assert t.summarize(["Alice Johnson's job is arborist."]) == "Facts about Alice Johnson."
    assert t.summarize(["Alice Johnson is married to Bob Meza."]) == "Facts about Alice Johnson."
