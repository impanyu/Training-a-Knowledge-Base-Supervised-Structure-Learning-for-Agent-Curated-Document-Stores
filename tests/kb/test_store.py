import pytest

from kb.store import Store, StoreError

from .fixtures import HashEmbedding


def three_nodes():
    nodes = [{"id": "s0001", "text": "Alpha beta.", "origin": "s0001"},
             {"id": "s0002", "text": "Gamma delta.", "origin": "s0002"},
             {"id": "s0003", "text": "Epsilon zeta.", "origin": "s0003",
              "links": ["s0001"]}]
    return Store.from_nodes(nodes, HashEmbedding())


# ---------------- add / edit / delete + provenance ----------------

def test_add_new_id_no_origin_authored_flag():
    s = three_nodes()
    nid = s.add("Theta iota.")
    assert nid == "s0004"                      # globally unique, next in line
    n = s.nodes[nid]
    assert n.text == "Theta iota." and n.links == []
    assert n.origin is None and n.flag == "authored"
    assert s.dirty == {nid}
    stats = s.stats()
    assert stats["authored_statements"] == 1
    assert stats["n_nodes"] == 4               # live, but outside provenance:
    assert stats["origins_total"] == 3         # no new origin minted
    assert stats["origins_alive"] == 3 and stats["dup_origins"] == 0


def test_edit_keeps_id_and_origin_sets_edited_flag():
    s = three_nodes()
    s.edit("s0001", "Alpha rewritten.")
    n = s.nodes["s0001"]
    assert n.text == "Alpha rewritten."
    assert n.origin == "s0001" and n.flag == "edited"
    assert s.dirty == {"s0001"}
    stats = s.stats()
    assert stats["edited_statements"] == 1
    # coverage is over origin-preserving UNEDITED nodes only
    assert stats["origins_alive"] == 2
    assert stats["n_nodes"] == 3               # nothing died


def test_delete_cascades_inbound_links_and_is_coverage_loss():
    s = three_nodes()
    assert s.stats()["origins_alive"] == 3
    s.delete("s0001")
    assert "s0001" not in s.nodes
    assert s.nodes["s0003"].links == []        # inbound entry removed
    assert s.stats()["origins_alive"] == 2     # allowed, integer-exact
    assert all(nid != "s0001" for nid, _ in s.search("Alpha beta.", 5))
    with pytest.raises(StoreError):
        s.delete("s0001")                      # dead id
    with pytest.raises(StoreError):
        s.edit("s0001", "x")


def test_ids_never_reused_after_death():
    s = three_nodes()
    s.delete("s0003")
    assert s.add("Theta iota.") == "s0004"
    s.delete("s0004")
    assert s.add("Kappa lambda.") == "s0005"


def test_empty_text_is_rejected():
    s = three_nodes()
    with pytest.raises(StoreError):
        s.add("   ")
    with pytest.raises(StoreError):
        s.edit("s0001", "")
    assert s.dirty == set()


def test_editing_an_authored_note_flags_it_edited():
    s = three_nodes()
    nid = s.add("Theta iota.")
    s.edit(nid, "Theta rewritten.")
    n = s.nodes[nid]
    assert n.origin is None and n.flag == "edited"
    stats = s.stats()
    assert stats["authored_statements"] == 0
    assert stats["edited_statements"] == 1


# ---------------- links ----------------

def test_link_unlink_and_errors():
    s = three_nodes()
    s.link("s0001", "s0002")
    assert s.nodes["s0001"].links == ["s0002"]
    with pytest.raises(StoreError):
        s.link("s0001", "s0002")               # duplicate
    with pytest.raises(StoreError):
        s.link("s0001", "s0001")               # self
    with pytest.raises(StoreError):
        s.link("s0001", "s9999")               # unknown target
    s.unlink("s0001", "s0002")
    assert s.nodes["s0001"].links == []
    with pytest.raises(StoreError):
        s.unlink("s0001", "s0002")


def test_link_changes_never_mark_dirty():
    s = three_nodes()
    s.link("s0001", "s0002")
    s.unlink("s0003", "s0001")
    assert s.dirty == set()                    # the embedded text is unchanged


# ---------------- dirty marking + batch re-embedding ----------------

def test_refresh_reembeds_only_dirty_and_search_sees_the_edit():
    s = three_nodes()
    s.edit("s0003", "Omicron pi.")
    # not re-embedded until refresh: the OLD text still matches
    assert s.search("Epsilon zeta.", 1)[0][0] == "s0003"
    n = s.refresh()
    assert n == 1 and s.dirty == set()
    hits = dict(s.search("Omicron pi.", 3))
    assert hits["s0003"] == "Omicron pi."      # live text returned
    assert s.refresh() == 0                    # nothing dirty


def test_added_note_becomes_searchable_at_refresh():
    s = three_nodes()
    nid = s.add("Sigma tau upsilon.")
    s.refresh()
    assert s.search("Sigma tau upsilon.", 1)[0] == (nid, "Sigma tau upsilon.")


def test_search_returns_id_text_pairs_top_k():
    s = three_nodes()
    hits = s.search("Alpha beta.", 1)
    assert hits == [("s0001", "Alpha beta.")]
    assert len(s.search("Alpha beta.", 5)) == 3


def test_read_returns_the_node():
    s = three_nodes()
    n = s.read("s0003")
    assert n.text == "Epsilon zeta." and n.links == ["s0001"]
    with pytest.raises(StoreError):
        s.read("s9999")


# ---------------- stats ----------------

def test_stats_shape_on_the_node_graph():
    s = three_nodes()
    stats = s.stats()
    assert stats["n_nodes"] == 3 and stats["n_links"] == 1
    assert stats["coverage"] == 1.0 and stats["dup_origins"] == 0
    assert stats["orphan_nodes"] == 1          # s0002: no in/out links
    assert stats["statement_tokens"] == (11 + 12 + 13) // 4
    s.link("s0002", "s0001")
    assert s.stats()["orphan_nodes"] == 0


# ---------------- snapshot roundtrip ----------------

def test_json_roundtrip_preserves_everything_and_rebuilds_embeddings():
    s = three_nodes()
    s.add("Theta iota.")
    s.edit("s0002", "Gamma rewritten.")
    s.delete("s0001")
    s.link("s0002", "s0004")
    s.refresh()
    state = s.to_json()
    s2 = Store.from_json(state, HashEmbedding())
    assert s2.to_json() == state
    assert s2.stats() == s.stats()
    assert s2.stats()["authored_statements"] == 1
    assert s2.stats()["edited_statements"] == 1
    assert s2.add("Kappa lambda.") == "s0005"  # counters restored
    assert s2.search("Theta iota.", 5)         # embeddings rebuilt
