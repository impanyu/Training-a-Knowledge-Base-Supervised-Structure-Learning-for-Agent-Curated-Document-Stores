import pytest

from kb.store import Store, StoreError

from .fixtures import HashEmbedding


def three_nodes(judge=None, threshold=0.90):
    nodes = [{"id": "s0001", "text": "Alpha beta.", "origin": "s0001"},
             {"id": "s0002", "text": "Gamma delta.", "origin": "s0002"},
             {"id": "s0003", "text": "Epsilon zeta.", "origin": "s0003",
              "links": ["s0001"]}]
    return Store.from_nodes(nodes, HashEmbedding(), judge=judge,
                            dedup_threshold=threshold)


class SpyJudge:
    """Deterministic duplicate-judge stub recording every pair it saw."""

    def __init__(self, verdict: bool = True):
        self.calls: list[tuple[str, str]] = []
        self.verdict = verdict

    def __call__(self, a: str, b: str) -> bool:
        self.calls.append((a, b))
        return self.verdict


# ---------------- add / edit / delete + provenance ----------------

def test_add_new_id_no_origin_authored_flag():
    s = three_nodes()
    nid, merged = s.add("Theta iota.")
    assert (nid, merged) == ("s0004", None)    # globally unique, next in line
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
    assert s.add("Theta iota.") == ("s0004", None)
    s.delete("s0004")
    assert s.add("Kappa lambda.") == ("s0005", None)


def test_empty_text_is_rejected():
    s = three_nodes()
    with pytest.raises(StoreError):
        s.add("   ")
    with pytest.raises(StoreError):
        s.edit("s0001", "")
    assert s.dirty == set()


def test_editing_an_authored_note_flags_it_edited():
    s = three_nodes()
    nid, _ = s.add("Theta iota.")
    s.edit(nid, "Theta rewritten.")
    n = s.nodes[nid]
    assert n.origin is None and n.flag == "edited"
    stats = s.stats()
    assert stats["authored_statements"] == 0
    assert stats["edited_statements"] == 1


# ---------------- infrastructure dedup (T42) ----------------

def test_add_duplicate_is_merged_not_created():
    judge = SpyJudge(True)
    s = three_nodes(judge=judge)
    nid, merged = s.add("Alpha beta.")           # exact text of s0001: cos 1.0
    assert nid is None and merged == "s0001"
    stats = s.stats()
    assert stats["n_nodes"] == 3                 # nothing was created
    assert stats["merges"] == 1
    assert stats["authored_statements"] == 0
    assert judge.calls == [("Alpha beta.", "Alpha beta.")]
    assert s.nodes["s0001"].absorbed == []       # an add has nothing to absorb
    assert s.add("Novel quokka fact.")[0] == "s0004"   # id was not burned


def test_below_threshold_never_calls_the_judge():
    judge = SpyJudge(True)
    s = three_nodes(judge=judge)
    nid, merged = s.add("Quixotic zephyr vortex.")
    assert merged is None and nid == "s0004"     # normal accept
    assert judge.calls == []                     # best cosine below threshold


def test_judge_says_no_accepts_the_note():
    judge = SpyJudge(False)
    s = three_nodes(judge=judge)
    nid, merged = s.add("Alpha beta.")
    assert merged is None and nid == "s0004"
    assert len(judge.calls) == 1                 # consulted, said no
    assert s.stats()["merges"] == 0


def test_no_judge_means_dedup_off():
    s = three_nodes()                            # judge None
    nid, merged = s.add("Alpha beta.")           # exact duplicate accepted
    assert merged is None and nid == "s0004"
    assert s.stats()["merges"] == 0


def test_edit_into_duplicate_merges_links_and_provenance():
    judge = SpyJudge(True)
    s = three_nodes(judge=judge)
    # X = s0002 gets an inbound link from s0003 (which ALSO links the
    # survivor already -> rewiring must dedupe) and an outbound link
    s.link("s0003", "s0002")
    s.link("s0002", "s0003")
    merged = s.edit("s0002", "Alpha beta.")      # now a duplicate of s0001
    assert merged == "s0001"
    assert "s0002" not in s.nodes
    y = s.nodes["s0001"]
    assert y.absorbed == ["s0002"]               # merged-away origin carried
    assert y.links == ["s0003"]                  # outbound unioned into Y
    assert s.nodes["s0003"].links == ["s0001"]   # inbound rewired + deduped
    assert judge.calls == [("Alpha beta.", "Alpha beta.")]
    stats = s.stats()
    assert stats["merges"] == 1 and stats["n_nodes"] == 2
    assert stats["origins_alive"] == 3           # s0002 alive via absorbed
    assert stats["dup_origins"] == 0


def test_absorbed_chains_through_a_second_merge():
    judge = SpyJudge(True)
    s = three_nodes(judge=judge)
    s.edit("s0002", "Alpha beta.")               # s0002 merges into s0001
    s.edit("s0001", "Epsilon zeta.")             # s0001 merges into s0003
    y = s.nodes["s0003"]
    assert y.absorbed == ["s0001", "s0002"]      # origin + what it had absorbed
    stats = s.stats()
    assert stats["n_nodes"] == 1 and stats["merges"] == 2
    assert stats["origins_alive"] == 3           # all three live via s0003


def test_coverage_rule_carrier_and_absorbed_both_count():
    s = three_nodes(judge=SpyJudge(True))
    s.edit("s0002", "Alpha beta.")
    # s0001 carries its own origin unedited AND lists s0002 as absorbed
    assert s.origin_counts() == {"s0001": 1, "s0002": 1, "s0003": 1}
    s.delete("s0001")                            # absorbed dies with survivor
    assert s.stats()["origins_alive"] == 1


def test_merged_state_snapshot_roundtrip():
    s = three_nodes(judge=SpyJudge(True))
    s.link("s0003", "s0002")
    s.edit("s0002", "Alpha beta.")
    s.refresh()
    state = s.to_json()
    assert state["merges"] == 1
    s2 = Store.from_json(state, HashEmbedding())
    assert s2.to_json() == state                 # absorbed + merges persist
    assert s2.stats() == s.stats()


def test_dedup_is_deterministic():
    def run():
        s = three_nodes(judge=SpyJudge(True))
        s.add("Alpha beta.")                     # merged, no state change
        s.link("s0003", "s0002")
        s.edit("s0002", "Epsilon zeta.")         # merged into s0003
        return s.to_json()
    assert run() == run()


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

def test_a_read_of_the_index_never_sees_stale_embeddings():
    """Writes batch, but searching flushes first. Deferring past a read made
    an edit invisible to the very next query."""
    s = three_nodes()
    s.edit("s0003", "Omicron pi.")
    assert s.dirty == {"s0003"}                # queued, not yet embedded
    hits = dict(s.search("Omicron pi.", 3))
    assert hits["s0003"] == "Omicron pi."      # the search flushed it
    assert s.dirty == set()
    assert s.refresh() == 0                    # nothing left over


def test_added_note_is_searchable_immediately():
    s = three_nodes()
    nid, _ = s.add("Sigma tau upsilon.")
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
    assert s2.add("Kappa lambda.") == ("s0005", None)  # counters restored
    assert s2.search("Theta iota.", 5)         # embeddings rebuilt
