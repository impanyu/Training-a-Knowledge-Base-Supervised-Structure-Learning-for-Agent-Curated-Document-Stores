import pytest
from fixtures import demo_library, demo_posted

from ca.economy import Ledger
from ca.taskboard import BoardError, TaskBoard
from ca.tasktree import TaskLibrary, TaskNode

FULL_T1 = {"q0001": "Paris", "q0002": "Loire", "q0003": "4"}


def setup():
    led = Ledger({"a": 10, "b": 10})
    return led, TaskBoard(demo_library(), demo_posted(), led)


def test_posting_a_zero_leaf_node_is_rejected():
    nodes = [TaskNode("t0009", "an empty task with no leaves at all", [])]
    lib = TaskLibrary(nodes, [])
    led = Ledger({"a": 10})
    with pytest.raises(ValueError):
        TaskBoard(lib, ["t0009"], led)


def test_deliver_defends_against_a_dangling_question_ref():
    """Belt-and-suspenders: even though TaskLibrary now validates dangling
    refs at construction, deliver() must not corrupt a task's leaves list
    with a partial settlement if a leaf id has no matching question."""
    led, tb = setup()
    tb.claim("a", "t0001")
    del tb.library.questions["q0002"]           # simulate library corruption
    with pytest.raises(BoardError):
        tb.deliver("a", "t0001", FULL_T1)
    assert tb.tasks["t0001"].status == "claimed"
    assert tb.tasks["t0001"].leaves == []


def test_only_posted_nodes_are_tasks():
    led, tb = setup()
    assert set(tb.tasks) == {"t0001", "t0004"}
    with pytest.raises(BoardError):
        tb.claim("a", "t0002")           # a library node that was never posted
    with pytest.raises(BoardError):
        tb.claim("a", "t9999")


def test_claim_hides_from_open_list_and_blocks_others():
    led, tb = setup()
    tb.claim("a", "t0001")
    assert [t.nid for t in tb.list_open()] == ["t0004"]
    with pytest.raises(BoardError, match="already claimed by another agent"):
        tb.claim("b", "t0001")


def test_claim_accepts_the_sentence_as_well_as_the_id():
    led, tb = setup()
    t = tb.claim("a", "Answer the French geography questions.")
    assert t.nid == "t0001"


def test_list_open_is_sorted_by_price_descending():
    led, tb = setup()
    assert [t.nid for t in tb.list_open()] == ["t0004", "t0001"]   # 700, 600


def test_packaged_delivery_grades_every_leaf_and_pays_the_sum():
    led, tb = setup()
    tb.claim("a", "t0001")
    per_leaf, total = tb.deliver("a", "t0001", FULL_T1)
    assert [(qid, round(f1, 3), pay) for qid, f1, pay in per_leaf] == [
        ("q0001", 1.0, 100), ("q0002", 1.0, 200), ("q0003", 1.0, 300)]
    assert total == 600
    assert led.balance("a") == 610
    assert led.conservation_ok()
    assert tb.tasks["t0001"].status == "closed"


def test_partial_and_wrong_leaves_are_paid_proportionally():
    led, tb = setup()
    tb.claim("a", "t0001")
    per_leaf, total = tb.deliver("a", "t0001",
                                 {"q0001": "Paris", "q0002": "Loire river", "q0003": "5"})
    pay = {qid: p for qid, _, p in per_leaf}
    assert pay["q0001"] == 100
    assert pay["q0002"] == 133           # round(200 * 2/3) on a 1-of-2 token overlap
    assert pay["q0003"] == 0
    assert total == 233 and led.balance("a") == 243


def test_missing_leaf_is_rejected_without_consuming_the_attempt():
    led, tb = setup()
    tb.claim("a", "t0001")
    with pytest.raises(BoardError) as e:
        tb.deliver("a", "t0001", {"q0001": "Paris", "q0002": "Loire"})
    assert "q0003" in str(e.value)
    assert tb.tasks["t0001"].status == "claimed" and tb.tasks["t0001"].claimed_by == "a"
    assert led.balance("a") == 10        # nothing paid
    # the attempt survives: a complete map still settles
    _, total = tb.deliver("a", "t0001", FULL_T1)
    assert total == 600


def test_extra_qid_is_rejected_without_consuming_the_attempt():
    led, tb = setup()
    tb.claim("a", "t0001")
    with pytest.raises(BoardError) as e:
        tb.deliver("a", "t0001", dict(FULL_T1, q0004="6"))
    assert "q0004" in str(e.value)
    assert tb.tasks["t0001"].status == "claimed"
    assert tb.deliver("a", "t0001", FULL_T1)[1] == 600


def test_complete_but_wrong_map_still_consumes_the_only_attempt():
    led, tb = setup()
    tb.claim("a", "t0001")
    _, total = tb.deliver("a", "t0001", {"q0001": "x", "q0002": "y", "q0003": "z"})
    assert total == 0
    with pytest.raises(BoardError):
        tb.deliver("a", "t0001", FULL_T1)          # one shot per task


def test_deliver_requires_the_claimer():
    led, tb = setup()
    tb.claim("a", "t0001")
    with pytest.raises(BoardError):
        tb.deliver("b", "t0001", FULL_T1)
    with pytest.raises(BoardError):
        tb.deliver("a", "t0004", {"q0003": "4", "q0004": "6"})   # never claimed


def test_repeat_pay_per_task_leaf_pair():
    """q0003 sits under BOTH posted tasks: answering it pays in each."""
    led, tb = setup()
    tb.claim("a", "t0001")
    tb.deliver("a", "t0001", FULL_T1)
    tb.claim("a", "t0004")
    per_leaf, total = tb.deliver("a", "t0004", {"q0003": "4", "q0004": "6"})
    assert dict((q, p) for q, _, p in per_leaf) == {"q0003": 300, "q0004": 400}
    assert total == 700
    assert led.balance("a") == 10 + 600 + 700
    assert led.conservation_ok()


def test_expire_claims_reopens_stale_undelivered_tasks():
    led, tb = setup()
    tb.claim("a", "t0001", round_no=1)
    assert tb.expire_claims(5, ttl=8) == []
    assert tb.expire_claims(10, ttl=8) == ["t0001"]
    t = tb.tasks["t0001"]
    assert t.status == "open" and t.claimed_by is None and t.claimed_round == 0
    tb.claim("b", "t0001", round_no=11)


def test_expire_claims_never_reopens_a_closed_task():
    led, tb = setup()
    tb.claim("a", "t0001", round_no=1)
    tb.deliver("a", "t0001", FULL_T1)
    assert tb.expire_claims(100, ttl=1) == []
    assert tb.tasks["t0001"].status == "closed"


def test_two_strike_claim_limit_is_per_agent_and_per_task():
    led, tb = setup()
    tb.claim("a", "t0001", round_no=1)
    tb.expire_claims(20, ttl=8)
    tb.claim("a", "t0001", round_no=21)
    tb.expire_claims(40, ttl=8)
    with pytest.raises(BoardError):
        tb.claim("a", "t0001", round_no=41)
    tb.claim("b", "t0001", round_no=41)            # other agents unaffected
    tb.claim("a", "t0004", round_no=41)            # other tasks unaffected


def test_all_done_when_every_posted_task_is_closed():
    led, tb = setup()
    tb.claim("a", "t0001")
    tb.deliver("a", "t0001", FULL_T1)
    assert not tb.all_done()
    tb.claim("a", "t0004")
    tb.deliver("a", "t0004", {"q0003": "4", "q0004": "6"})
    assert tb.all_done()


def test_results_carry_task_and_per_leaf_records():
    led, tb = setup()
    tb.claim("a", "t0001", round_no=2)
    tb.deliver("a", "t0001", {"q0001": "Paris", "q0002": "Seine", "q0003": "four"})
    rec = next(r for r in tb.results() if r["nid"] == "t0001")
    assert rec["status"] == "closed" and rec["claimed_by"] == "a"
    assert rec["price"] == 600 and rec["payout"] == 400
    assert rec["sentence"] == "answer the french geography questions"
    leaves = {l["qid"]: l for l in rec["leaves"]}
    assert leaves["q0002"]["score"] == 0.0 and leaves["q0002"]["payout"] == 0
    assert leaves["q0003"]["em"] == 1.0 and leaves["q0003"]["submitted"] == "four"
    # the untouched task is reported too, as open
    assert next(r for r in tb.results() if r["nid"] == "t0004")["status"] == "open"


def test_leaf_results_flatten_every_task_leaf_pair():
    led, tb = setup()
    tb.claim("a", "t0001")
    tb.deliver("a", "t0001", FULL_T1)
    rows = tb.leaf_results()
    assert len(rows) == 5                                   # 3 under t0001 + 2 under t0004
    done = [r for r in rows if r["status"] == "closed"]
    assert len(done) == 3 and sum(r["score"] for r in done) == 3.0
    shared = [r for r in rows if r["qid"] == "q0003"]
    assert {r["nid"] for r in shared} == {"t0001", "t0004"}  # counted once per task
