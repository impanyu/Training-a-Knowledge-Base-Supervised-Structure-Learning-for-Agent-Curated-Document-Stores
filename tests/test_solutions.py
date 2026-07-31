"""T26: per-agent solution KV memory (auto-written) + recall_solutions."""
import json

import pytest
from fixtures import demo_infra

from ca.actions import ACTION_SPECS, classify, dispatch, visible_tools
from ca.config import CONFIGS
from ca.context import render_turn
from ca.memory import FifoMemory, GoalStack
from ca.skills import role_skill
from ca.solutions import SolutionMemory

FULL_T1 = json.dumps({"q0001": "Paris", "q0002": "Loire", "q0003": "4"})
T2_MAP = json.dumps({"q0001": "Paris", "q0002": "Loire"})


def make(level="C0", capital=1000):
    return demo_infra(level, capital)


# ---------------- SolutionMemory unit ----------------

def test_records_and_reads_back_both_mapping_kinds():
    sm = SolutionMemory()
    sm.record_decomposition("a", "t0001", ["t0002", "q0003"])
    sm.record_answer("a", "q0003", "4", f1=1.0)
    assert sm.mapping("a", "t0001") == ["t0002", "q0003"]
    assert sm.answer("a", "q0003") == {"answer": "4", "f1": 1.0}
    assert sm.mapping("a", "t0002") is None
    assert sm.answer("a", "q0001") is None


def test_store_is_private_per_agent_by_default():
    sm = SolutionMemory()
    sm.record_answer("a", "q0003", "4")
    assert sm.answer("b", "q0003") is None
    assert sm.recall("b", "q0003")["known"] == {}


def test_answer_without_grade_has_no_f1_tag():
    sm = SolutionMemory()
    sm.record_answer("a", "q0003", "4")
    assert sm.answer("a", "q0003") == {"answer": "4"}


def test_a_graded_answer_supersedes_an_ungraded_one():
    """Contract deliveries are ungraded; a later WORLD grade is strictly more
    informative, so it must not be lost -- nor overwritten by a re-delivery."""
    sm = SolutionMemory()
    sm.record_answer("a", "q0003", "four")
    sm.record_answer("a", "q0003", "4", f1=1.0)
    assert sm.answer("a", "q0003") == {"answer": "4", "f1": 1.0}
    sm.record_answer("a", "q0003", "IV")          # ungraded, later
    assert sm.answer("a", "q0003") == {"answer": "4", "f1": 1.0}


def test_recall_expands_the_stored_tree_recursively():
    sm = SolutionMemory()
    sm.record_decomposition("a", "t0001", ["t0002", "q0003"])
    sm.record_decomposition("a", "t0002", ["q0001", "q0002"])
    for qid, ans in (("q0001", "Paris"), ("q0002", "Loire"), ("q0003", "4")):
        sm.record_answer("a", qid, ans, f1=1.0)
    r = sm.recall("a", "t0001")
    assert set(r["known"]) == {"q0001", "q0002", "q0003"}
    assert r["known"]["q0001"]["answer"] == "Paris"
    assert r["missing"] == [] and r["unexpanded"] == []


def test_recall_reports_missing_leaves_and_unexpanded_branches():
    sm = SolutionMemory()
    sm.record_decomposition("a", "t0001", ["t0002", "q0003"])
    sm.record_answer("a", "q0003", "4", f1=0.5)
    r = sm.recall("a", "t0001")
    assert list(r["known"]) == ["q0003"]
    assert r["missing"] == []          # q0001/q0002 are not known to exist yet
    assert r["unexpanded"] == ["t0002"]
    # once t0002 is decomposed its leaves become *known to be missing*
    sm.record_decomposition("a", "t0002", ["q0001", "q0002"])
    r = sm.recall("a", "t0001")
    assert r["missing"] == ["q0001", "q0002"] and r["unexpanded"] == []


def test_recall_on_a_bare_question():
    sm = SolutionMemory()
    sm.record_answer("a", "q0003", "4")
    assert list(sm.recall("a", "q0003")["known"]) == ["q0003"]
    assert sm.recall("a", "q0009") == {"known": {}, "missing": ["q0009"],
                                       "unexpanded": []}


def test_recall_on_a_node_never_decomposed():
    sm = SolutionMemory()
    assert sm.recall("a", "t0001") == {"known": {}, "missing": [],
                                       "unexpanded": ["t0001"]}


def test_recall_is_cycle_safe():
    sm = SolutionMemory()
    sm.record_decomposition("a", "t0001", ["t0002"])
    sm.record_decomposition("a", "t0002", ["t0001", "q0001"])
    sm.record_answer("a", "q0001", "Paris")
    r = sm.recall("a", "t0001")          # must terminate
    assert list(r["known"]) == ["q0001"]


def test_recall_dedupes_a_shared_subtree():
    sm = SolutionMemory()
    sm.record_decomposition("a", "t0001", ["t0002", "t0003"])
    sm.record_decomposition("a", "t0002", ["q0001"])
    sm.record_decomposition("a", "t0003", ["q0001", "q0002"])
    r = sm.recall("a", "t0001")
    assert r["missing"] == ["q0001", "q0002"]


def test_stats_counts_per_agent():
    sm = SolutionMemory()
    assert sm.stats("a") == {"answers": 0, "decompositions": 0}
    sm.record_decomposition("a", "t0001", ["q0001"])
    sm.record_answer("a", "q0001", "Paris")
    sm.record_answer("a", "q0002", "Loire")
    assert sm.stats("a") == {"answers": 2, "decompositions": 1}
    assert sm.stats("b") == {"answers": 0, "decompositions": 0}


def test_shared_store_pools_every_agent():
    sm = SolutionMemory(shared=True)
    sm.record_decomposition("a", "t0001", ["q0001"])
    sm.record_answer("a", "q0001", "Paris", f1=1.0)
    assert list(sm.recall("b", "t0001")["known"]) == ["q0001"]
    assert sm.stats("zzz") == {"answers": 1, "decompositions": 1}


# ---------------- auto-record triggers (through dispatch) ----------------

def test_decompose_records_the_mapping():
    i = make()
    dispatch(i, "agent_1", "decompose", {"node": "t0001"})
    assert i.solutions.mapping("agent_1", "t0001") == ["t0002", "q0003"]
    # by sentence too -- the handler resolves before storing, keys stay nids
    dispatch(i, "agent_1", "decompose", {"node": "name the capital and the river"})
    assert i.solutions.mapping("agent_1", "t0002") == ["q0001", "q0002"]
    assert i.solutions.mapping("agent_2", "t0001") is None


def test_echoing_a_question_records_nothing():
    i = make()
    dispatch(i, "agent_1", "decompose", {"node": "q0003"})
    assert i.solutions.stats("agent_1") == {"answers": 0, "decompositions": 0}


def test_world_delivery_records_graded_answers():
    i = make()
    dispatch(i, "agent_1", "claim_task", {"task": "t0001"})
    out = dispatch(i, "agent_1", "deliver_work", {"target_id": "t0001", "content": FULL_T1})
    assert not out.startswith("ERROR")
    assert i.solutions.answer("agent_1", "q0001") == {"answer": "Paris", "f1": 1.0}
    assert i.solutions.answer("agent_1", "q0003")["f1"] == 1.0
    assert i.solutions.stats("agent_1")["answers"] == 3


def test_rejected_world_delivery_records_nothing():
    i = make()
    dispatch(i, "agent_1", "claim_task", {"task": "t0001"})
    dispatch(i, "agent_1", "deliver_work",
             {"target_id": "t0001", "content": json.dumps({"q0001": "Paris"})})
    assert i.solutions.stats("agent_1")["answers"] == 0


def _bound_contract(i, payer="agent_1", worker="agent_2", price=5):
    dispatch(i, payer, "propose_contract",
             {"to": worker, "task": "name the capital and the river", "price": price})
    cid = list(i.contracts.contracts)[-1]
    dispatch(i, worker, "accept_contract", {"contract_id": cid})
    return cid


def test_bound_contract_delivery_records_for_contractor_and_payer():
    i = make()
    cid = _bound_contract(i)
    out = dispatch(i, "agent_2", "deliver_work", {"target_id": cid, "content": T2_MAP})
    assert not out.startswith("ERROR")
    # contractor: they produced the answers, ungraded (no F1 known)
    assert i.solutions.answer("agent_2", "q0001") == {"answer": "Paris"}
    assert i.solutions.answer("agent_2", "q0002") == {"answer": "Loire"}
    # payer: receiving the deliverable teaches them the same answers
    assert i.solutions.answer("agent_1", "q0001") == {"answer": "Paris"}
    assert i.solutions.answer("agent_1", "q0002") == {"answer": "Loire"}


def test_free_text_contract_delivery_records_nothing():
    i = make()
    dispatch(i, "agent_1", "propose_contract",
             {"to": "agent_2", "task": "help me out somehow", "price": 5})
    cid = list(i.contracts.contracts)[-1]
    dispatch(i, "agent_2", "accept_contract", {"contract_id": cid})
    dispatch(i, "agent_2", "deliver_work", {"target_id": cid, "content": "here you go"})
    assert i.solutions.stats("agent_2")["answers"] == 0
    assert i.solutions.stats("agent_1")["answers"] == 0


def test_incomplete_bound_delivery_records_nothing():
    i = make()
    cid = _bound_contract(i)
    out = dispatch(i, "agent_2", "deliver_work",
                   {"target_id": cid, "content": json.dumps({"q0001": "Paris"})})
    assert out.startswith("ERROR")
    assert i.solutions.stats("agent_2")["answers"] == 0
    assert i.solutions.stats("agent_1")["answers"] == 0


# ---------------- the recall_solutions action ----------------

def test_recall_solutions_is_classified_as_solving():
    assert classify("recall_solutions", {"name": "t0001"}) == "solving"


def test_recall_solutions_is_visible_everywhere_including_solo():
    for name in CONFIGS:
        for who in ("agent_1", "hub"):
            if who == "hub" and not CONFIGS[name].has_hub:
                continue
            names = {t["name"] for t in visible_tools(CONFIGS[name], who)}
            assert "recall_solutions" in names, (name, who)
    assert "recall_solutions" in ACTION_SPECS


def test_recall_solutions_reports_known_missing_and_unexpanded():
    i = make()
    dispatch(i, "agent_1", "decompose", {"node": "t0001"})
    dispatch(i, "agent_1", "decompose", {"node": "t0002"})
    i.solutions.record_answer("agent_1", "q0001", "Paris", f1=0.5)
    i.solutions.record_answer("agent_1", "q0003", "4")
    out = dispatch(i, "agent_1", "recall_solutions", {"name": "t0001"})
    assert "known 2/3" in out
    assert '"q0001": "Paris"' in out and "F1 0.50" in out
    assert '"q0003": "4"' in out
    assert "missing" in out and "q0002" in out


def test_recall_solutions_accepts_sentence_and_qid():
    i = make()
    i.solutions.record_decomposition("agent_1", "t0002", ["q0001", "q0002"])
    i.solutions.record_answer("agent_1", "q0001", "Paris")
    by_sentence = dispatch(i, "agent_1", "recall_solutions",
                           {"name": "name the capital and the river"})
    assert '"q0001": "Paris"' in by_sentence and "t0002" in by_sentence
    by_qid = dispatch(i, "agent_1", "recall_solutions", {"name": "q0001"})
    assert "known 1/1" in by_qid and '"q0001": "Paris"' in by_qid


def test_recall_solutions_when_nothing_is_stored():
    i = make()
    out = dispatch(i, "agent_1", "recall_solutions", {"name": "t0001"})
    assert out == "(no stored solutions under t0001)"


def test_recall_solutions_flags_unexpanded_branches():
    i = make()
    dispatch(i, "agent_1", "decompose", {"node": "t0001"})
    i.solutions.record_answer("agent_1", "q0003", "4", f1=1.0)
    out = dispatch(i, "agent_1", "recall_solutions", {"name": "t0001"})
    assert "known 1/1" in out and "t0002" in out


def test_recall_solutions_unknown_name_is_a_friendly_error():
    i = make()
    out = dispatch(i, "agent_1", "recall_solutions", {"name": "utter nonsense here"})
    assert out.startswith("ERROR")


def test_recall_solutions_does_not_leak_across_agents_at_c0():
    i = make()
    dispatch(i, "agent_1", "claim_task", {"task": "t0001"})
    dispatch(i, "agent_1", "deliver_work", {"target_id": "t0001", "content": FULL_T1})
    dispatch(i, "agent_1", "decompose", {"node": "t0002"})
    assert dispatch(i, "agent_2", "recall_solutions", {"name": "t0002"}) == \
        "(no stored solutions under t0002)"


# ---------------- C2 vs C0: the one mechanism C2 flips ----------------

def test_c2_shares_the_store_across_agents_and_c0_does_not():
    for level, shared in (("C2", True), ("C0", False)):
        i = make(level)
        dispatch(i, "agent_1", "decompose", {"node": "t0002"})
        i.solutions.record_answer("agent_1", "q0001", "Paris", f1=1.0)
        out = dispatch(i, "agent_2", "recall_solutions", {"name": "t0002"})
        assert ('"q0001": "Paris"' in out) is shared, level


def test_c2_agents_are_told_the_store_is_shared():
    shared_line = "The knowledge base is SHARED"
    assert shared_line in role_skill(CONFIGS["C2"], "agent_1")
    for name in CONFIGS:
        if name == "C2":
            continue
        for who in ("agent_1", "hub"):
            if who == "hub" and not CONFIGS[name].has_hub:
                continue
            assert shared_line not in role_skill(CONFIGS[name], who), (name, who)


# ---------------- skills & context surfaces ----------------

def test_every_role_at_every_config_gets_the_reuse_demo():
    for name in CONFIGS:
        for who in ("agent_1", "hub"):
            if who == "hub" and not CONFIGS[name].has_hub:
                continue
            s = role_skill(CONFIGS[name], who)
            assert "recall_solutions(" in s, (name, who)
            assert "Demo: reuse what you already solved" in s, (name, who)


def test_turn_view_advertises_the_store_only_when_it_is_non_empty():
    i = make()
    fifo, goals = FifoMemory(k=3), GoalStack("maximize token balance")
    assert "Solution memory" not in render_turn(i, "agent_1", fifo, goals)
    dispatch(i, "agent_1", "decompose", {"node": "t0002"})
    i.solutions.record_answer("agent_1", "q0001", "Paris", f1=1.0)
    view = render_turn(i, "agent_1", fifo, goals)
    assert ("Solution memory: 1 answers stored; decomposed: t0002 "
            "(recall_solutions to reuse)") in view
    assert "recall_solutions" in view


def test_shared_store_is_advertised_to_everyone_at_c2():
    i = make("C2")
    fifo, goals = FifoMemory(k=3), GoalStack("maximize token balance")
    i.solutions.record_answer("agent_1", "q0001", "Paris", f1=1.0)
    assert "Solution memory: 1 answers" in render_turn(i, "agent_2", fifo, goals)


def _memory_line(view: str) -> str:
    return next(l for l in view.splitlines() if l.startswith("Solution memory:"))


def test_turn_view_lists_decomposed_ids_in_insertion_order():
    i = make()
    fifo, goals = FifoMemory(k=3), GoalStack("maximize token balance")
    dispatch(i, "agent_1", "decompose", {"node": "t0001"})
    dispatch(i, "agent_1", "decompose", {"node": "t0002"})
    view = render_turn(i, "agent_1", fifo, goals)
    assert _memory_line(view) == ("Solution memory: 0 answers stored; "
                                  "decomposed: t0001, t0002 "
                                  "(recall_solutions to reuse)")


def test_turn_view_answers_only_omits_the_decomposed_list():
    i = make()
    fifo, goals = FifoMemory(k=3), GoalStack("maximize token balance")
    i.solutions.record_answer("agent_1", "q0001", "Paris", f1=1.0)
    view = render_turn(i, "agent_1", fifo, goals)
    assert _memory_line(view) == ("Solution memory: 1 answers stored "
                                  "(recall_solutions to reuse)")


def test_turn_view_truncates_the_decomposed_list_at_twelve():
    i = make()
    fifo, goals = FifoMemory(k=3), GoalStack("maximize token balance")
    for n in range(14):
        i.solutions.record_decomposition("agent_1", f"t{n:04d}", ["q0001"])
    line = _memory_line(render_turn(i, "agent_1", fifo, goals))
    shown = ", ".join(f"t{n:04d}" for n in range(12))
    assert f"decomposed: {shown} … +2 more (recall_solutions to reuse)" in line
    assert "t0012" not in line and "t0013" not in line


# ---------------- T30: repeat-decompose redirect ----------------

def test_has_decomposition_and_decomposed_ids():
    sm = SolutionMemory()
    assert not sm.has_decomposition("a", "t0001")
    assert sm.decomposed_ids("a") == []
    sm.record_decomposition("a", "t0001", ["q0001"])
    sm.record_decomposition("a", "t0002", ["q0002"])
    assert sm.has_decomposition("a", "t0001")
    assert not sm.has_decomposition("b", "t0001")
    assert sm.decomposed_ids("a") == ["t0001", "t0002"]
    assert sm.decomposed_ids("b") == []


def test_repeat_decompose_redirects_to_recall():
    i = make()
    assert "breaks down into" in dispatch(i, "agent_1", "decompose", {"node": "t0001"})
    again = dispatch(i, "agent_1", "decompose", {"node": "t0001"})
    assert again == ('(t0001 already decomposed: t0002, q0003 — '
                     'recall_solutions("t0001") for stored answers)')
    # by sentence too: the redirect keys on the resolved nid
    by_sentence = dispatch(i, "agent_1", "decompose",
                           {"node": "answer the french geography questions"})
    assert by_sentence == again
    # private store: another agent still gets the full first-time breakdown
    assert "breaks down into" in dispatch(i, "agent_2", "decompose", {"node": "t0001"})


def test_repeat_decompose_does_not_re_record(monkeypatch):
    i = make()
    dispatch(i, "agent_1", "decompose", {"node": "t0001"})
    calls = []
    monkeypatch.setattr(i.solutions, "record_decomposition",
                        lambda *a, **k: calls.append(a))
    dispatch(i, "agent_1", "decompose", {"node": "t0001"})
    assert calls == []
    assert i.solutions.mapping("agent_1", "t0001") == ["t0002", "q0003"]


def test_recall_after_decompose_shows_structure_not_empty_store():
    """The C7 ping-pong loop: decompose said "already decomposed, go recall",
    recall said "no stored solutions". With structure stored but no answers,
    recall must show the structure instead of contradicting the redirect."""
    i = make()
    dispatch(i, "agent_1", "decompose", {"node": "t0001"})
    out = dispatch(i, "agent_1", "recall_solutions", {"name": "t0001"})
    assert out.startswith("(no stored answers yet under t0001)")
    assert "not yet decomposed" in out or "unanswered" in out
    assert "no stored solutions" not in out


def test_leaf_echo_is_never_redirected():
    i = make()
    for _ in range(2):
        out = dispatch(i, "agent_1", "decompose", {"node": "q0003"})
        assert "[q0003]" in out and "already decomposed" not in out


def test_c2_shared_bucket_redirects_other_agents_too():
    i = make("C2")
    assert "breaks down into" in dispatch(i, "agent_1", "decompose", {"node": "t0001"})
    out = dispatch(i, "agent_2", "decompose", {"node": "t0001"})
    assert out == ('(t0001 already decomposed: t0002, q0003 — '
                   'recall_solutions("t0001") for stored answers)')


def test_reuse_demo_teaches_the_repeat_decompose_redirect():
    for name in CONFIGS:
        for who in ("agent_1", "hub"):
            if who == "hub" and not CONFIGS[name].has_hub:
                continue
            s = role_skill(CONFIGS[name], who)
            # the WRONG line must show the redirect exactly as _h_decompose
            # formats it -- a stale format would mis-teach the model
            assert ('(t0042 already decomposed: t0043, q0017 — '
                    'recall_solutions("t0042") for stored answers)') in s, (name, who)
            assert 'recall_solutions(name="t0042")' in s, (name, who)
            assert "Do NOT bounce" in s, (name, who)


def test_solution_store_is_separate_from_free_text_ltm():
    i = make()
    dispatch(i, "agent_1", "decompose", {"node": "t0002"})
    assert dispatch(i, "agent_1", "memory_search", {"query": "capital river q0001"}) \
        == "(no matching memories)"
    dispatch(i, "agent_1", "memory_write", {"content": "a free text note"})
    assert i.solutions.stats("agent_1") == {"answers": 0, "decompositions": 1}
