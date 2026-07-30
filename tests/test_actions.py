import json

import pytest
from fixtures import demo_infra, demo_library, demo_posted

from ca.actions import ACTION_SPECS, classify, dispatch, permission_error, visible_tools
from ca.config import LEVELS, ExperimentConfig
from ca.infra import Infra
from ca.retrieval import KeywordBackend
from ca.taskboard import Question
from ca.tasktree import TaskLibrary, TaskNode

DOCS = [{"title": "Paris", "text": "Paris is the capital of France."}]
FULL_T1 = json.dumps({"q0001": "Paris", "q0002": "Loire", "q0003": "4"})


def make(level="L0", capital=1000, **kw):
    return demo_infra(level, capital, retriever=KeywordBackend(DOCS), **kw)


def wide(n=25):
    """A flat library of n single-leaf tasks, for pagination checks."""
    nodes = [TaskNode(f"t{i:04d}", f"handle question number {i}", [f"q{i:04d}"])
             for i in range(1, n + 1)]
    qs = [Question(f"q{i:04d}", f"question {i}", ["x"], "2hop", 100 + i)
          for i in range(1, n + 1)]
    lib = TaskLibrary(nodes, qs)
    cfg = ExperimentConfig(level=LEVELS["L0"], seed=0, seed_capital_total=1000)
    return Infra(cfg, lib, [n.nid for n in nodes], retriever=None)


# ---------------- classification & gating ----------------

def test_classify():
    assert classify("retrieve", {"query": "x"}) == "solving"
    assert classify("work_on", {"task_id": "q0001", "thought": "t"}) == "solving"
    assert classify("decompose", {"node": "t0001"}) == "solving"
    assert classify("deliver_work", {"target_id": "q0001", "content": "Paris"}) == "solving"
    assert classify("deliver_work", {"target_id": "t0001", "content": "{}"}) == "solving"
    # sentence-addressed WORLD delivery is solving too
    assert classify("deliver_work", {"target_id": "answer the french geography questions",
                                     "content": "{}"}) == "solving"
    assert classify("deliver_work", {"target_id": "c0001", "content": "x"}) == "admin"
    assert classify("send_message", {"to": "a", "text": "x"}) == "admin"
    assert classify("check_balance", {}) == "admin"


def test_world_gating_by_level():
    i0 = make("L0")
    assert permission_error(i0, "agent_1", "claim_task", {"task": "t0001"}) is None
    assert permission_error(i0, "agent_1", "list_tasks", {}) is None
    i1 = make("L1")
    assert permission_error(i1, "agent_1", "claim_task", {"task": "t0001"}) is not None
    assert permission_error(i1, "agent_1", "list_tasks", {}) is not None
    assert permission_error(i1, "interface", "claim_task", {"task": "t0001"}) is None


def test_contract_routing_uses_id_shape_not_leading_letter():
    """A task sentence that happens to start with 'c' must NOT be misrouted to
    the contract branch: classify must call it "solving", and delivering it
    to the WORLD by sentence must work exactly like any other task."""
    nodes = [TaskNode("t0001", "compare the premiere years of two operas", ["q0001"])]
    qs = [Question("q0001", "which opera premiered first, Salome or Elektra?",
                   ["Salome"], "easy", 100)]
    lib = TaskLibrary(nodes, qs)
    cfg0 = ExperimentConfig(level=LEVELS["L0"], seed=0, seed_capital_total=1000)
    infra0 = Infra(cfg0, lib, ["t0001"], retriever=None)

    assert classify("deliver_work",
                    {"target_id": "compare the premiere years of two operas",
                     "content": "{}"}) == "solving"

    dispatch(infra0, "agent_1", "claim_task", {"task": "t0001"})
    out = dispatch(infra0, "agent_1", "deliver_work",
                   {"target_id": "compare the premiere years of two operas",
                    "content": json.dumps({"q0001": "Salome"})})
    assert not out.startswith("ERROR")
    assert infra0.board.tasks["t0001"].status == "closed"

    # at L1 a non-interface agent addressing the same sentence hits the world
    # gate, not the (wrong) contract branch that would report "unknown contract"
    cfg1 = ExperimentConfig(level=LEVELS["L1"], seed=0, seed_capital_total=1000)
    infra1 = Infra(cfg1, lib, ["t0001"], retriever=None)
    err = permission_error(infra1, "agent_1", "deliver_work",
                           {"target_id": "compare the premiere years of two operas",
                            "content": "{}"})
    assert err is not None and "interface" in err


def test_world_delivery_gating_covers_sentence_targets():
    i1 = make("L1")
    assert permission_error(i1, "agent_1", "deliver_work",
                            {"target_id": "t0001", "content": "{}"}) is not None
    assert permission_error(i1, "agent_1", "deliver_work",
                            {"target_id": "answer the french geography questions",
                             "content": "{}"}) is not None
    # contract deliveries stay open to everyone at every level
    assert permission_error(i1, "agent_1", "deliver_work",
                            {"target_id": "c0001", "content": "x"}) is None


def test_decompose_is_visible_and_permitted_to_everyone():
    """Subcontractors must be able to inspect the structure they were hired for."""
    for lvl in ("L0", "L1", "L2", "L5"):
        infra = make(lvl)
        assert permission_error(infra, "agent_1", "decompose", {"node": "t0001"}) is None
        assert "decompose" in {t["name"] for t in visible_tools(LEVELS[lvl], "agent_1")}


def test_retrieve_gating_and_star_comms():
    i2 = make("L2")
    assert permission_error(i2, "agent_1", "retrieve", {"query": "x"}) is not None
    assert permission_error(i2, "interface", "retrieve", {"query": "x"}) is None
    i5 = make("L5")
    assert permission_error(i5, "agent_1", "send_message", {"to": "agent_2", "text": "hi"}) is not None
    assert permission_error(i5, "agent_1", "send_message", {"to": "interface", "text": "hi"}) is None
    assert permission_error(i5, "interface", "send_message", {"to": "agent_2", "text": "hi"}) is None


def test_central_credit_gating_at_L4():
    i4 = make("L4")
    err = permission_error(i4, "agent_1", "propose_loan", {"to": "agent_2", "amount": 10})
    assert err is not None and "interface agent" in err
    assert permission_error(i4, "agent_1", "propose_loan", {"to": "interface", "amount": 10}) is None
    err_i = permission_error(i4, "interface", "propose_loan", {"to": "agent_1", "amount": 10})
    assert err_i is not None and "sole lender" in err_i


def test_central_credit_not_gated_below_L4():
    i3 = make("L3")
    assert permission_error(i3, "agent_1", "propose_loan", {"to": "agent_2", "amount": 10}) is None
    assert permission_error(i3, "interface", "propose_loan", {"to": "agent_1", "amount": 10}) is None


def test_star_comms_extends_to_loans_at_L5():
    i5 = make("L5")
    assert permission_error(i5, "agent_1", "propose_loan", {"to": "agent_2", "amount": 10}) is not None
    assert permission_error(i5, "agent_1", "propose_loan", {"to": "interface", "amount": 10}) is None


def test_central_pricing_at_L3():
    i3 = make("L3")
    c = i3.contracts.propose("interface", "agent_1", "solve t0001", 50)
    assert permission_error(i3, "agent_1", "counter_offer",
                            {"contract_id": c.cid, "price": 80}) is not None
    out = dispatch(i3, "agent_1", "propose_contract", {"to": "agent_2", "task": "subtask"})
    assert "pricing" in out
    c2 = i3.contracts.get("c0002")
    assert c2.status == "unpriced"
    assert permission_error(i3, "agent_1", "set_price",
                            {"contract_id": "c0002", "price": 30}) is not None
    assert permission_error(i3, "interface", "set_price",
                            {"contract_id": "c0002", "price": 30}) is None
    dispatch(i3, "interface", "set_price", {"contract_id": "c0002", "price": 30})
    dispatch(i3, "agent_2", "accept_contract", {"contract_id": "c0002"})
    assert i3.ledger.escrow["c0002"] == 30
    assert i3.ledger.conservation_ok()


def test_propose_to_interface_under_central_pricing_is_not_double_notified():
    i3 = make("L3")
    dispatch(i3, "agent_1", "propose_contract", {"to": "interface", "task": "look up X"})
    iface = i3.chat.unread("interface")
    assert len(iface) == 1
    assert "c0001" in iface[0].text
    dispatch(i3, "agent_1", "propose_contract", {"to": "agent_2", "task": "sub"})
    assert len(i3.chat.unread("interface")) == 2
    assert len(i3.chat.unread("agent_2")) == 1


def test_free_bargaining_below_L3():
    i0 = make("L0")
    c = i0.contracts.propose("agent_1", "agent_2", "sub", 10)
    assert permission_error(i0, "agent_2", "counter_offer",
                            {"contract_id": c.cid, "price": 20}) is None
    assert permission_error(i0, "agent_1", "set_price",
                            {"contract_id": c.cid, "price": 5}) is not None
    out = dispatch(i0, "agent_1", "propose_contract", {"to": "agent_3", "task": "t"})
    assert out.startswith("ERROR")


def test_bankrupt_blocks_solving_only():
    i0 = make("L0", capital=8)
    i0.ledger.burn("agent_1", 5)
    err = permission_error(i0, "agent_1", "retrieve", {"query": "x"})
    assert err is not None and "coordinate or borrow" in err
    assert permission_error(i0, "agent_1", "decompose", {"node": "t0001"}) is not None
    assert permission_error(i0, "agent_1", "send_message", {"to": "agent_2", "text": "s"}) is None


# ---------------- board actions ----------------

def test_list_tasks_shows_sentence_leafcount_and_reward():
    i0 = make("L0")
    out = dispatch(i0, "agent_1", "list_tasks", {})
    lines = out.splitlines()
    assert lines[0] == "[t0004] «resolve the two arithmetic warmup questions» (2 questions, reward 700)"
    assert lines[1] == "[t0001] «answer the french geography questions» (3 questions, reward 600)"
    assert "q0001" not in out          # leaves stay hidden until decompose


def test_list_tasks_pagination_offset():
    infra = wide(25)
    first = dispatch(infra, "agent_1", "list_tasks", {})
    lines = first.splitlines()
    assert len(lines) == 21                                   # 20 tasks + overflow note
    assert lines[0].startswith("[t0025]")                     # most valuable first
    assert lines[-1] == "... and 5 more (call list_tasks with offset=20 to see them)"
    second = dispatch(infra, "agent_1", "list_tasks", {"offset": 20})
    assert second.count("[t0") == 5 and "more" not in second
    empty = dispatch(infra, "agent_1", "list_tasks", {"offset": 99})
    assert "25 open in total" in empty


def test_claim_task_accepts_id_or_sentence_and_hides_the_task():
    i0 = make("L0")
    out = dispatch(i0, "agent_1", "claim_task", {"task": "Answer the French geography questions"})
    assert "t0001" in out and "600" in out
    assert "t0001" not in dispatch(i0, "agent_2", "list_tasks", {})
    assert dispatch(i0, "agent_2", "claim_task", {"task": "t0001"}).startswith("ERROR")


def test_claim_unknown_task_lists_candidates():
    i0 = make("L0")
    out = dispatch(i0, "agent_1", "claim_task", {"task": "who painted the sistine chapel ceiling"})
    assert out.startswith("ERROR") and "did you mean" in out


def test_decompose_reveals_children_only_one_level_down():
    i0 = make("L0")
    out = dispatch(i0, "agent_1", "decompose", {"node": "t0001"})
    assert "[t0002] «name the capital and the river» (2 questions, reward 300)" in out
    assert "[q0003] 2+2?" in out
    assert "q0001" not in out                  # hidden one level deeper
    deeper = dispatch(i0, "agent_1", "decompose", {"node": "t0002"})
    assert "[q0001] capital of France?" in deeper and "[q0002]" in deeper


def test_decompose_of_a_leaf_returns_the_question_text():
    i0 = make("L0")
    out = dispatch(i0, "agent_1", "decompose", {"node": "q0003"})
    assert out == "[q0003] 2+2?"


def test_decompose_ambiguous_reference_errors_with_candidates():
    i0 = make("L0")
    out = dispatch(i0, "agent_1", "decompose",
                   {"node": "answer the french geograpology questions"})
    assert out.startswith("ERROR") and "t0001" in out and "t0005" in out


def test_decompose_works_on_unposted_library_nodes():
    """A subcontractor may be hired for a subtree that is not itself posted."""
    i0 = make("L0")
    assert "[q0005]" in dispatch(i0, "agent_1", "decompose", {"node": "t0005"})


# ---------------- packaged delivery to WORLD ----------------

def test_packaged_delivery_grades_pays_and_closes():
    i0 = make("L0")
    dispatch(i0, "agent_1", "claim_task", {"task": "t0001"})
    out = dispatch(i0, "agent_1", "deliver_work", {"target_id": "t0001", "content": FULL_T1})
    assert "600" in out and "q0002" in out
    assert i0.ledger.balance("agent_1") == 125 + 600
    assert i0.ledger.conservation_ok()
    assert i0.board.tasks["t0001"].status == "closed"


def test_packaged_delivery_accepts_the_sentence_as_target():
    i0 = make("L0")
    dispatch(i0, "agent_1", "claim_task", {"task": "t0001"})
    out = dispatch(i0, "agent_1", "deliver_work",
                   {"target_id": "answer the french geography questions", "content": FULL_T1})
    assert not out.startswith("ERROR") and i0.board.tasks["t0001"].status == "closed"


def test_bad_json_delivery_is_rejected_without_consuming_the_attempt():
    i0 = make("L0")
    dispatch(i0, "agent_1", "claim_task", {"task": "t0001"})
    out = dispatch(i0, "agent_1", "deliver_work", {"target_id": "t0001", "content": "Paris"})
    assert out.startswith("ERROR") and "JSON" in out
    assert i0.board.tasks["t0001"].status == "claimed"
    # a JSON array is well-formed JSON but not a {qid: answer} map
    out2 = dispatch(i0, "agent_1", "deliver_work",
                    {"target_id": "t0001", "content": '["Paris"]'})
    assert out2.startswith("ERROR") and i0.board.tasks["t0001"].status == "claimed"
    ok = dispatch(i0, "agent_1", "deliver_work", {"target_id": "t0001", "content": FULL_T1})
    assert not ok.startswith("ERROR")


def test_incomplete_json_delivery_is_rejected_without_consuming_the_attempt():
    i0 = make("L0")
    dispatch(i0, "agent_1", "claim_task", {"task": "t0001"})
    out = dispatch(i0, "agent_1", "deliver_work",
                   {"target_id": "t0001", "content": json.dumps({"q0001": "Paris"})})
    assert out.startswith("ERROR") and "q0003" in out
    assert i0.board.tasks["t0001"].status == "claimed"
    assert i0.ledger.balance("agent_1") == 125
    assert not dispatch(i0, "agent_1", "deliver_work",
                        {"target_id": "t0001", "content": FULL_T1}).startswith("ERROR")


def test_second_packaged_delivery_is_refused():
    i0 = make("L0")
    dispatch(i0, "agent_1", "claim_task", {"task": "t0001"})
    dispatch(i0, "agent_1", "deliver_work", {"target_id": "t0001", "content": FULL_T1})
    assert dispatch(i0, "agent_1", "deliver_work",
                    {"target_id": "t0001", "content": FULL_T1}).startswith("ERROR")


def test_repeat_pay_across_two_tasks_sharing_a_leaf():
    i0 = make("L0")
    dispatch(i0, "agent_1", "claim_task", {"task": "t0001"})
    dispatch(i0, "agent_1", "deliver_work", {"target_id": "t0001", "content": FULL_T1})
    dispatch(i0, "agent_1", "claim_task", {"task": "t0004"})
    out = dispatch(i0, "agent_1", "deliver_work",
                   {"target_id": "t0004", "content": json.dumps({"q0003": "4", "q0004": "6"})})
    assert "700" in out
    assert i0.ledger.balance("agent_1") == 125 + 600 + 700


def test_full_solo_answer_flow():
    i0 = make("L0")
    assert "t0001" in dispatch(i0, "agent_1", "list_tasks", {})
    dispatch(i0, "agent_1", "claim_task", {"task": "t0001"})
    assert "[q0003]" in dispatch(i0, "agent_1", "decompose", {"node": "t0001"})
    assert "Paris" in dispatch(i0, "agent_1", "retrieve", {"query": "capital of France"})
    dispatch(i0, "agent_1", "work_on", {"task_id": "q0001", "thought": "answer is Paris"})
    out = dispatch(i0, "agent_1", "deliver_work", {"target_id": "t0001", "content": FULL_T1})
    assert "600" in out
    assert i0.ledger.conservation_ok()


# ---------------- contracts ----------------

def test_dispatch_contract_flow_delivers_to_chat():
    i0 = make("L0")
    dispatch(i0, "agent_1", "propose_contract",
             {"to": "agent_2", "task": "find the capital", "price": 30})
    assert i0.contracts.get("c0001").node_id is None       # free-text contract
    dispatch(i0, "agent_2", "accept_contract", {"contract_id": "c0001"})
    dispatch(i0, "agent_2", "deliver_work", {"target_id": "c0001", "content": "it is Paris"})
    assert any("Paris" in m.text for m in i0.chat.unread("agent_1"))
    assert i0.ledger.conservation_ok()


def test_propose_contract_binds_a_recognised_subtask_node():
    i0 = make("L0")
    out = dispatch(i0, "agent_1", "propose_contract",
                   {"to": "agent_2", "task": "Name the capital and the river", "price": 30})
    assert "t0002" in out
    c = i0.contracts.get("c0001")
    assert c.node_id == "t0002"
    assert any("t0002" in m.text and "coverage" in m.text
               for m in i0.chat.unread("agent_2"))


def test_propose_contract_near_miss_sentence_does_not_bind():
    """A free-text task that merely RESEMBLES a node sentence (fuzzy match)
    must not opportunistically bind the contract to that node -- only an
    exact id or exact normalized-sentence match may bind."""
    i0 = make("L0")
    # this typo'd sentence is close enough to t0001 to fuzzy-resolve via
    # TaskLibrary.resolve (see test_resolve_fuzzy_best_match_above_threshold)
    out = dispatch(i0, "agent_1", "propose_contract",
                   {"to": "agent_2", "task": "answer the french geograhy questions",
                    "price": 30})
    c = i0.contracts.get("c0001")
    assert c.node_id is None
    assert "bound to" not in out
    dispatch(i0, "agent_2", "accept_contract", {"contract_id": "c0001"})
    good = dispatch(i0, "agent_2", "deliver_work",
                    {"target_id": "c0001", "content": "free-text answer, no JSON needed"})
    assert not good.startswith("ERROR")


def test_propose_contract_exact_sentence_binds():
    i0 = make("L0")
    out = dispatch(i0, "agent_1", "propose_contract",
                   {"to": "agent_2", "task": "name the capital and the river", "price": 30})
    assert "t0002" in out
    assert i0.contracts.get("c0001").node_id == "t0002"


def test_node_bound_contract_requires_full_leaf_coverage():
    i0 = make("L0")
    dispatch(i0, "agent_1", "propose_contract",
             {"to": "agent_2", "task": "name the capital and the river", "price": 30})
    dispatch(i0, "agent_2", "accept_contract", {"contract_id": "c0001"})
    bad = dispatch(i0, "agent_2", "deliver_work",
                   {"target_id": "c0001", "content": json.dumps({"q0001": "Paris"})})
    assert bad.startswith("ERROR") and "q0002" in bad
    assert i0.contracts.get("c0001").status == "accepted"
    assert i0.ledger.escrow["c0001"] == 30                 # not settled
    not_json = dispatch(i0, "agent_2", "deliver_work",
                        {"target_id": "c0001", "content": "Paris and the Loire"})
    assert not_json.startswith("ERROR")
    assert i0.contracts.get("c0001").status == "accepted"
    good = dispatch(i0, "agent_2", "deliver_work",
                    {"target_id": "c0001",
                     "content": json.dumps({"q0001": "Paris", "q0002": "Loire"})})
    assert not good.startswith("ERROR")
    assert i0.contracts.get("c0001").status == "delivered"
    assert i0.ledger.escrow == {} and i0.ledger.conservation_ok()


def test_node_bound_contract_does_not_grade_quality():
    i0 = make("L0")
    dispatch(i0, "agent_1", "propose_contract",
             {"to": "agent_2", "task": "name the capital and the river", "price": 30})
    dispatch(i0, "agent_2", "accept_contract", {"contract_id": "c0001"})
    out = dispatch(i0, "agent_2", "deliver_work",
                   {"target_id": "c0001",
                    "content": json.dumps({"q0001": "wrong", "q0002": "wrong"})})
    assert not out.startswith("ERROR")                     # coverage only, no grading
    assert i0.ledger.balance("agent_2") == 125 + 30


# ---------------- misc invariants ----------------

def test_dispatch_error_string_not_exception():
    i0 = make("L0")
    assert dispatch(i0, "agent_1", "claim_task", {"task": "t9999"}).startswith("ERROR")
    assert dispatch(i0, "agent_1", "decompose", {"node": "t9999"}).startswith("ERROR")


def test_claiming_an_unposted_subtask_is_refused():
    i0 = make("L0")
    out = dispatch(i0, "agent_1", "claim_task", {"task": "t0002"})
    assert out.startswith("ERROR") and "not a task posted" in out


def test_visible_tools_filtered():
    names_l0 = {t["name"] for t in visible_tools(LEVELS["L0"], "agent_1")}
    assert {"claim_task", "list_tasks", "decompose", "retrieve"} <= names_l0
    assert "claim_question" not in names_l0 and "list_questions" not in names_l0
    assert "counter_offer" in names_l0 and "set_price" not in names_l0
    names_l2 = {t["name"] for t in visible_tools(LEVELS["L2"], "agent_1")}
    assert "claim_task" not in names_l2 and "retrieve" not in names_l2
    assert "decompose" in names_l2
    names_l2i = {t["name"] for t in visible_tools(LEVELS["L2"], "interface")}
    assert "claim_task" in names_l2i and "retrieve" in names_l2i
    names_l3 = {t["name"] for t in visible_tools(LEVELS["L3"], "agent_1")}
    assert "counter_offer" not in names_l3 and "set_price" not in names_l3
    names_l3i = {t["name"] for t in visible_tools(LEVELS["L3"], "interface")}
    assert "set_price" in names_l3i and "counter_offer" not in names_l3i
    assert set(ACTION_SPECS) >= names_l0


def test_L6_hides_multi_agent_tool_schemas():
    names = {t["name"] for t in visible_tools(LEVELS["L6"], "agent_1")}
    assert names == {"retrieve", "work_on", "deliver_work", "list_tasks",
                     "claim_task", "decompose", "push_goal", "pop_goal",
                     "memory_write", "memory_search", "check_balance"}


def test_pay_insufficient_returns_error_string():
    i0 = make("L0")
    assert dispatch(i0, "agent_1", "pay", {"to": "agent_2", "amount": 10_000_000}).startswith("ERROR")
    assert i0.ledger.conservation_ok()


def test_pay_unknown_recipient_destroys_nothing():
    i0 = make("L0")
    before = i0.ledger.balance("agent_1")
    out = dispatch(i0, "agent_1", "pay", {"to": "agent_99", "amount": 10})
    assert out.startswith("ERROR") and "agent_99" in out
    assert i0.ledger.balance("agent_1") == before
    assert i0.ledger.conservation_ok()


def test_unknown_agent_error_lists_roster():
    i0 = make("L0")
    out = dispatch(i0, "agent_1", "pay", {"to": "Interface", "amount": 5})
    assert out.startswith("ERROR") and "valid agents" in out and "agent_2" in out
    assert "valid agents" in dispatch(i0, "agent_1", "send_message",
                                      {"to": "agent_99", "text": "x"})
