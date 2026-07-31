import json

import pytest
from fixtures import demo_infra, demo_library, demo_posted

from ca.actions import ACTION_SPECS, classify, dispatch, permission_error, visible_tools
from ca.config import CONFIGS, ExperimentConfig
from ca.infra import Infra
from ca.retrieval import KeywordBackend
from ca.taskboard import Question
from ca.tasktree import TaskLibrary, TaskNode

DOCS = [{"title": "Paris", "text": "Paris is the capital of France."}]
FULL_T1 = json.dumps({"q0001": "Paris", "q0002": "Loire", "q0003": "4"})


def make(level="C0", capital=1000, **kw):
    return demo_infra(level, capital, retriever=KeywordBackend(DOCS), **kw)


def wide(n=25):
    """A flat library of n single-leaf tasks, for pagination checks."""
    nodes = [TaskNode(f"t{i:04d}", f"handle question number {i}", [f"q{i:04d}"])
             for i in range(1, n + 1)]
    qs = [Question(f"q{i:04d}", f"question {i}", ["x"], "2hop", 100 + i)
          for i in range(1, n + 1)]
    lib = TaskLibrary(nodes, qs)
    cfg = ExperimentConfig(level=CONFIGS["C0"], seed=0, seed_capital_total=1000)
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
    i0 = make("C0")
    assert permission_error(i0, "agent_1", "claim_task", {"task": "t0001"}) is None
    assert permission_error(i0, "agent_1", "list_tasks", {}) is None
    i1 = make("C1")
    assert permission_error(i1, "agent_1", "claim_task", {"task": "t0001"}) is not None
    assert permission_error(i1, "agent_1", "list_tasks", {}) is not None
    assert permission_error(i1, "hub", "claim_task", {"task": "t0001"}) is None


def test_contract_routing_uses_id_shape_not_leading_letter():
    """A task sentence that happens to start with 'c' must NOT be misrouted to
    the contract branch: classify must call it "solving", and delivering it
    to the WORLD by sentence must work exactly like any other task."""
    nodes = [TaskNode("t0001", "compare the premiere years of two operas", ["q0001"])]
    qs = [Question("q0001", "which opera premiered first, Salome or Elektra?",
                   ["Salome"], "easy", 100)]
    lib = TaskLibrary(nodes, qs)
    cfg0 = ExperimentConfig(level=CONFIGS["C0"], seed=0, seed_capital_total=1000)
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

    # at C1 a non-hub agent addressing the same sentence hits the world
    # gate, not the (wrong) contract branch that would report "unknown contract"
    cfg1 = ExperimentConfig(level=CONFIGS["C1"], seed=0, seed_capital_total=1000)
    infra1 = Infra(cfg1, lib, ["t0001"], retriever=None)
    err = permission_error(infra1, "agent_1", "deliver_work",
                           {"target_id": "compare the premiere years of two operas",
                            "content": "{}"})
    assert err is not None and "hub" in err


def test_world_delivery_gating_covers_sentence_targets():
    i1 = make("C1")
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
    for lvl in ("C0", "C1", "C3", "C5"):
        infra = make(lvl)
        assert permission_error(infra, "agent_1", "decompose", {"node": "t0001"}) is None
        assert "decompose" in {t["name"] for t in visible_tools(CONFIGS[lvl], "agent_1")}


def test_retrieve_is_open_to_everyone_at_every_config():
    """v3 deleted info centralization: retrieval is shared infrastructure."""
    for name in CONFIGS:
        infra = make(name)
        for who in ("agent_1", "hub"):
            if who == "hub" and not CONFIGS[name].has_hub:
                continue
            assert permission_error(infra, who, "retrieve", {"query": "x"}) is None
            assert "retrieve" in {t["name"] for t in visible_tools(CONFIGS[name], who)}


def test_star_comms_gating():
    i5 = make("C5")
    assert permission_error(i5, "agent_1", "send_message", {"to": "agent_2", "text": "hi"}) is not None
    assert permission_error(i5, "agent_1", "send_message", {"to": "hub", "text": "hi"}) is None
    assert permission_error(i5, "hub", "send_message", {"to": "agent_2", "text": "hi"}) is None


def test_central_credit_gating_at_C4():
    i4 = make("C4")
    err = permission_error(i4, "agent_1", "propose_loan", {"to": "agent_2", "amount": 10})
    assert err is not None and "hub agent" in err
    assert permission_error(i4, "agent_1", "propose_loan", {"to": "hub", "amount": 10}) is None
    err_i = permission_error(i4, "hub", "propose_loan", {"to": "agent_1", "amount": 10})
    assert err_i is not None and "sole lender" in err_i


def test_credit_stays_free_where_it_is_not_the_flipped_mechanism():
    for name in ("C0", "C1", "C2", "C3", "C6"):
        infra = make(name)
        assert permission_error(infra, "agent_1", "propose_loan",
                                {"to": "agent_2", "amount": 10}) is None, name


def test_star_comms_extends_to_loans_at_C5():
    i5 = make("C5")
    assert permission_error(i5, "agent_1", "propose_loan", {"to": "agent_2", "amount": 10}) is not None
    assert permission_error(i5, "agent_1", "propose_loan", {"to": "hub", "amount": 10}) is None


def test_central_pricing_at_C3():
    i3 = make("C3")
    c = i3.contracts.propose("hub", "agent_1", "solve t0001", 50)
    assert permission_error(i3, "agent_1", "counter_offer",
                            {"contract_id": c.cid, "price": 80}) is not None
    out = dispatch(i3, "agent_1", "propose_contract", {"to": "agent_2", "task": "subtask"})
    assert "pricing" in out
    c2 = i3.contracts.get("c0002")
    assert c2.status == "unpriced"
    assert permission_error(i3, "agent_1", "set_price",
                            {"contract_id": "c0002", "price": 30}) is not None
    assert permission_error(i3, "hub", "set_price",
                            {"contract_id": "c0002", "price": 30}) is None
    dispatch(i3, "hub", "set_price", {"contract_id": "c0002", "price": 30})
    dispatch(i3, "agent_2", "accept_contract", {"contract_id": "c0002"})
    assert i3.ledger.escrow["c0002"] == 30
    assert i3.ledger.conservation_ok()


def test_propose_to_hub_under_central_pricing_is_not_double_notified():
    i3 = make("C3")
    dispatch(i3, "agent_1", "propose_contract", {"to": "hub", "task": "look up X"})
    iface = i3.chat.unread("hub")
    assert len(iface) == 1
    assert "c0001" in iface[0].text
    dispatch(i3, "agent_1", "propose_contract", {"to": "agent_2", "task": "sub"})
    assert len(i3.chat.unread("hub")) == 2
    assert len(i3.chat.unread("agent_2")) == 1


def test_c3_has_no_demand_monopoly_so_workers_still_claim_and_deliver():
    """C3 flips pricing ONLY: the hub exists but everybody keeps equal
    access to the task board, unlike the cumulative v2 levels."""
    i3 = make("C3")
    assert permission_error(i3, "agent_1", "list_tasks", {}) is None
    assert permission_error(i3, "agent_1", "claim_task", {"task": "t0001"}) is None
    out = dispatch(i3, "agent_1", "claim_task", {"task": "t0001"})
    assert not out.startswith("ERROR") and i3.board.tasks["t0001"].claimed_by == "agent_1"
    assert permission_error(i3, "agent_1", "deliver_work",
                            {"target_id": "t0001", "content": FULL_T1}) is None
    paid = dispatch(i3, "agent_1", "deliver_work",
                    {"target_id": "t0001", "content": FULL_T1})
    assert "600" in paid and i3.board.tasks["t0001"].status == "closed"
    assert i3.ledger.conservation_ok()


def test_world_access_is_open_wherever_it_is_not_the_flipped_mechanism():
    for name in ("C0", "C2", "C3", "C4", "C5", "C6"):
        infra = make(name)
        assert permission_error(infra, "agent_1", "claim_task", {"task": "t0001"}) is None, name
        assert permission_error(infra, "agent_1", "list_tasks", {}) is None, name


def test_free_bargaining_without_central_pricing():
    i0 = make("C0")
    c = i0.contracts.propose("agent_1", "agent_2", "sub", 10)
    assert permission_error(i0, "agent_2", "counter_offer",
                            {"contract_id": c.cid, "price": 20}) is None
    assert permission_error(i0, "agent_1", "set_price",
                            {"contract_id": c.cid, "price": 5}) is not None
    out = dispatch(i0, "agent_1", "propose_contract", {"to": "agent_3", "task": "t"})
    assert out.startswith("ERROR")


def test_bankrupt_blocks_solving_only():
    i0 = make("C0", capital=8)
    i0.ledger.burn("agent_1", 5)
    err = permission_error(i0, "agent_1", "retrieve", {"query": "x"})
    assert err is not None and "coordinate or borrow" in err
    assert permission_error(i0, "agent_1", "decompose", {"node": "t0001"}) is not None
    assert permission_error(i0, "agent_1", "send_message", {"to": "agent_2", "text": "s"}) is None


# ---------------- board actions ----------------

def test_list_tasks_shows_sentence_leafcount_and_reward():
    i0 = make("C0")
    out = dispatch(i0, "agent_1", "list_tasks", {})
    lines = out.splitlines()
    # per-viewer stable shuffle: both tasks present with full detail, any order
    assert sorted(lines) == sorted([
        "[t0004] «resolve the two arithmetic warmup questions» (2 questions, reward 700)",
        "[t0001] «answer the french geography questions» (3 questions, reward 600)"])
    assert dispatch(i0, "agent_1", "list_tasks", {}) == out    # stable per viewer
    assert "q0001" not in out          # leaves stay hidden until decompose


def test_list_tasks_pagination_offset():
    infra = wide(25)
    first = dispatch(infra, "agent_1", "list_tasks", {})
    lines = first.splitlines()
    assert len(lines) == 21                                   # 20 tasks + overflow note
    assert lines[-1] == "... and 5 more (call list_tasks with offset=20 to see them)"
    second = dispatch(infra, "agent_1", "list_tasks", {"offset": 20})
    assert second.count("[t0") == 5 and "more" not in second
    # stable per-viewer order => the two pages partition all 25 exactly once
    import re
    seen = re.findall(r"\[t\d{4}\]", first) + re.findall(r"\[t\d{4}\]", second)
    assert len(seen) == 25 and len(set(seen)) == 25
    empty = dispatch(infra, "agent_1", "list_tasks", {"offset": 99})
    assert "25 open in total" in empty


def test_claim_task_accepts_id_or_sentence_and_hides_the_task():
    i0 = make("C0")
    out = dispatch(i0, "agent_1", "claim_task", {"task": "Answer the French geography questions"})
    assert "t0001" in out and "600" in out
    assert "t0001" not in dispatch(i0, "agent_2", "list_tasks", {})
    assert dispatch(i0, "agent_2", "claim_task", {"task": "t0001"}).startswith("ERROR")


def test_claim_unknown_task_lists_candidates():
    i0 = make("C0")
    out = dispatch(i0, "agent_1", "claim_task", {"task": "who painted the sistine chapel ceiling"})
    assert out.startswith("ERROR") and "did you mean" in out


def test_decompose_reveals_children_only_one_level_down():
    i0 = make("C0")
    out = dispatch(i0, "agent_1", "decompose", {"node": "t0001"})
    assert "[t0002] «name the capital and the river» (2 questions, reward 300)" in out
    assert "[q0003] 2+2?" in out
    assert "q0001" not in out                  # hidden one level deeper
    deeper = dispatch(i0, "agent_1", "decompose", {"node": "t0002"})
    assert "[q0001] capital of France?" in deeper and "[q0002]" in deeper


def test_decompose_of_a_leaf_returns_the_question_text():
    i0 = make("C0")
    out = dispatch(i0, "agent_1", "decompose", {"node": "q0003"})
    assert out == "[q0003] 2+2?"


def test_decompose_ambiguous_reference_errors_with_candidates():
    i0 = make("C0")
    out = dispatch(i0, "agent_1", "decompose",
                   {"node": "answer the french geograpology questions"})
    assert out.startswith("ERROR") and "t0001" in out and "t0005" in out


def test_decompose_works_on_unposted_library_nodes():
    """A subcontractor may be hired for a subtree that is not itself posted."""
    i0 = make("C0")
    assert "[q0005]" in dispatch(i0, "agent_1", "decompose", {"node": "t0005"})


# ---------------- packaged delivery to WORLD ----------------

def test_packaged_delivery_grades_pays_and_closes():
    i0 = make("C0")
    dispatch(i0, "agent_1", "claim_task", {"task": "t0001"})
    out = dispatch(i0, "agent_1", "deliver_work", {"target_id": "t0001", "content": FULL_T1})
    assert "600" in out and "q0002" in out
    assert i0.ledger.balance("agent_1") == 125 + 600
    assert i0.ledger.conservation_ok()
    assert i0.board.tasks["t0001"].status == "closed"


def test_packaged_delivery_accepts_the_sentence_as_target():
    i0 = make("C0")
    dispatch(i0, "agent_1", "claim_task", {"task": "t0001"})
    out = dispatch(i0, "agent_1", "deliver_work",
                   {"target_id": "answer the french geography questions", "content": FULL_T1})
    assert not out.startswith("ERROR") and i0.board.tasks["t0001"].status == "closed"


def test_bad_json_delivery_is_rejected_without_consuming_the_attempt():
    i0 = make("C0")
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
    i0 = make("C0")
    dispatch(i0, "agent_1", "claim_task", {"task": "t0001"})
    out = dispatch(i0, "agent_1", "deliver_work",
                   {"target_id": "t0001", "content": json.dumps({"q0001": "Paris"})})
    assert out.startswith("ERROR") and "q0003" in out
    assert i0.board.tasks["t0001"].status == "claimed"
    assert i0.ledger.balance("agent_1") == 125
    assert not dispatch(i0, "agent_1", "deliver_work",
                        {"target_id": "t0001", "content": FULL_T1}).startswith("ERROR")


def test_second_packaged_delivery_is_refused():
    i0 = make("C0")
    dispatch(i0, "agent_1", "claim_task", {"task": "t0001"})
    dispatch(i0, "agent_1", "deliver_work", {"target_id": "t0001", "content": FULL_T1})
    assert dispatch(i0, "agent_1", "deliver_work",
                    {"target_id": "t0001", "content": FULL_T1}).startswith("ERROR")


def test_repeat_pay_across_two_tasks_sharing_a_leaf():
    i0 = make("C0")
    dispatch(i0, "agent_1", "claim_task", {"task": "t0001"})
    dispatch(i0, "agent_1", "deliver_work", {"target_id": "t0001", "content": FULL_T1})
    dispatch(i0, "agent_1", "claim_task", {"task": "t0004"})
    out = dispatch(i0, "agent_1", "deliver_work",
                   {"target_id": "t0004", "content": json.dumps({"q0003": "4", "q0004": "6"})})
    assert "700" in out
    assert i0.ledger.balance("agent_1") == 125 + 600 + 700


def test_full_solo_answer_flow():
    i0 = make("C0")
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
    i0 = make("C0")
    dispatch(i0, "agent_1", "propose_contract",
             {"to": "agent_2", "task": "find the capital", "price": 30})
    assert i0.contracts.get("c0001").node_id is None       # free-text contract
    dispatch(i0, "agent_2", "accept_contract", {"contract_id": "c0001"})
    dispatch(i0, "agent_2", "deliver_work", {"target_id": "c0001", "content": "it is Paris"})
    assert any("Paris" in m.text for m in i0.chat.unread("agent_1"))
    assert i0.ledger.conservation_ok()


def test_propose_contract_binds_a_recognised_subtask_node():
    i0 = make("C0")
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
    i0 = make("C0")
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
    i0 = make("C0")
    out = dispatch(i0, "agent_1", "propose_contract",
                   {"to": "agent_2", "task": "name the capital and the river", "price": 30})
    assert "t0002" in out
    assert i0.contracts.get("c0001").node_id == "t0002"


def test_node_bound_contract_requires_full_leaf_coverage():
    i0 = make("C0")
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
    i0 = make("C0")
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
    i0 = make("C0")
    assert dispatch(i0, "agent_1", "claim_task", {"task": "t9999"}).startswith("ERROR")
    assert dispatch(i0, "agent_1", "decompose", {"node": "t9999"}).startswith("ERROR")


def test_claiming_an_unposted_subtask_is_refused():
    i0 = make("C0")
    out = dispatch(i0, "agent_1", "claim_task", {"task": "t0002"})
    assert out.startswith("ERROR") and "not a task posted" in out


def test_visible_tools_filtered():
    names_c0 = {t["name"] for t in visible_tools(CONFIGS["C0"], "agent_1")}
    assert {"claim_task", "list_tasks", "decompose", "retrieve"} <= names_c0
    assert "claim_question" not in names_c0 and "list_questions" not in names_c0
    assert "counter_offer" in names_c0 and "set_price" not in names_c0
    names_c1 = {t["name"] for t in visible_tools(CONFIGS["C1"], "agent_1")}
    assert "claim_task" not in names_c1 and "list_tasks" not in names_c1
    assert {"decompose", "retrieve"} <= names_c1        # v3: retrieval is never gated
    names_c1i = {t["name"] for t in visible_tools(CONFIGS["C1"], "hub")}
    assert "claim_task" in names_c1i and "retrieve" in names_c1i
    names_c3 = {t["name"] for t in visible_tools(CONFIGS["C3"], "agent_1")}
    assert "counter_offer" not in names_c3 and "set_price" not in names_c3
    assert "claim_task" in names_c3                     # C3 flips pricing only
    names_c3i = {t["name"] for t in visible_tools(CONFIGS["C3"], "hub")}
    assert "set_price" in names_c3i and "counter_offer" not in names_c3i
    assert set(ACTION_SPECS) >= names_c0


def test_collective_goal_changes_no_permissions():
    """C6 centralizes the objective function, not any right: every action must
    behave exactly as it does at C0, for workers and would-be hubs alike."""
    assert (visible_tools(CONFIGS["C6"], "agent_1")
            == visible_tools(CONFIGS["C0"], "agent_1"))
    i6, i0 = make("C6"), make("C0")
    probes = [("list_tasks", {}), ("claim_task", {"task": "t0001"}),
              ("retrieve", {"query": "x"}), ("decompose", {"node": "t0001"}),
              ("deliver_work", {"target_id": "t0001", "content": FULL_T1}),
              ("send_message", {"to": "agent_2", "text": "hi"}),
              ("propose_contract", {"to": "agent_2", "task": "sub", "price": 5}),
              ("counter_offer", {"contract_id": "c0001", "price": 9}),
              ("set_price", {"contract_id": "c0001", "price": 9}),
              ("propose_loan", {"to": "agent_2", "amount": 10}),
              ("pay", {"to": "agent_2", "amount": 1}), ("check_balance", {})]
    for name, inp in probes:
        assert (permission_error(i6, "agent_1", name, inp)
                == permission_error(i0, "agent_1", name, inp)), name


def test_shared_solution_memory_changes_reach_not_permissions():
    """T26 landed the store, so C2 is no longer byte-identical to C0 -- but the
    difference is exactly one of REACH (whose solutions you can read; see
    test_solutions.test_c2_shares_the_store_across_agents_and_c0_does_not).
    Every agent still sees the same tool schemas and the same gating: nothing
    is permitted or forbidden at C2 that is not permitted or forbidden at C0,
    including the memory-aware decompose itself."""
    assert CONFIGS["C2"].shared_solution_memory is True
    assert (visible_tools(CONFIGS["C2"], "agent_1")
            == visible_tools(CONFIGS["C0"], "agent_1"))
    i2, i0 = make("C2"), make("C0")
    for name, inp in (("retrieve", {"query": "x"}), ("claim_task", {"task": "t0001"}),
                      ("decompose", {"node": "t0001"}),
                      ("propose_contract", {"to": "agent_2", "task": "s", "price": 5}),
                      ("propose_loan", {"to": "agent_2", "amount": 10})):
        assert (permission_error(i2, "agent_1", name, inp)
                == permission_error(i0, "agent_1", name, inp)), name


def test_C7_hides_multi_agent_tool_schemas():
    names = {t["name"] for t in visible_tools(CONFIGS["C7"], "agent_1")}
    assert names == {"retrieve", "work_on", "deliver_work", "list_tasks",
                     "claim_task", "decompose",
                     "push_goal", "pop_goal",
                     "memory_write", "memory_search", "check_balance"}


def test_pay_insufficient_returns_error_string():
    i0 = make("C0")
    assert dispatch(i0, "agent_1", "pay", {"to": "agent_2", "amount": 10_000_000}).startswith("ERROR")
    assert i0.ledger.conservation_ok()


def test_pay_unknown_recipient_destroys_nothing():
    i0 = make("C0")
    before = i0.ledger.balance("agent_1")
    out = dispatch(i0, "agent_1", "pay", {"to": "agent_99", "amount": 10})
    assert out.startswith("ERROR") and "agent_99" in out
    assert i0.ledger.balance("agent_1") == before
    assert i0.ledger.conservation_ok()


def test_unknown_agent_error_lists_roster():
    i0 = make("C0")
    out = dispatch(i0, "agent_1", "pay", {"to": "Hub", "amount": 5})
    assert out.startswith("ERROR") and "valid agents" in out and "agent_2" in out
    assert "valid agents" in dispatch(i0, "agent_1", "send_message",
                                      {"to": "agent_99", "text": "x"})
