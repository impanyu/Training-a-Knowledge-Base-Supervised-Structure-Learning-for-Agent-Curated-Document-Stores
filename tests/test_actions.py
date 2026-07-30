import pytest
from ca.actions import classify, dispatch, permission_error, visible_tools, ACTION_SPECS
from ca.config import LEVELS, ExperimentConfig
from ca.infra import Infra
from ca.retrieval import KeywordBackend
from ca.taskboard import Question

DOCS = [{"title": "Paris", "text": "Paris is the capital of France."}]


def make(level="L0", capital=1000):
    cfg = ExperimentConfig(level=LEVELS[level], seed=0, seed_capital_total=capital)
    qs = [Question("q0001", "capital of France?", ["Paris"], "easy", 100)]
    return Infra(cfg, qs, retriever=KeywordBackend(DOCS))


def test_classify():
    assert classify("retrieve", {"query": "x"}) == "solving"
    assert classify("work_on", {"task_id": "q0001", "thought": "t"}) == "solving"
    # decompose doesn't exist yet (T20) but must classify as solving, future-proof
    assert classify("decompose", {}) == "solving"
    assert classify("deliver_work", {"target_id": "q0001", "content": "Paris"}) == "solving"
    assert classify("deliver_work", {"target_id": "t0001", "content": "Paris"}) == "solving"
    assert classify("deliver_work", {"target_id": "c0001", "content": "x"}) == "admin"
    assert classify("send_message", {"to": "a", "text": "x"}) == "admin"
    assert classify("check_balance", {}) == "admin"


def test_world_gating_by_level():
    i0 = make("L0")
    assert permission_error(i0, "agent_1", "claim_question", {"qid": "q0001"}) is None
    i1 = make("L1")
    assert permission_error(i1, "agent_1", "claim_question", {"qid": "q0001"}) is not None
    assert permission_error(i1, "interface", "claim_question", {"qid": "q0001"}) is None


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
    # a non-interface agent may only borrow FROM the interface
    err = permission_error(i4, "agent_1", "propose_loan", {"to": "agent_2", "amount": 10})
    assert err is not None and "interface agent" in err
    assert permission_error(i4, "agent_1", "propose_loan", {"to": "interface", "amount": 10}) is None
    # the interface is the sole lender: it may not itself borrow
    err_i = permission_error(i4, "interface", "propose_loan", {"to": "agent_1", "amount": 10})
    assert err_i is not None and "sole lender" in err_i


def test_central_credit_not_gated_below_L4():
    i3 = make("L3")
    assert permission_error(i3, "agent_1", "propose_loan", {"to": "agent_2", "amount": 10}) is None
    assert permission_error(i3, "interface", "propose_loan", {"to": "agent_1", "amount": 10}) is None


def test_star_comms_extends_to_loans_at_L5():
    i5 = make("L5")
    err = permission_error(i5, "agent_1", "propose_loan", {"to": "agent_2", "amount": 10})
    assert err is not None
    assert permission_error(i5, "agent_1", "propose_loan", {"to": "interface", "amount": 10}) is None


def test_central_pricing_at_L3():
    i3 = make("L3")
    # counter_offer blocked for EVERYONE (including on agent-agent contracts)
    c = i3.contracts.propose("interface", "agent_1", "solve q0001", 50)
    assert permission_error(i3, "agent_1", "counter_offer",
                            {"contract_id": c.cid, "price": 80}) is not None
    # non-interface proposals enter unpriced state; interface prices them
    out = dispatch(i3, "agent_1", "propose_contract",
                   {"to": "agent_2", "task": "subtask"})
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
    assert len(iface) == 1                       # pricing + offer notice, not two copies
    assert "c0001" in iface[0].text
    # a proposal to a peer still notifies BOTH the interface and that peer
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
    # propose without price is an error outside central pricing
    out = dispatch(i0, "agent_1", "propose_contract", {"to": "agent_3", "task": "t"})
    assert out.startswith("ERROR")


def test_bankrupt_blocks_solving_only():
    i0 = make("L0", capital=8)  # 1 token each
    i0.ledger.burn("agent_1", 5)
    err = permission_error(i0, "agent_1", "retrieve", {"query": "x"})
    assert err is not None
    assert "coordinate or borrow" in err
    assert permission_error(i0, "agent_1", "send_message", {"to": "agent_2", "text": "s"}) is None


def test_dispatch_full_answer_flow():
    i0 = make("L0")
    out = dispatch(i0, "agent_1", "list_questions", {})
    assert "q0001" in out
    dispatch(i0, "agent_1", "claim_question", {"qid": "q0001"})
    out = dispatch(i0, "agent_1", "retrieve", {"query": "capital of France"})
    assert "Paris" in out
    dispatch(i0, "agent_1", "work_on", {"task_id": "q0001", "thought": "answer is Paris"})
    out = dispatch(i0, "agent_1", "deliver_work", {"target_id": "q0001", "content": "Paris"})
    assert "100" in out  # payout mentioned
    assert i0.ledger.balance("agent_1") > 125  # 125 seed + 100 payout
    assert i0.ledger.conservation_ok()


def test_dispatch_contract_flow_delivers_to_chat():
    i0 = make("L0")
    dispatch(i0, "agent_1", "propose_contract", {"to": "agent_2", "task": "find capital", "price": 30})
    dispatch(i0, "agent_2", "accept_contract", {"contract_id": "c0001"})
    dispatch(i0, "agent_2", "deliver_work", {"target_id": "c0001", "content": "it is Paris"})
    unread = i0.chat.unread("agent_1")
    assert any("Paris" in m.text for m in unread)
    assert i0.ledger.conservation_ok()


def test_list_questions_shows_top_n_by_price_with_overflow_note():
    cfg = ExperimentConfig(level=LEVELS["L0"], seed=0, seed_capital_total=1000)
    qs = [Question(f"q{i:04d}", f"question {i}", ["x"], "easy", i) for i in range(1, 26)]
    infra = Infra(cfg, qs, retriever=None)
    out = dispatch(infra, "agent_1", "list_questions", {})
    lines = out.splitlines()
    assert len(lines) == 21                          # 20 questions + overflow note
    assert lines[0].startswith("q0025")              # most valuable first
    assert lines[-1] == "... and 5 more (call list_questions with offset=20 to see them)"
    assert "q0001 " not in out                       # the 5 cheapest are dropped


def test_dispatch_error_string_not_exception():
    i0 = make("L0")
    out = dispatch(i0, "agent_1", "claim_question", {"qid": "q9999"})
    assert out.startswith("ERROR")


def test_visible_tools_filtered():
    names_l0 = {t["name"] for t in visible_tools(LEVELS["L0"], "agent_1")}
    assert "claim_question" in names_l0 and "retrieve" in names_l0
    assert "counter_offer" in names_l0 and "set_price" not in names_l0
    names_l2 = {t["name"] for t in visible_tools(LEVELS["L2"], "agent_1")}
    assert "claim_question" not in names_l2 and "retrieve" not in names_l2
    names_l2i = {t["name"] for t in visible_tools(LEVELS["L2"], "interface")}
    assert "claim_question" in names_l2i and "retrieve" in names_l2i
    names_l3 = {t["name"] for t in visible_tools(LEVELS["L3"], "agent_1")}
    assert "counter_offer" not in names_l3 and "set_price" not in names_l3
    names_l3i = {t["name"] for t in visible_tools(LEVELS["L3"], "interface")}
    assert "set_price" in names_l3i and "counter_offer" not in names_l3i
    assert set(ACTION_SPECS) >= names_l0


def test_L6_hides_multi_agent_tool_schemas():
    # a solo agent has nobody to message, contract or pay: billing it for those
    # schemas on every turn is pure waste
    names = {t["name"] for t in visible_tools(LEVELS["L6"], "agent_1")}
    assert names == {"retrieve", "work_on", "deliver_work", "list_questions",
                     "claim_question", "push_goal", "pop_goal",
                     "memory_write", "memory_search", "check_balance"}


def test_pay_insufficient_returns_error_string():
    i0 = make("L0")
    out = dispatch(i0, "agent_1", "pay", {"to": "agent_2", "amount": 10_000_000})
    assert out.startswith("ERROR")
    assert i0.ledger.conservation_ok()


def test_pay_unknown_recipient_destroys_nothing():
    i0 = make("L0")
    before = i0.ledger.balance("agent_1")
    out = dispatch(i0, "agent_1", "pay", {"to": "agent_99", "amount": 10})
    assert out.startswith("ERROR") and "agent_99" in out
    assert i0.ledger.balance("agent_1") == before   # tokens not destroyed
    assert i0.ledger.conservation_ok()


def test_unknown_agent_error_lists_roster():
    i0 = make("L0")
    out = dispatch(i0, "agent_1", "pay", {"to": "Interface", "amount": 5})
    assert out.startswith("ERROR") and "valid agents" in out and "agent_2" in out
    out2 = dispatch(i0, "agent_1", "send_message", {"to": "agent_99", "text": "x"})
    assert "valid agents" in out2


def test_list_questions_pagination_offset():
    cfg = ExperimentConfig(level=LEVELS["L0"], seed=0, seed_capital_total=1000)
    qs = [Question(f"q{i:04d}", f"t{i}", ["x"], "2hop", 100 + i) for i in range(1, 26)]
    infra = Infra(cfg, qs, retriever=KeywordBackend(DOCS))
    first = dispatch(infra, "agent_1", "list_questions", {})
    assert "offset=20" in first          # 25 open, page size 20 -> pointer to next page
    second = dispatch(infra, "agent_1", "list_questions", {"offset": 20})
    assert second.count("q0") == 5 and "more" not in second
    empty = dispatch(infra, "agent_1", "list_questions", {"offset": 99})
    assert "25 open in total" in empty
