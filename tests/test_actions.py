import json

from fixtures import HashEmbedding, demo_infra

from ca.actions import ACTION_SPECS, classify, dispatch, permission_error, visible_tools
from ca.bank import Job, Question, QuestionBank
from ca.config import CONFIGS, ExperimentConfig
from ca.infra import Infra
from ca.retrieval import KeywordBackend

DOCS = [{"title": "Paris", "text": "Paris is the capital of France."}]

J1 = {"q0001": "Paris", "q0002": "Loire", "q0003": "4"}      # a perfect j0001 map


def make(level="C0", capital=1000, **kw):
    return demo_infra(level, capital, retriever=KeywordBackend(DOCS), **kw)


def dj(infra, agent, jid, answers):
    return dispatch(infra, agent, "deliver_work",
                    {"target_id": jid, "content": json.dumps(answers)})


def wide(n=25):
    """A bank of n one-question jobs, for pagination checks."""
    qs = [Question(f"q{i:04d}", f"question {i}", ["x"], "2hop", 100 + i, "k00")
          for i in range(1, n + 1)]
    jobs = [Job(f"j{i:04d}", [f"q{i:04d}"], 100 + i) for i in range(1, n + 1)]
    cfg = ExperimentConfig(level=CONFIGS["C0"], seed=0, seed_capital_total=1000)
    return Infra(cfg, QuestionBank(qs, jobs), retriever=None,
                 embedding_function=HashEmbedding())


# ---------------- classification & gating ----------------

def test_classify():
    assert classify("retrieve", {"query": "x"}) == "solving"
    assert classify("work_on", {"question_id": "q0001", "thought": "t"}) == "solving"
    assert classify("deliver_work", {"target_id": "j0001", "content": "{}"}) == "solving"
    assert classify("deliver_work", {"target_id": "c0001", "content": "x"}) == "admin"
    assert classify("claim_job", {"jid": "j0001"}) == "admin"
    assert classify("send_message", {"to": "a", "text": "x"}) == "admin"
    assert classify("check_balance", {}) == "admin"


def test_the_replaced_and_v3_tree_actions_are_gone():
    for dead in ("decompose", "recall_solutions", "list_tasks", "claim_task",
                 "list_questions", "claim_question"):
        assert dead not in ACTION_SPECS


def test_world_gating_by_level():
    i0 = make("C0")
    assert permission_error(i0, "agent_1", "claim_job", {"jid": "j0001"}) is None
    assert permission_error(i0, "agent_1", "list_jobs", {}) is None
    i1 = make("C1")
    assert permission_error(i1, "agent_1", "claim_job", {"jid": "j0001"}) is not None
    assert permission_error(i1, "agent_1", "list_jobs", {}) is not None
    assert permission_error(i1, "hub", "claim_job", {"jid": "j0001"}) is None


def test_world_delivery_gating_and_contract_deliveries_stay_open():
    i1 = make("C1")
    assert permission_error(i1, "agent_1", "deliver_work",
                            {"target_id": "j0001", "content": "{}"}) is not None
    # contract deliveries stay open to everyone at every level
    assert permission_error(i1, "agent_1", "deliver_work",
                            {"target_id": "c0001", "content": "x"}) is None


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
    c = i3.contracts.propose("hub", "agent_1", "solve q0001", 50)
    assert permission_error(i3, "agent_1", "counter_offer",
                            {"contract_id": c.cid, "price": 80}) is not None
    out = dispatch(i3, "agent_1", "propose_contract", {"to": "agent_2", "task": "some work"})
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
    access to the job board."""
    i3 = make("C3")
    assert permission_error(i3, "agent_1", "list_jobs", {}) is None
    out = dispatch(i3, "agent_1", "claim_job", {"jid": "j0001"})
    assert not out.startswith("ERROR")
    paid = dj(i3, "agent_1", "j0001", J1)
    assert "600 tokens" in paid and i3.ledger.conservation_ok()


def test_world_access_is_open_wherever_it_is_not_the_flipped_mechanism():
    for name in ("C0", "C2", "C3", "C4", "C5", "C6"):
        infra = make(name)
        assert permission_error(infra, "agent_1", "claim_job",
                                {"jid": "j0001"}) is None, name
        assert permission_error(infra, "agent_1", "list_jobs", {}) is None, name


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
    assert permission_error(i0, "agent_1", "deliver_work",
                            {"target_id": "j0001", "content": "{}"}) is not None
    assert permission_error(i0, "agent_1", "send_message", {"to": "agent_2", "text": "s"}) is None
    # claiming is coordination, not solving: a bankrupt agent may still claim
    assert permission_error(i0, "agent_1", "claim_job", {"jid": "j0001"}) is None


# ---------------- board actions ----------------

def test_list_jobs_shows_size_topic_and_reward():
    i0 = make("C0")
    out = dispatch(i0, "agent_1", "list_jobs", {})
    assert sorted(out.splitlines()) == sorted([
        "[j0001] 3 questions, topic k01, reward 600",
        "[j0002] 2 questions, topic k07, reward 700",
        "[j0003] 1 questions, topic k02, reward 50",
    ])
    assert dispatch(i0, "agent_1", "list_jobs", {}) == out     # stable per viewer


def test_list_jobs_order_differs_between_viewers_but_covers_the_same_set():
    i0 = wide(25)
    a = dispatch(i0, "agent_1", "list_jobs", {"offset": 0}).splitlines()[:20]
    b = dispatch(i0, "agent_2", "list_jobs", {"offset": 0}).splitlines()[:20]
    assert sorted(a) == sorted(b)
    assert a != b                       # de-herding: no shared "top" job


def test_list_jobs_drops_a_job_the_moment_someone_holds_it():
    i0 = make("C0")
    dispatch(i0, "agent_1", "claim_job", {"jid": "j0002"})
    assert "j0002" not in dispatch(i0, "agent_2", "list_jobs", {})
    assert "j0001" in dispatch(i0, "agent_2", "list_jobs", {})


def test_list_jobs_pagination_offset():
    infra = wide(25)
    first = dispatch(infra, "agent_1", "list_jobs", {})
    lines = first.splitlines()
    assert len(lines) == 21                                   # 20 jobs + overflow note
    assert lines[-1] == "... and 5 more (call list_jobs with offset=20 to see them)"
    second = dispatch(infra, "agent_1", "list_jobs", {"offset": 20})
    assert second.count("[j0") == 5 and "more" not in second
    import re
    seen = re.findall(r"\[j\d{4}\]", first) + re.findall(r"\[j\d{4}\]", second)
    assert len(seen) == 25 and len(set(seen)) == 25
    empty = dispatch(infra, "agent_1", "list_jobs", {"offset": 99})
    assert "25 open in total" in empty


def test_claim_job_reveals_every_member_question():
    i0 = make("C0")
    out = dispatch(i0, "agent_1", "claim_job", {"jid": "j0001"})
    lines = out.splitlines()
    assert lines[0] == "claimed [j0001] 3 questions, topic k01, reward 600"
    assert 'deliver ALL 3 answers in ONE map' in lines[1]
    assert lines[2:] == [
        "  — q0001 capital of France? (unanswered)",
        "  — q0002 longest river in France? (unanswered)",
        "  — q0003 2+2? (unanswered)",
    ]
    assert "memory:" not in out


def test_only_one_agent_may_hold_a_job():
    i0 = make("C0")
    assert not dispatch(i0, "agent_1", "claim_job", {"jid": "j0001"}).startswith("ERROR")
    out = dispatch(i0, "agent_2", "claim_job", {"jid": "j0001"})
    assert out.startswith("ERROR") and "another agent" in out


def test_claim_unknown_jid_lists_near_ids():
    i0 = make("C0")
    out = dispatch(i0, "agent_1", "claim_job", {"jid": "j9999"})
    assert out.startswith("ERROR") and "j0003" in out


def test_claim_rejects_contract_and_question_ids_with_a_targeted_hint():
    i0 = make("C0")
    out = dispatch(i0, "agent_1", "claim_job", {"jid": "c0002"})
    assert out.startswith("ERROR") and "contract id" in out and "accept_contract" in out
    out = dispatch(i0, "agent_1", "claim_job", {"jid": "q0002"})
    assert out.startswith("ERROR") and "list_jobs" in out


def test_two_strikes_close_a_job_to_one_agent_but_not_to_others():
    i0 = make("C0")
    dispatch(i0, "agent_1", "claim_job", {"jid": "j0001"})
    i0.board.expire_claims(100, ttl=i0.cfg.claim_ttl)
    dispatch(i0, "agent_1", "claim_job", {"jid": "j0001"})
    i0.board.expire_claims(200, ttl=i0.cfg.claim_ttl)
    third = dispatch(i0, "agent_1", "claim_job", {"jid": "j0001"})
    assert third.startswith("ERROR") and "twice" in third
    assert not dispatch(i0, "agent_2", "claim_job", {"jid": "j0001"}).startswith("ERROR")


# ---------------- delivery to WORLD ----------------

def test_delivery_grades_every_question_pays_once_and_closes_the_job():
    i0 = make("C0")
    dispatch(i0, "agent_1", "claim_job", {"jid": "j0001"})
    out = dj(i0, "agent_1", "j0001", J1)
    assert out.splitlines() == [
        "delivered j0001: 3 questions, total F1 3.00 -> 600 tokens",
        "  q0001 F1=1.00 -> 100",
        "  q0002 F1=1.00 -> 200",
        "  q0003 F1=1.00 -> 300",
    ]
    assert i0.ledger.balance("agent_1") == 125 + 600
    assert i0.ledger.conservation_ok()
    assert i0.board.closed == {"j0001"} and len(i0.board.results) == 3


def test_delivery_without_a_claim_is_refused():
    i0 = make("C0")
    out = dj(i0, "agent_1", "j0001", J1)
    assert out.startswith("ERROR") and "claim_job" in out
    assert i0.ledger.balance("agent_1") == 125


def test_second_delivery_on_one_claim_is_refused():
    i0 = make("C0")
    dispatch(i0, "agent_1", "claim_job", {"jid": "j0001"})
    dj(i0, "agent_1", "j0001", J1)
    assert dj(i0, "agent_1", "j0001", J1).startswith("ERROR")


def test_partial_f1_pays_partially():
    i0 = make("C0")
    dispatch(i0, "agent_1", "claim_job", {"jid": "j0001"})
    out = dj(i0, "agent_1", "j0001", dict(J1, q0002="Loire River"))
    assert "q0002 F1=0.67 -> 133" in out          # round(200 * 2/3)
    assert "-> 533 tokens" in out


# ---------------- all-or-nothing delivery ----------------

def test_an_incomplete_map_is_rejected_and_the_attempt_survives():
    i0 = make("C0")
    dispatch(i0, "agent_1", "claim_job", {"jid": "j0001"})
    out = dj(i0, "agent_1", "j0001", {"q0001": "Paris", "q0002": "Loire"})
    assert out.startswith("ERROR") and "q0003" in out and "STILL YOURS" in out
    assert i0.board.results == [] and i0.ledger.balance("agent_1") == 125
    assert dj(i0, "agent_1", "j0001", J1).startswith("delivered j0001")


def test_a_map_naming_a_question_outside_the_job_is_rejected_free():
    i0 = make("C0")
    dispatch(i0, "agent_1", "claim_job", {"jid": "j0001"})
    out = dj(i0, "agent_1", "j0001", dict(J1, q0005="sedimentary"))
    assert out.startswith("ERROR") and "q0005" in out
    assert dj(i0, "agent_1", "j0001", J1).startswith("delivered j0001")


def test_content_that_is_not_json_is_rejected_free_with_a_hint():
    i0 = make("C0")
    dispatch(i0, "agent_1", "claim_job", {"jid": "j0001"})
    out = dispatch(i0, "agent_1", "deliver_work",
                   {"target_id": "j0001", "content": "Paris, Loire, 4"})
    assert out.startswith("ERROR") and "JSON" in out and '"q0042"' in out
    assert dj(i0, "agent_1", "j0001", J1).startswith("delivered j0001")


def test_a_json_value_that_is_not_a_short_answer_is_rejected_free():
    i0 = make("C0")
    dispatch(i0, "agent_1", "claim_job", {"jid": "j0001"})
    nested = json.dumps({"q0001": {"answer": "Paris"}, "q0002": "Loire", "q0003": "4"})
    out = dispatch(i0, "agent_1", "deliver_work",
                   {"target_id": "j0001", "content": nested})
    assert out.startswith("ERROR") and "q0001" in out
    out = dispatch(i0, "agent_1", "deliver_work",
                   {"target_id": "j0001", "content": json.dumps(["Paris"])})
    assert out.startswith("ERROR") and "not an object" in out
    assert dj(i0, "agent_1", "j0001", J1).startswith("delivered j0001")


def test_delivering_a_question_id_points_at_the_job():
    i0 = make("C0")
    dispatch(i0, "agent_1", "claim_job", {"jid": "j0001"})
    out = dispatch(i0, "agent_1", "deliver_work",
                   {"target_id": "q0001", "content": "Paris"})
    assert out.startswith("ERROR") and "whole jobs" in out
    assert i0.board.results == []


# ---------------- automatic memory ----------------

def test_delivery_writes_every_graded_answer_into_memory():
    i0 = make("C0")
    dispatch(i0, "agent_1", "claim_job", {"jid": "j0001"})
    dj(i0, "agent_1", "j0001", J1)
    rec = i0.memory.answer("agent_1", "q0001")
    assert rec == {"text": '[q0001] capital of France? -> "Paris" (F1 1.00)',
                   "kind": "answer", "qid": "q0001", "f1": 1.0}
    assert i0.memory.n_answers("agent_1") == 3
    assert i0.memory.answer("agent_2", "q0001") is None      # private at C0


def test_claim_marks_the_members_already_in_memory():
    i0 = make("C0")
    dispatch(i0, "agent_1", "claim_job", {"jid": "j0002"})
    dj(i0, "agent_1", "j0002", {"q0003": "4", "q0004": "nonsense"})
    out = dispatch(i0, "agent_1", "claim_job", {"jid": "j0001"})
    lines = out.splitlines()
    assert lines[2] == ("memory: stored answer for 1 of these 3 questions "
                        "(marked ✓ below) - that part is already paid for")
    assert lines[3].startswith("  — q0001")
    assert lines[5].startswith("  ✓ q0003 2+2?")
    assert '"4" (F1 1.00)' in lines[5] and "GOOD (F1 1.00" in lines[5]


def test_claim_flags_a_low_quality_stored_answer_as_worth_re_solving():
    i0 = make("C0")
    dispatch(i0, "agent_1", "claim_job", {"jid": "j0002"})
    dj(i0, "agent_1", "j0002", {"q0003": "seven", "q0004": "6"})
    out = dispatch(i0, "agent_1", "claim_job", {"jid": "j0001"})
    assert "LOW QUALITY (F1 0.00 of 1.00)" in out and "re-solve" in out


def test_a_repeat_delivery_reports_whether_it_beat_the_stored_f1():
    i0 = make("C0")
    dispatch(i0, "agent_1", "claim_job", {"jid": "j0002"})
    dj(i0, "agent_1", "j0002", {"q0003": "five", "q0004": "six"})
    dispatch(i0, "agent_1", "claim_job", {"jid": "j0001"})
    out = dj(i0, "agent_1", "j0001", J1)
    assert "IMPROVED on your stored F1 0.00" in out
    assert out.count("stored F1") == 1        # only q0003 was a repeat

    i1 = make("C0")
    dispatch(i1, "agent_1", "claim_job", {"jid": "j0002"})
    dj(i1, "agent_1", "j0002", {"q0003": "4", "q0004": "6"})
    dispatch(i1, "agent_1", "claim_job", {"jid": "j0001"})
    out = dj(i1, "agent_1", "j0001", dict(J1, q0003="five"))
    assert "no better than your stored F1 1.00" in out


def test_memory_is_append_only_so_both_attempts_survive():
    i0 = make("C0")
    dispatch(i0, "agent_1", "claim_job", {"jid": "j0002"})
    dj(i0, "agent_1", "j0002", {"q0003": "five", "q0004": "6"})
    dispatch(i0, "agent_1", "claim_job", {"jid": "j0001"})
    dj(i0, "agent_1", "j0001", J1)
    assert i0.memory.n_answers("agent_1") == 5      # 2 + 3 entries
    assert i0.memory.answer("agent_1", "q0003")["f1"] == 1.0     # best kept on top


def test_shared_memory_at_c2_hands_one_agents_answer_to_another():
    i2 = make("C2")
    dispatch(i2, "agent_1", "claim_job", {"jid": "j0002"})
    dj(i2, "agent_1", "j0002", {"q0003": "4", "q0004": "6"})
    out = dispatch(i2, "agent_2", "claim_job", {"jid": "j0001"})
    assert "memory: stored answer" in out and "✓ q0003" in out


def test_memory_write_and_search_are_free_text_notes():
    i0 = make("C0")
    assert dispatch(i0, "agent_1", "memory_write",
                    {"content": "chalk is a sedimentary rock"}) == "saved to long-term memory"
    out = dispatch(i0, "agent_1", "memory_search", {"query": "chalk rock"})
    assert "- chalk is a sedimentary rock" in out
    assert dispatch(i0, "agent_2", "memory_search",
                    {"query": "chalk rock"}) == "(no matching memories)"


# ---------------- contracts ----------------

def test_dispatch_contract_flow_delivers_to_chat():
    i0 = make("C0")
    dispatch(i0, "agent_1", "propose_contract",
             {"to": "agent_2", "task": "find the capital", "price": 30})
    assert i0.contracts.get("c0001").qid is None       # free-text contract
    dispatch(i0, "agent_2", "accept_contract", {"contract_id": "c0001"})
    dispatch(i0, "agent_2", "deliver_work", {"target_id": "c0001", "content": "it is Paris"})
    assert any("Paris" in m.text for m in i0.chat.unread("agent_1"))
    assert i0.ledger.conservation_ok()


def test_propose_contract_binds_when_the_task_is_exactly_a_qid():
    i0 = make("C0")
    out = dispatch(i0, "agent_1", "propose_contract",
                   {"to": "agent_2", "task": "q0002", "price": 30})
    assert "bound to q0002" in out
    assert i0.contracts.get("c0001").qid == "q0002"
    assert any("bound to q0002" in m.text and "longest river" in m.text
               for m in i0.chat.unread("agent_2"))


def test_a_task_that_merely_mentions_a_qid_does_not_bind():
    """Only an EXACT question id binds -- `task` is free text chosen by the
    proposer, and naming a question in a sentence is not consent to bind."""
    i0 = make("C0")
    out = dispatch(i0, "agent_1", "propose_contract",
                   {"to": "agent_2", "task": "help me with q0002 please", "price": 30})
    assert "bound to" not in out
    assert i0.contracts.get("c0001").qid is None
    dispatch(i0, "agent_2", "accept_contract", {"contract_id": "c0001"})
    good = dispatch(i0, "agent_2", "deliver_work",
                    {"target_id": "c0001", "content": "free text, nothing bound"})
    assert not good.startswith("ERROR")


def test_a_bound_deliverable_is_written_into_both_parties_memories_ungraded():
    i0 = make("C0")
    dispatch(i0, "agent_1", "propose_contract",
             {"to": "agent_2", "task": "q0002", "price": 30})
    dispatch(i0, "agent_2", "accept_contract", {"contract_id": "c0001"})
    out = dispatch(i0, "agent_2", "deliver_work",
                   {"target_id": "c0001", "content": "Loire"})
    assert not out.startswith("ERROR")
    for who in ("agent_1", "agent_2"):
        rec = i0.memory.answer(who, "q0002")
        assert rec["text"] == '[q0002] longest river in France? -> "Loire" ' \
                              '(never graded, from a contract)'
        assert rec["f1"] is None, who
    # ... and the payer can then claim the job and see it on the q0002 row
    claimed = dispatch(i0, "agent_1", "claim_job", {"jid": "j0001"})
    assert "UNVERIFIED" in claimed and "✓ q0002" in claimed
    assert i0.ledger.escrow == {} and i0.ledger.conservation_ok()


def test_a_subcontracted_answer_is_merged_into_the_job_delivery():
    """The behaviour v4.1 exists to enable: a job holder buys one of its
    questions from a peer and ships it inside its own map."""
    i0 = make("C0")
    dispatch(i0, "agent_1", "claim_job", {"jid": "j0001"})
    dispatch(i0, "agent_1", "propose_contract",
             {"to": "agent_2", "task": "q0002", "price": 30})
    dispatch(i0, "agent_2", "accept_contract", {"contract_id": "c0001"})
    dispatch(i0, "agent_2", "deliver_work", {"target_id": "c0001", "content": "Loire"})
    bought = i0.memory.answer("agent_1", "q0002")["text"]
    assert "Loire" in bought
    out = dj(i0, "agent_1", "j0001", J1)
    assert "q0002 F1=1.00 -> 200" in out
    assert i0.ledger.balance("agent_1") == 125 - 30 + 600
    assert i0.ledger.balance("agent_2") == 125 + 30
    assert i0.ledger.conservation_ok()


def test_a_free_text_deliverable_writes_no_answer():
    i0 = make("C0")
    dispatch(i0, "agent_1", "propose_contract",
             {"to": "agent_2", "task": "look something up", "price": 30})
    dispatch(i0, "agent_2", "accept_contract", {"contract_id": "c0001"})
    dispatch(i0, "agent_2", "deliver_work", {"target_id": "c0001", "content": "Loire"})
    assert i0.memory.n_answers("agent_1") == 0 and i0.memory.n_answers("agent_2") == 0


def test_bound_deliverable_is_stored_once_in_the_shared_bucket_at_c2():
    i2 = make("C2")
    dispatch(i2, "agent_1", "propose_contract",
             {"to": "agent_2", "task": "q0002", "price": 30})
    dispatch(i2, "agent_2", "accept_contract", {"contract_id": "c0001"})
    dispatch(i2, "agent_2", "deliver_work", {"target_id": "c0001", "content": "Loire"})
    assert i2.memory.n_answers("agent_1") == 1


# ---------------- misc invariants ----------------

def test_dispatch_error_string_not_exception():
    i0 = make("C0")
    assert dispatch(i0, "agent_1", "claim_job", {"jid": "j9999"}).startswith("ERROR")
    assert dispatch(i0, "agent_1", "deliver_work",
                    {"target_id": "nonsense", "content": "x"}).startswith("ERROR")


def test_visible_tools_filtered():
    names_c0 = {t["name"] for t in visible_tools(CONFIGS["C0"], "agent_1")}
    assert {"claim_job", "list_jobs", "retrieve", "work_on"} <= names_c0
    assert "claim_task" not in names_c0 and "decompose" not in names_c0
    assert "counter_offer" in names_c0 and "set_price" not in names_c0
    names_c1 = {t["name"] for t in visible_tools(CONFIGS["C1"], "agent_1")}
    assert "claim_job" not in names_c1 and "list_jobs" not in names_c1
    assert "retrieve" in names_c1                       # v3: retrieval is never gated
    names_c1i = {t["name"] for t in visible_tools(CONFIGS["C1"], "hub")}
    assert "claim_job" in names_c1i and "retrieve" in names_c1i
    names_c3 = {t["name"] for t in visible_tools(CONFIGS["C3"], "agent_1")}
    assert "counter_offer" not in names_c3 and "set_price" not in names_c3
    assert "claim_job" in names_c3                      # C3 flips pricing only
    names_c3i = {t["name"] for t in visible_tools(CONFIGS["C3"], "hub")}
    assert "set_price" in names_c3i and "counter_offer" not in names_c3i
    assert set(ACTION_SPECS) >= names_c0


def test_collective_goal_changes_no_permissions():
    """C6 centralizes the objective function, not any right: every action must
    behave exactly as it does at C0."""
    assert (visible_tools(CONFIGS["C6"], "agent_1")
            == visible_tools(CONFIGS["C0"], "agent_1"))
    i6, i0 = make("C6"), make("C0")
    probes = [("list_jobs", {}), ("claim_job", {"jid": "j0001"}),
              ("retrieve", {"query": "x"}),
              ("deliver_work", {"target_id": "j0001", "content": "{}"}),
              ("send_message", {"to": "agent_2", "text": "hi"}),
              ("propose_contract", {"to": "agent_2", "task": "sub", "price": 5}),
              ("counter_offer", {"contract_id": "c0001", "price": 9}),
              ("set_price", {"contract_id": "c0001", "price": 9}),
              ("propose_loan", {"to": "agent_2", "amount": 10}),
              ("pay", {"to": "agent_2", "amount": 1}), ("check_balance", {})]
    for name, inp in probes:
        assert (permission_error(i6, "agent_1", name, inp)
                == permission_error(i0, "agent_1", name, inp)), name


def test_shared_memory_changes_reach_not_permissions():
    """C2's difference from C0 is one of REACH (whose answers you can read),
    not of rights: same tool schemas, same gating."""
    assert CONFIGS["C2"].shared_memory is True
    assert (visible_tools(CONFIGS["C2"], "agent_1")
            == visible_tools(CONFIGS["C0"], "agent_1"))
    i2, i0 = make("C2"), make("C0")
    for name, inp in (("retrieve", {"query": "x"}), ("claim_job", {"jid": "j0001"}),
                      ("memory_search", {"query": "x"}),
                      ("propose_contract", {"to": "agent_2", "task": "s", "price": 5}),
                      ("propose_loan", {"to": "agent_2", "amount": 10})):
        assert (permission_error(i2, "agent_1", name, inp)
                == permission_error(i0, "agent_1", name, inp)), name


def test_C7_hides_multi_agent_tool_schemas():
    names = {t["name"] for t in visible_tools(CONFIGS["C7"], "agent_1")}
    assert names == {"retrieve", "work_on", "deliver_work", "list_jobs",
                     "claim_job", "push_goal", "pop_goal",
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


def test_work_on_files_notes_per_question():
    i0 = make("C0")
    out = dispatch(i0, "agent_1", "work_on",
                   {"question_id": "q0001", "thought": "the answer is Paris"})
    assert "q0001" in out and "1 entries" in out
    assert i0.scratchpads["agent_1"]["q0001"] == ["the answer is Paris"]


def test_full_solo_job_flow():
    i0 = make("C0")
    assert "j0001" in dispatch(i0, "agent_1", "list_jobs", {})
    assert not dispatch(i0, "agent_1", "claim_job", {"jid": "j0001"}).startswith("ERROR")
    assert "Paris" in dispatch(i0, "agent_1", "retrieve", {"query": "capital of France"})
    dispatch(i0, "agent_1", "work_on", {"question_id": "q0001", "thought": "Paris"})
    out = dj(i0, "agent_1", "j0001", J1)
    assert "600 tokens" in out
    assert i0.ledger.conservation_ok()
