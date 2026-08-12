from fixtures import DEMO_CORPUS, HashEmbedding, demo_corpus_embeddings, demo_infra

from ca.actions import ACTION_SPECS, classify, dispatch, permission_error, visible_tools
from ca.bank import Question, QuestionBank
from ca.config import CONFIGS, ExperimentConfig
from ca.infra import Infra


def make(level="C0", **kw):
    return demo_infra(level, **kw)


def wide(n=25):
    """A bank of n questions, for pagination checks."""
    qs = [Question(f"q{i:04d}", f"question {i}", ["x"], "2hop", 100 + i, "k00")
          for i in range(1, n + 1)]
    cfg = ExperimentConfig(level=CONFIGS["C0"], seed=0)
    return Infra(cfg, QuestionBank(qs), corpus=DEMO_CORPUS,
                 corpus_embeddings=demo_corpus_embeddings(),
                 embedding_function=HashEmbedding())


# ---------------- the catalog ----------------

def test_the_action_catalog_is_exactly_the_v6_eleven():
    assert set(ACTION_SPECS) == {
        "memory_search", "deliver_work",
        "list_questions", "claim_question", "release_question", "memory_write",
        "send_message", "read_chat", "push_goal", "pop_goal", "list_agents"}


def test_the_dead_actions_are_gone():
    for dead in ("retrieve", "list_jobs", "claim_job", "work_on", "decompose",
                 "recall_solutions", "list_tasks", "claim_task",
                 "propose_contract", "accept_contract", "reject_contract",
                 "counter_offer", "cancel_contract", "set_price", "pay",
                 "propose_loan", "accept_loan", "repay_loan", "check_balance"):
        assert dead not in ACTION_SPECS


def test_no_module_imports_the_deleted_economy():
    import pathlib
    import re
    root = pathlib.Path(__file__).resolve().parent.parent
    pattern = re.compile(r"\b(?:import|from)\s+(?:ca\.)?(economy|contracts|loans)\b")
    offenders = []
    for path in list((root / "src").rglob("*.py")) + list((root / "tests").rglob("*.py")) \
            + list((root / "scripts").rglob("*.py")):
        if pattern.search(path.read_text()):
            offenders.append(str(path.relative_to(root)))
    assert offenders == []


# ---------------- classification & gating ----------------

def test_classify():
    assert classify("memory_search", {"query": "x"}) == "solving"
    assert classify("deliver_work", {"target_id": "q0001", "content": "Paris"}) == "solving"
    assert classify("claim_question", {"qid": "q0001"}) == "admin"
    assert classify("release_question", {"qid": "q0001"}) == "admin"
    assert classify("memory_write", {"content": "x"}) == "admin"
    assert classify("send_message", {"to": "a", "text": "x"}) == "admin"
    assert classify("list_agents", {}) == "admin"


def test_world_gating_by_level():
    i0 = make("C0")
    assert permission_error(i0, "agent_1", "claim_question", {"qid": "q0001"}) is None
    assert permission_error(i0, "agent_1", "list_questions", {}) is None
    i1 = make("C1")
    for name, inp in (("claim_question", {"qid": "q0001"}), ("list_questions", {}),
                      ("release_question", {"qid": "q0001"}),
                      ("deliver_work", {"target_id": "q0001", "content": "Paris"})):
        assert permission_error(i1, "agent_1", name, inp) is not None, name
        assert permission_error(i1, "hub", name, inp) is None, name


def test_memory_search_is_open_to_everyone_at_every_config():
    """Knowledge is infrastructure: no config may hide the corpus-seeded
    memory from an agent."""
    for name in CONFIGS:
        infra = make(name)
        for who in ("agent_1", "hub"):
            if who == "hub" and not CONFIGS[name].has_hub:
                continue
            assert permission_error(infra, who, "memory_search", {"query": "x"}) is None
            assert "memory_search" in {t["name"] for t in visible_tools(CONFIGS[name], who)}


def test_star_comms_gating():
    i5 = make("C5")
    assert permission_error(i5, "agent_1", "send_message", {"to": "agent_2", "text": "hi"}) is not None
    assert permission_error(i5, "agent_1", "send_message", {"to": "hub", "text": "hi"}) is None
    assert permission_error(i5, "agent_1", "read_chat", {"with_agent": "agent_2"}) is not None
    assert permission_error(i5, "agent_1", "read_chat", {"with_agent": "hub"}) is None
    assert permission_error(i5, "hub", "send_message", {"to": "agent_2", "text": "hi"}) is None


def test_comms_stay_free_where_they_are_not_the_flipped_mechanism():
    for name in ("C0", "C1", "C2"):
        infra = make(name)
        assert permission_error(infra, "agent_1", "send_message",
                                {"to": "agent_2", "text": "hi"}) is None, name


def test_world_access_is_open_wherever_it_is_not_the_flipped_mechanism():
    for name in ("C0", "C2", "C5"):
        infra = make(name)
        assert permission_error(infra, "agent_1", "claim_question",
                                {"qid": "q0001"}) is None, name
        assert permission_error(infra, "agent_1", "list_questions", {}) is None, name


# ---------------- board actions ----------------

def test_list_questions_shows_text_and_difficulty_only():
    i0 = make("C0")
    out = dispatch(i0, "agent_1", "list_questions", {})
    assert sorted(out.splitlines()) == sorted([
        "[q0001] capital of France? (2hop)",
        "[q0002] longest river in France? (2hop)",
        "[q0003] 2+2? (2hop)",
        "[q0004] 3+3? (3hop)",
        "[q0005] which rock type is chalk? (2hop)",
    ])
    assert dispatch(i0, "agent_1", "list_questions", {}) == out     # stable per viewer


def test_list_questions_order_differs_between_viewers_but_covers_the_same_set():
    i0 = wide(25)
    a = dispatch(i0, "agent_1", "list_questions", {"offset": 0}).splitlines()[:20]
    b = dispatch(i0, "agent_2", "list_questions", {"offset": 0}).splitlines()[:20]
    assert sorted(a) != sorted(b) or a != b     # de-herding: no shared "top" question
    assert a != b


def test_list_questions_drops_a_question_the_moment_someone_holds_it():
    i0 = make("C0")
    dispatch(i0, "agent_1", "claim_question", {"qid": "q0002"})
    assert "q0002" not in dispatch(i0, "agent_2", "list_questions", {})
    assert "q0001" in dispatch(i0, "agent_2", "list_questions", {})


def test_list_questions_pagination_offset():
    infra = wide(25)
    first = dispatch(infra, "agent_1", "list_questions", {})
    lines = first.splitlines()
    assert len(lines) == 21                                   # 20 questions + overflow note
    assert lines[-1] == "... and 5 more (call list_questions with offset=20 to see them)"
    second = dispatch(infra, "agent_1", "list_questions", {"offset": 20})
    assert second.count("[q0") == 5 and "more" not in second
    import re
    seen = re.findall(r"\[q\d{4}\]", first) + re.findall(r"\[q\d{4}\]", second)
    assert len(seen) == 25 and len(set(seen)) == 25
    empty = dispatch(infra, "agent_1", "list_questions", {"offset": 99})
    assert "25 open in total" in empty


def test_claim_question_shows_text_and_the_deliver_hint():
    i0 = make("C0")
    out = dispatch(i0, "agent_1", "claim_question", {"qid": "q0001"})
    assert out.splitlines() == [
        "claimed [q0001] capital of France? (2hop)",
        'deliver ONE short answer: deliver_work(target_id="q0001", content="<answer>")',
    ]
    assert "memory:" not in out


def test_only_one_agent_may_hold_a_question():
    i0 = make("C0")
    assert not dispatch(i0, "agent_1", "claim_question", {"qid": "q0001"}).startswith("ERROR")
    out = dispatch(i0, "agent_2", "claim_question", {"qid": "q0001"})
    assert out.startswith("ERROR") and "another agent" in out


def test_claim_unknown_qid_lists_near_ids():
    i0 = make("C0")
    out = dispatch(i0, "agent_1", "claim_question", {"qid": "q9999"})
    assert out.startswith("ERROR") and "q0005" in out


def test_two_strikes_close_a_question_to_one_agent_but_not_to_others():
    i0 = make("C0")
    dispatch(i0, "agent_1", "claim_question", {"qid": "q0001"})
    dispatch(i0, "agent_1", "release_question", {"qid": "q0001"})
    dispatch(i0, "agent_1", "claim_question", {"qid": "q0001"})
    dispatch(i0, "agent_1", "release_question", {"qid": "q0001"})
    third = dispatch(i0, "agent_1", "claim_question", {"qid": "q0001"})
    assert third.startswith("ERROR") and "twice" in third
    assert not dispatch(i0, "agent_2", "claim_question", {"qid": "q0001"}).startswith("ERROR")


# ---------------- release ----------------

def test_release_hands_the_question_back_to_the_open_board():
    i0 = make("C0")
    dispatch(i0, "agent_1", "claim_question", {"qid": "q0002"})
    assert "q0002" not in dispatch(i0, "agent_2", "list_questions", {})
    out = dispatch(i0, "agent_1", "release_question", {"qid": "q0002"})
    assert out == "released q0002; it is open on the board again for any agent"
    assert "q0002" in dispatch(i0, "agent_2", "list_questions", {})
    assert not dispatch(i0, "agent_2", "claim_question", {"qid": "q0002"}).startswith("ERROR")


def test_release_of_a_question_you_do_not_hold_is_refused():
    i0 = make("C0")
    out = dispatch(i0, "agent_1", "release_question", {"qid": "q0002"})
    assert out.startswith("ERROR") and "no active claim" in out
    dispatch(i0, "agent_1", "claim_question", {"qid": "q0002"})
    assert dispatch(i0, "agent_2", "release_question", {"qid": "q0002"}).startswith("ERROR")
    assert i0.board.active["q0002"].agent == "agent_1"


def test_release_of_an_unknown_qid_lists_near_ids():
    i0 = make("C0")
    out = dispatch(i0, "agent_1", "release_question", {"qid": "q9999"})
    assert out.startswith("ERROR") and "q0005" in out


# ---------------- delivery to WORLD ----------------

def test_delivery_grades_and_closes_the_question():
    i0 = make("C0")
    dispatch(i0, "agent_1", "claim_question", {"qid": "q0001"})
    out = dispatch(i0, "agent_1", "deliver_work",
                   {"target_id": "q0001", "content": "Paris"})
    assert out == "delivered q0001: F1 1.00"
    assert i0.board.closed == {"q0001"} and len(i0.board.results) == 1


def test_delivery_without_a_claim_is_refused():
    i0 = make("C0")
    out = dispatch(i0, "agent_1", "deliver_work",
                   {"target_id": "q0001", "content": "Paris"})
    assert out.startswith("ERROR") and "claim_question" in out


def test_delivery_after_release_is_refused():
    i0 = make("C0")
    dispatch(i0, "agent_1", "claim_question", {"qid": "q0001"})
    dispatch(i0, "agent_1", "release_question", {"qid": "q0001"})
    assert dispatch(i0, "agent_1", "deliver_work",
                    {"target_id": "q0001", "content": "Paris"}).startswith("ERROR")


def test_second_delivery_on_one_claim_is_refused():
    i0 = make("C0")
    dispatch(i0, "agent_1", "claim_question", {"qid": "q0001"})
    dispatch(i0, "agent_1", "deliver_work", {"target_id": "q0001", "content": "Paris"})
    assert dispatch(i0, "agent_1", "deliver_work",
                    {"target_id": "q0001", "content": "Paris"}).startswith("ERROR")


def test_partial_f1_is_reported():
    i0 = make("C0")
    dispatch(i0, "agent_1", "claim_question", {"qid": "q0002"})
    out = dispatch(i0, "agent_1", "deliver_work",
                   {"target_id": "q0002", "content": "Loire River"})
    assert out == "delivered q0002: F1 0.67"


def test_delivering_an_unknown_target_lists_near_ids():
    i0 = make("C0")
    out = dispatch(i0, "agent_1", "deliver_work",
                   {"target_id": "nonsense", "content": "x"})
    assert out.startswith("ERROR") and "list_questions" in out
    out = dispatch(i0, "agent_1", "deliver_work",
                   {"target_id": "q9999", "content": "x"})
    assert out.startswith("ERROR") and "q0005" in out
    assert i0.board.results == []


# ---------------- automatic memory ----------------

def test_delivery_writes_the_graded_answer_into_memory():
    i0 = make("C0")
    dispatch(i0, "agent_1", "claim_question", {"qid": "q0001"})
    dispatch(i0, "agent_1", "deliver_work", {"target_id": "q0001", "content": "Paris"})
    rec = i0.memory.answer("agent_1", "q0001")
    assert rec == {"text": '[q0001] capital of France? -> "Paris" (F1 1.00)',
                   "kind": "answer", "qid": "q0001", "f1": 1.0, "title": None}
    assert i0.memory.n_answers("agent_1") == 1
    assert i0.memory.answer("agent_2", "q0001") is None      # private at C0


def test_claim_auto_recalls_the_stored_answer():
    i0 = make("C0")
    i0.memory.write("agent_1", '[q0003] 2+2? -> "4" (F1 1.00)',
                    kind="answer", qid="q0003", f1=1.0)
    out = dispatch(i0, "agent_1", "claim_question", {"qid": "q0003"})
    assert "memory: stored answer" in out
    assert '"4" (F1 1.00)' in out and "GOOD (F1 1.00" in out


def test_claim_flags_a_low_quality_stored_answer_as_worth_re_solving():
    i0 = make("C0")
    i0.memory.write("agent_1", '[q0003] 2+2? -> "seven" (F1 0.00)',
                    kind="answer", qid="q0003", f1=0.0)
    out = dispatch(i0, "agent_1", "claim_question", {"qid": "q0003"})
    assert "LOW QUALITY (F1 0.00 of 1.00)" in out and "re-solve" in out


def test_corpus_entries_never_trigger_the_auto_recall():
    """The corpus paragraph about Paris is knowledge, not a stored ANSWER: a
    fresh claim of q0001 must come back bare."""
    i0 = make("C0")
    out = dispatch(i0, "agent_1", "claim_question", {"qid": "q0001"})
    assert "memory:" not in out and "Paris" not in out


def test_a_repeat_delivery_reports_whether_it_beat_the_stored_f1():
    i0 = make("C0")
    i0.memory.write("agent_1", "old try", kind="answer", qid="q0001", f1=0.0)
    dispatch(i0, "agent_1", "claim_question", {"qid": "q0001"})
    out = dispatch(i0, "agent_1", "deliver_work", {"target_id": "q0001", "content": "Paris"})
    assert "IMPROVED on your stored F1 0.00" in out

    i1 = make("C0")
    i1.memory.write("agent_1", "old try", kind="answer", qid="q0001", f1=1.0)
    dispatch(i1, "agent_1", "claim_question", {"qid": "q0001"})
    out = dispatch(i1, "agent_1", "deliver_work", {"target_id": "q0001", "content": "Lyon"})
    assert "no better than your stored F1 1.00" in out


def test_memory_is_append_only_so_both_attempts_survive():
    i0 = make("C0")
    i0.memory.write("agent_1", "old try", kind="answer", qid="q0001", f1=0.0)
    dispatch(i0, "agent_1", "claim_question", {"qid": "q0001"})
    dispatch(i0, "agent_1", "deliver_work", {"target_id": "q0001", "content": "Paris"})
    assert i0.memory.n_answers("agent_1") == 2
    assert i0.memory.answer("agent_1", "q0001")["f1"] == 1.0     # best kept on top


def test_shared_memory_at_c2_hands_one_agents_answer_to_another():
    i2 = make("C2")
    dispatch(i2, "agent_1", "claim_question", {"qid": "q0003"})
    dispatch(i2, "agent_1", "deliver_work", {"target_id": "q0003", "content": "4"})
    out = dispatch(i2, "agent_2", "claim_question", {"qid": "q0004"})
    assert "memory:" not in out                       # different question
    # q0003 is closed; the SAME answer surfaces on a search instead
    hits = dispatch(i2, "agent_2", "memory_search", {"query": "2+2?"})
    assert '"4" (F1 1.00)' in hits


# ---------------- memory as the single knowledge query ----------------

def test_memory_search_reaches_the_born_in_corpus():
    i0 = make("C0")
    out = dispatch(i0, "agent_1", "memory_search", {"query": "capital of France."})
    assert out.splitlines()[0] == "- [Paris] Paris is the capital of France."


def test_memory_search_mixes_corpus_notes_and_answers():
    i0 = make("C0")
    dispatch(i0, "agent_1", "memory_write", {"content": "chalk note: white cliffs"})
    dispatch(i0, "agent_1", "claim_question", {"qid": "q0005"})
    dispatch(i0, "agent_1", "deliver_work", {"target_id": "q0005", "content": "sedimentary"})
    out = dispatch(i0, "agent_1", "memory_search",
                   {"query": "which rock type is chalk? sedimentary white"})
    assert "- [Chalk] Chalk is a sedimentary rock type." in out       # corpus
    assert "- chalk note: white cliffs" in out                        # note
    assert '- [q0005] which rock type is chalk? -> "sedimentary" (F1 1.00)' in out


def test_memory_write_and_search_notes_stay_private_at_c0():
    i0 = make("C0")
    assert dispatch(i0, "agent_1", "memory_write",
                    {"content": "private note: the chalk cliffs are white"}) \
        == "saved to long-term memory"
    assert "chalk cliffs are white" in dispatch(
        i0, "agent_1", "memory_search", {"query": "private note: the chalk cliffs are white"})
    # agent_2 sees the same CORPUS but never agent_1's note
    other = dispatch(i0, "agent_2", "memory_search",
                     {"query": "private note: the chalk cliffs are white"})
    assert "chalk cliffs are white" not in other


def test_memory_search_k_is_a_config_knob():
    infra = make("C0", memory_k=2)
    for i in range(4):
        dispatch(infra, "agent_1", "memory_write", {"content": f"apple banana note {i}"})
    out = dispatch(infra, "agent_1", "memory_search", {"query": "apple banana note"})
    assert len(out.splitlines()) == 2


# ---------------- chat ----------------

def test_a_peer_can_be_asked_and_can_answer_from_memory():
    i0 = make("C0")
    assert dispatch(i0, "agent_1", "send_message",
                    {"to": "agent_2", "text": "what rock type is chalk?"}) == "sent to agent_2"
    assert any("chalk" in m.text for m in i0.chat.unread("agent_2"))
    found = dispatch(i0, "agent_2", "memory_search", {"query": "chalk rock type"})
    assert "sedimentary" in found
    dispatch(i0, "agent_2", "send_message", {"to": "agent_1", "text": "sedimentary"})
    assert [m.text for m in i0.chat.unread("agent_1")] == ["sedimentary"]
    assert "sedimentary" in dispatch(i0, "agent_1", "read_chat", {"with_agent": "agent_2"})


def test_unknown_agent_error_lists_roster():
    i0 = make("C0")
    out = dispatch(i0, "agent_1", "send_message", {"to": "agent_99", "text": "x"})
    assert out.startswith("ERROR") and "valid agents" in out and "agent_2" in out


# ---------------- misc invariants ----------------

def test_dispatch_error_string_not_exception():
    i0 = make("C0")
    assert dispatch(i0, "agent_1", "claim_question", {"qid": "q9999"}).startswith("ERROR")
    assert dispatch(i0, "agent_1", "deliver_work",
                    {"target_id": "nonsense", "content": "x"}).startswith("ERROR")


def test_visible_tools_filtered():
    names_c0 = {t["name"] for t in visible_tools(CONFIGS["C0"], "agent_1")}
    assert {"claim_question", "list_questions", "release_question",
            "memory_search"} <= names_c0
    assert "claim_job" not in names_c0 and "retrieve" not in names_c0
    names_c1 = {t["name"] for t in visible_tools(CONFIGS["C1"], "agent_1")}
    assert names_c1 == {"memory_search", "memory_write", "send_message",
                        "read_chat", "push_goal", "pop_goal", "list_agents"}
    names_c1i = {t["name"] for t in visible_tools(CONFIGS["C1"], "hub")}
    assert {"claim_question", "deliver_work", "memory_search"} <= names_c1i
    assert set(ACTION_SPECS) >= names_c0


def test_shared_memory_changes_reach_not_permissions():
    """C2's difference from C0 is one of REACH (whose answers you can read),
    not of rights: same tool schemas, same gating."""
    assert CONFIGS["C2"].shared_memory is True
    assert (visible_tools(CONFIGS["C2"], "agent_1")
            == visible_tools(CONFIGS["C0"], "agent_1"))
    i2, i0 = make("C2"), make("C0")
    for name, inp in (("memory_search", {"query": "x"}),
                      ("claim_question", {"qid": "q0001"}),
                      ("release_question", {"qid": "q0001"}),
                      ("send_message", {"to": "agent_2", "text": "s"})):
        assert (permission_error(i2, "agent_1", name, inp)
                == permission_error(i0, "agent_1", name, inp)), name


def test_C7_hides_multi_agent_tool_schemas():
    names = {t["name"] for t in visible_tools(CONFIGS["C7"], "agent_1")}
    assert names == {"deliver_work", "list_questions", "claim_question",
                     "release_question", "push_goal", "pop_goal",
                     "memory_write", "memory_search"}


def test_full_solo_question_flow():
    i0 = make("C0")
    assert "q0001" in dispatch(i0, "agent_1", "list_questions", {})
    assert not dispatch(i0, "agent_1", "claim_question", {"qid": "q0001"}).startswith("ERROR")
    assert "Paris" in dispatch(i0, "agent_1", "memory_search", {"query": "capital of France."})
    dispatch(i0, "agent_1", "memory_write", {"content": "q0001: it is Paris"})
    out = dispatch(i0, "agent_1", "deliver_work", {"target_id": "q0001", "content": "Paris"})
    assert out == "delivered q0001: F1 1.00"
