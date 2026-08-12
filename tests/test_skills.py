from ca.config import CONFIGS
from ca.skills import role_skill

_ASK_HEADING = "### Demo: asking a peer instead of searching alone"
_ANSWER_HEADING = "### Demo: answering a peer who asks you"
_RELEASE_HEADING = "### Demo: handing a question back"
_MULTI = [c for c in CONFIGS if CONFIGS[c].n_agents > 1]

MONEY_WORDS = ("balance", "reward", "price", "pay", "paid", "afford", "escrow",
               "contract", "loan", "borrow", "lend", "profit", "income",
               "bankrupt", "tokens", "buy", "sell", "money")


def _roles():
    for name, level in CONFIGS.items():
        for who in ("agent_1", "hub"):
            if who == "hub" and not level.has_hub:
                continue
            yield name, who, role_skill(level, who)


def test_no_handbook_contains_money_vocabulary():
    for name, who, s in _roles():
        low = s.lower()
        for word in MONEY_WORDS:
            assert word not in low, (name, who, word)


def test_dead_verbs_are_gone_everywhere():
    for name, who, s in _roles():
        for dead in ("retrieve", "list_jobs", "claim_job", "work_on",
                     "decompose", "recall_solutions", "claim_task", "list_tasks",
                     "propose_contract", "accept_contract", "set_price",
                     "propose_loan", "repay_loan", "check_balance",
                     "JSON", "all-or-nothing", "ALL-OR-NOTHING"):
            assert dead not in s, (name, who, dead)


def test_worker_demo_matches_config_permissions():
    c0 = role_skill(CONFIGS["C0"], "agent_1")
    assert "claim_question" in c0 and "memory_search" in c0
    c1 = role_skill(CONFIGS["C1"], "agent_1")
    assert "claim_question" not in c1                            # board monopoly
    assert "memory_search" in c1                                 # but memory is free to all


def test_world_demos_teach_list_claim_search_deliver_one_answer():
    for s in (role_skill(CONFIGS["C0"], "agent_1"), role_skill(CONFIGS["C1"], "hub")):
        assert "list_questions" in s and 'claim_question(qid="q0107")' in s
        assert 'deliver_work(target_id="q0107", content="1911")' in s
        assert "SHORT ANSWER" in s or "short answer" in s
    assert "ONE graded attempt" in role_skill(CONFIGS["C0"], "agent_1")


def test_world_demos_show_the_auto_recall_on_claim():
    s = role_skill(CONFIGS["C0"], "agent_1")
    assert "memory: stored answer" in s
    assert "GOOD" in s and "LOW QUALITY" in s
    assert "CUT LOSSES" in s


def test_memory_block_teaches_corpus_search_and_intermediate_findings():
    for name in CONFIGS:
        s = role_skill(CONFIGS[name], "agent_1")
        assert "born knowing the corpus" in s, name
        assert "Juanda International" in s, name
        assert "intermediate findings" in s or "WHAT YOU LEARN ON THE WAY" in s, name
    assert "COLLECTIVE asset" in role_skill(CONFIGS["C2"], "agent_1")


def test_memory_block_is_present_for_every_role_and_config():
    for name, who, s in _roles():
        assert "memory_write" in s and "memory_search" in s, (name, who)


def test_shared_memory_block_only_at_c2():
    assert "Memory is SHARED at this configuration" in role_skill(CONFIGS["C2"], "agent_1")
    for name in CONFIGS:
        if name == "C2":
            continue
        assert "Memory is SHARED" not in role_skill(CONFIGS[name], "agent_1"), name


def test_only_the_board_monopoly_config_withholds_the_world_demo():
    """C5 has a hub but no board monopoly: workers there still claim questions
    and deliver to the WORLD, so they keep the full pipeline demo."""
    for name in _MULTI:
        s = role_skill(CONFIGS[name], "agent_1")
        assert ("claim_question" in s) == (CONFIGS[name].world_access == "all"), name


# ---------------- cooperation demos ----------------

def test_every_agent_with_peers_is_taught_to_answer_them():
    for name in _MULTI:
        for who in ("agent_1", "hub"):
            if who == "hub" and not CONFIGS[name].has_hub:
                continue
            s = role_skill(CONFIGS[name], who)
            assert s.count(_ANSWER_HEADING) == 1, (name, who)
            assert "send_message" in s and "memory_search" in s
    assert _ANSWER_HEADING not in role_skill(CONFIGS["C7"], "agent_1")


def test_agents_who_can_deliver_are_taught_to_ask_a_peer():
    for name in _MULTI:
        s = role_skill(CONFIGS[name], "agent_1")
        # C1 workers cannot deliver, so asking a peer for an answer is not their move
        assert (_ASK_HEADING in s) == (CONFIGS[name].world_access == "all"), name
    assert _ASK_HEADING in role_skill(CONFIGS["C1"], "hub")
    assert _ASK_HEADING not in role_skill(CONFIGS["C7"], "agent_1")


def test_ask_demo_names_the_fact_not_the_question_id():
    s = role_skill(CONFIGS["C0"], "agent_1")
    assert "NAMING THE FACT YOU NEED" in s
    assert 'send_message(to="agent_5"' in s


def test_peer_demos_target_the_hub_under_star_comms():
    s = role_skill(CONFIGS["C5"], "agent_1")
    assert 'send_message(to="hub"' in s
    assert 'to="agent_5"' not in s


def test_c1_workers_are_told_the_hub_is_the_one_who_asks():
    """At C1 the hub holds every question, so it is the only agent a worker
    can usefully answer."""
    s = role_skill(CONFIGS["C1"], "agent_1")
    assert 'send_message(to="hub"' in s
    assert 'to="agent_5"' not in s


def test_the_hub_asks_a_worker_not_itself():
    s = role_skill(CONFIGS["C1"], "hub")
    assert 'send_message(to="agent_3"' in s
    assert 'to="hub"' not in s


def test_release_demo_wherever_the_board_is_reachable():
    for name, who, s in _roles():
        can_world = CONFIGS[name].world_access == "all" or who == "hub"
        assert (_RELEASE_HEADING in s) == can_world, (name, who)
        if can_world:
            assert 'release_question(qid="q0107")' in s


def test_hub_demo_matches_config_permissions():
    c1 = role_skill(CONFIGS["C1"], "hub")
    assert "deliver_work" in c1 and "send_message" in c1
    c7 = role_skill(CONFIGS["C7"], "agent_1")
    assert "send_message" not in c7                              # solo: nobody to talk to
    assert "claim_question" in c7 and "list_questions" in c7      # but a full solo demo


def test_hub_bottleneck_claim_only_under_board_centralization():
    assert "SCARCEST RESOURCE" in role_skill(CONFIGS["C1"], "hub")
    assert "SCARCEST RESOURCE" not in role_skill(CONFIGS["C5"], "hub")


def test_no_demo_leaks_unformatted_placeholders():
    for name, who, s in _roles():
        assert "{{" not in s and "}}" not in s
        assert "{peer}" not in s and "{asker}" not in s
