from ca.config import CONFIGS
from ca.skills import role_skill

_HIRE_HEADING = "### Demo: splitting a claimed job across peers"
_MULTI = [c for c in CONFIGS if CONFIGS[c].n_agents > 1]


def test_worker_demo_matches_config_permissions():
    c0 = role_skill(CONFIGS["C0"], "agent_1")
    assert "claim_job" in c0 and "retrieve" in c0               # full market demo
    c1 = role_skill(CONFIGS["C1"], "agent_1")
    assert "claim_job" not in c1                                 # demand monopoly
    assert "retrieve" in c1                                      # but retrieval is free to all
    c3 = role_skill(CONFIGS["C3"], "agent_1")
    assert "counter_offer" not in c3                             # bargaining disabled
    assert "counter_offer" in role_skill(CONFIGS["C0"], "agent_1")


def test_every_worker_is_taught_retrieve():
    """Info centralization is gone: no config may hide the corpus from a worker."""
    for name in CONFIGS:
        for who in ("agent_1", "hub"):
            if who == "hub" and not CONFIGS[name].has_hub:
                continue
            assert "retrieve" in role_skill(CONFIGS[name], who), (name, who)


def test_v3_tree_verbs_are_gone_everywhere():
    for name in CONFIGS:
        for who in ("agent_1", "hub"):
            s = role_skill(CONFIGS[name], who)
            for dead in ("decompose", "recall_solutions", "claim_task", "list_tasks",
                         "leaf", "subtree", "package"):
                assert dead not in s, (name, who, dead)


def test_world_demos_teach_list_claim_solve_deliver_one_map():
    for s in (role_skill(CONFIGS["C0"], "agent_1"), role_skill(CONFIGS["C1"], "hub")):
        assert "list_jobs" in s and 'claim_job(jid="j0007")' in s
        assert 'deliver_work(target_id="j0007"' in s
        assert '"q0042": "Richard Strauss"' in s
        assert "SHORT ANSWER" in s or "short answer" in s


def test_world_demos_explain_all_or_nothing_delivery():
    s = role_skill(CONFIGS["C0"], "agent_1")
    assert "ALL-OR-NOTHING" in s
    assert "KEEP YOUR CLAIM" in s
    assert "ONE graded attempt" in s


def test_world_demos_do_the_delegation_arithmetic():
    """Jobs come in a range of sizes and claims never expire, so the handbook
    must price the choice: what a job costs you in turns, and why buying a
    question below its own price is profit."""
    for s in (role_skill(CONFIGS["C0"], "agent_1"), role_skill(CONFIGS["C1"], "hub")):
        assert "3 turns" in s or "~24 turns" in s
        assert "never expire" in s or "claim a size you can" in s


def test_world_demos_show_the_auto_recall_on_claim():
    s = role_skill(CONFIGS["C0"], "agent_1")
    assert "memory: stored answer" in s
    assert "GOOD" in s and "LOW QUALITY" in s


def test_memory_block_teaches_writing_intermediate_findings():
    """Spec §9: the agents' own decomposition is the notes they leave."""
    for name in CONFIGS:
        s = role_skill(CONFIGS[name], "agent_1")
        assert "Juanda International" in s, name
        assert "intermediate findings" in s or "WHAT YOU LEARN ON THE WAY" in s, name
    assert "COLLECTIVE asset" in role_skill(CONFIGS["C2"], "agent_1")


def test_memory_block_is_present_for_every_role_and_config():
    for name in CONFIGS:
        for who in ("agent_1", "hub"):
            if who == "hub" and not CONFIGS[name].has_hub:
                continue
            s = role_skill(CONFIGS[name], who)
            assert "memory_write" in s and "memory_search" in s, (name, who)


def test_shared_memory_block_only_at_c2():
    assert "Memory is SHARED at this configuration" in role_skill(CONFIGS["C2"], "agent_1")
    for name in CONFIGS:
        if name == "C2":
            continue
        assert "Memory is SHARED" not in role_skill(CONFIGS[name], "agent_1"), name


def test_only_the_demand_monopoly_config_withholds_the_world_demo():
    """C3/C4/C5 have a hub but no demand monopoly: workers there still claim
    jobs and deliver to the WORLD, so they keep the solo demo."""
    for name in _MULTI:
        s = role_skill(CONFIGS[name], "agent_1")
        assert ("claim_job" in s) == (CONFIGS[name].world_access == "all"), name


def test_hub_demo_matches_config_permissions():
    c1 = role_skill(CONFIGS["C1"], "hub")
    assert "propose_contract" in c1 and "deliver_work" in c1 and "set_price" not in c1
    assert "set_price" in role_skill(CONFIGS["C3"], "hub")
    c7 = role_skill(CONFIGS["C7"], "agent_1")
    assert "propose_contract" not in c7                          # solo: no contracting demo
    assert "claim_job" in c7 and "list_jobs" in c7                # but a full solo demo


def test_hub_bottleneck_claim_only_under_demand_centralization():
    assert "SCARCEST RESOURCE" in role_skill(CONFIGS["C1"], "hub")
    for name in ("C3", "C4", "C5"):
        assert "SCARCEST RESOURCE" not in role_skill(CONFIGS[name], "hub"), name


def test_workers_are_shown_how_to_hire_wherever_contracting_exists():
    # without this, workers only ever see themselves as sellers
    for name in _MULTI:
        s = role_skill(CONFIGS[name], "agent_1")
        assert "propose_contract" in s
        assert s.count(_HIRE_HEADING) == 1
    assert "propose_contract" not in role_skill(CONFIGS["C7"], "agent_1")


def test_hire_demo_teaches_question_binding_by_qid():
    s = role_skill(CONFIGS["C0"], "agent_1")
    assert "BINDS the contract" in s
    assert 'propose_contract(to="agent_5", task="q0107", price=120)' in s
    assert "one contract PER QUESTION" in s


def test_contractors_everywhere_are_taught_bound_delivery():
    for name in _MULTI:
        s = role_skill(CONFIGS[name], "agent_1")
        assert 'deliver_work(target_id="c0007", content="1911")' in s
        assert "bound to q0031" in s


def test_hire_demo_targets_the_hub_under_star_comms():
    s = role_skill(CONFIGS["C5"], "agent_1")
    assert 'propose_contract(to="hub"' in s
    assert 'to="agent_5"' not in s


def test_hire_demo_omits_price_under_central_pricing():
    assert "price=120" in role_skill(CONFIGS["C0"], "agent_1")
    assert "price=120" not in role_skill(CONFIGS["C3"], "agent_1")


def test_worker_borrowing_demo_targets_any_peer_without_credit_centralization():
    for name in ("C0", "C1", "C2", "C3", "C6"):
        s = role_skill(CONFIGS[name], "agent_1")
        assert "propose_loan" in s
        assert "repay_loan" in s
        assert 'propose_loan(to="hub"' not in s


def test_worker_borrowing_demo_targets_hub_under_credit_centralization():
    for name in ("C4", "C5"):
        s = role_skill(CONFIGS[name], "agent_1")
        assert 'propose_loan(to="hub"' in s
        assert "repay_loan" in s


def test_hub_lender_demo_shown_only_under_credit_centralization():
    for name in ("C1", "C3", "C5"):
        assert "SOLE lender" not in role_skill(CONFIGS[name], "hub")
    assert "SOLE lender" in role_skill(CONFIGS["C4"], "hub")


def test_collective_block_only_at_c6():
    s = role_skill(CONFIGS["C6"], "agent_1")
    assert "### Collective mode" in s
    assert "total system balance" in s
    assert "never haggle for margin" in s
    assert "Never duplicate a question" in s
    assert "Split every claimed job" in s
    for name in CONFIGS:
        if name != "C6":
            assert "### Collective mode" not in role_skill(CONFIGS[name], "agent_1"), name


def test_solo_has_no_borrowing_demo():
    s = role_skill(CONFIGS["C7"], "agent_1")
    assert "propose_loan" not in s and "repay_loan" not in s


def test_no_demo_leaks_unformatted_placeholders():
    for name in CONFIGS:
        for who in ("agent_1", "hub"):
            s = role_skill(CONFIGS[name], who)
            assert "{{" not in s and "}}" not in s
            assert "{lender}" not in s and "{price_arg}" not in s and "{peer}" not in s
            assert "{counter_line}" not in s
