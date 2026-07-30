from ca.config import CONFIGS
from ca.skills import role_skill

_HIRE_HEADING = "### Demo: hiring another agent"
_MULTI = [c for c in CONFIGS if CONFIGS[c].n_agents > 1]


def test_worker_demo_matches_config_permissions():
    c0 = role_skill(CONFIGS["C0"], "agent_1")
    assert "claim_task" in c0 and "retrieve" in c0              # full market demo
    c1 = role_skill(CONFIGS["C1"], "agent_1")
    assert "claim_task" not in c1                                # demand monopoly
    assert "retrieve" in c1                                      # but retrieval is free to all
    c3 = role_skill(CONFIGS["C3"], "agent_1")
    assert "counter_offer" not in c3                             # bargaining disabled
    assert "counter_offer" in role_skill(CONFIGS["C0"], "agent_1")


def test_every_worker_is_taught_retrieve_v3():
    """Info centralization is gone: no config may hide the corpus from a worker."""
    for name in CONFIGS:
        for who in ("agent_1", "interface"):
            if who == "interface" and not CONFIGS[name].has_interface:
                continue
            assert "retrieve" in role_skill(CONFIGS[name], who), (name, who)


def test_v1_question_level_verbs_are_gone_everywhere():
    for name in CONFIGS:
        for who in ("agent_1", "interface"):
            s = role_skill(CONFIGS[name], who)
            assert "claim_question" not in s and "list_questions" not in s


def test_world_demos_teach_decompose_then_packaged_json_delivery():
    for s in (role_skill(CONFIGS["C0"], "agent_1"), role_skill(CONFIGS["C1"], "interface")):
        assert "list_tasks" in s and "claim_task" in s and "decompose" in s
        assert 'deliver_work(target_id="t0007"' in s
        assert '"q0031"' in s                       # a JSON package, not a bare answer


def test_contractors_everywhere_are_taught_decompose_and_bound_delivery():
    """Subcontractors get handed subtrees: they must know how to open them."""
    for name in _MULTI:
        s = role_skill(CONFIGS[name], "agent_1")
        assert "decompose" in s
        assert "full leaf coverage" in s


def test_only_the_demand_monopoly_config_withholds_the_world_demo():
    """C3/C4/C5 have an interface but no demand monopoly: workers there still
    claim tasks and deliver to the WORLD, so they keep the solo demo."""
    for name in _MULTI:
        s = role_skill(CONFIGS[name], "agent_1")
        assert ("claim_task" in s) == (CONFIGS[name].world_access == "all"), name


def test_interface_demo_matches_config_permissions():
    c1 = role_skill(CONFIGS["C1"], "interface")
    assert "propose_contract" in c1 and "deliver_work" in c1 and "set_price" not in c1
    assert "set_price" in role_skill(CONFIGS["C3"], "interface")
    c7 = role_skill(CONFIGS["C7"], "agent_1")
    assert "propose_contract" not in c7                          # solo: no contracting demo
    assert "claim_task" in c7 and "decompose" in c7              # but still a full solo demo


def test_interface_bottleneck_claim_only_under_demand_centralization():
    """'your turn is the scarcest resource' is only true when the interface is
    the system's sole income channel (C1), not at C3/C4/C5."""
    assert "SCARCEST RESOURCE" in role_skill(CONFIGS["C1"], "interface")
    for name in ("C3", "C4", "C5"):
        assert "SCARCEST RESOURCE" not in role_skill(CONFIGS[name], "interface"), name


def test_workers_are_shown_how_to_hire_wherever_contracting_exists():
    # without this, workers only ever see themselves as sellers
    for name in _MULTI:
        s = role_skill(CONFIGS[name], "agent_1")
        assert "propose_contract" in s
        assert s.count(_HIRE_HEADING) == 1
    assert "propose_contract" not in role_skill(CONFIGS["C7"], "agent_1")


def test_hire_demo_teaches_subtree_binding_by_sentence():
    s = role_skill(CONFIGS["C0"], "agent_1")
    assert "BINDS the contract" in s
    assert 'propose_contract(to="agent_5", task="date the two premieres", price=120)' in s


def test_hire_demo_targets_the_hub_under_star_comms():
    s = role_skill(CONFIGS["C5"], "agent_1")
    assert 'propose_contract(to="interface"' in s
    assert 'to="agent_5"' not in s


def test_hire_demo_omits_price_under_central_pricing():
    assert "price=120" in role_skill(CONFIGS["C0"], "agent_1")
    assert "price=120" not in role_skill(CONFIGS["C3"], "agent_1")


def test_worker_borrowing_demo_targets_any_peer_without_credit_centralization():
    for name in ("C0", "C1", "C2", "C3", "C6"):
        s = role_skill(CONFIGS[name], "agent_1")
        assert "propose_loan" in s
        assert "repay_loan" in s
        assert 'propose_loan(to="interface"' not in s


def test_worker_borrowing_demo_targets_interface_under_credit_centralization():
    for name in ("C4", "C5"):
        s = role_skill(CONFIGS[name], "agent_1")
        assert 'propose_loan(to="interface"' in s
        assert "repay_loan" in s


def test_interface_lender_demo_shown_only_under_credit_centralization():
    for name in ("C1", "C3", "C5"):
        assert "SOLE lender" not in role_skill(CONFIGS[name], "interface")
    assert "SOLE lender" in role_skill(CONFIGS["C4"], "interface")


def test_collective_block_only_at_c6():
    s = role_skill(CONFIGS["C6"], "agent_1")
    assert "### Collective mode" in s
    assert "total system balance" in s
    assert "never haggle for margin" in s
    assert "Never duplicate a task" in s
    for name in CONFIGS:
        if name != "C6":
            assert "### Collective mode" not in role_skill(CONFIGS[name], "agent_1"), name


def test_solo_has_no_borrowing_demo():
    s = role_skill(CONFIGS["C7"], "agent_1")
    assert "propose_loan" not in s and "repay_loan" not in s


def test_no_demo_leaks_unformatted_placeholders():
    for name in CONFIGS:
        for who in ("agent_1", "interface"):
            s = role_skill(CONFIGS[name], who)
            assert "{{" not in s and "}}" not in s
            assert "{lender}" not in s and "{price_arg}" not in s and "{peer}" not in s
            assert "{bottleneck}" not in s and "{counter_line}" not in s
