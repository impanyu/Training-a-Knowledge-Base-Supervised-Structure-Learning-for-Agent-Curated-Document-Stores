from ca.config import LEVELS
from ca.skills import role_skill

_HIRE_HEADING = "### Demo: hiring another agent"


def test_worker_demo_matches_level_permissions():
    l0 = role_skill(LEVELS["L0"], "agent_1")
    assert "claim_task" in l0 and "retrieve" in l0              # full market demo
    l1 = role_skill(LEVELS["L1"], "agent_1")
    assert "claim_task" not in l1 and "retrieve" in l1          # contractor demo, may retrieve
    l2 = role_skill(LEVELS["L2"], "agent_1")
    assert "retrieve" not in l2                                  # info monopoly: no retrieve demo
    assert "propose_contract" in l2                              # buy info from interface instead
    l3 = role_skill(LEVELS["L3"], "agent_1")
    assert "counter_offer" not in l3                             # bargaining disabled
    assert "counter_offer" in role_skill(LEVELS["L0"], "agent_1")


def test_v1_question_level_verbs_are_gone_everywhere():
    for lvl in LEVELS:
        for who in ("agent_1", "interface"):
            s = role_skill(LEVELS[lvl], who)
            assert "claim_question" not in s and "list_questions" not in s


def test_world_demos_teach_decompose_then_packaged_json_delivery():
    for s in (role_skill(LEVELS["L0"], "agent_1"), role_skill(LEVELS["L1"], "interface")):
        assert "list_tasks" in s and "claim_task" in s and "decompose" in s
        assert 'deliver_work(target_id="t0007"' in s
        assert '"q0031"' in s                       # a JSON package, not a bare answer


def test_contractors_everywhere_are_taught_decompose_and_bound_delivery():
    """Subcontractors get handed subtrees: they must know how to open them."""
    for lvl in ("L0", "L1", "L2", "L3", "L4", "L5"):
        s = role_skill(LEVELS[lvl], "agent_1")
        assert "decompose" in s
        assert "full leaf coverage" in s


def test_interface_demo_matches_level_permissions():
    l1 = role_skill(LEVELS["L1"], "interface")
    assert "propose_contract" in l1 and "deliver_work" in l1 and "set_price" not in l1
    l3 = role_skill(LEVELS["L3"], "interface")
    assert "set_price" in l3
    l6 = role_skill(LEVELS["L6"], "agent_1")
    assert "propose_contract" not in l6                          # solo: no contracting demo
    assert "claim_task" in l6 and "decompose" in l6              # but still a full solo demo


def test_l0_and_l1_workers_are_shown_how_to_hire():
    # without this, decentralised workers only ever see themselves as sellers
    for lvl in ("L0", "L1"):
        s = role_skill(LEVELS[lvl], "agent_1")
        assert "propose_contract" in s
        assert s.count(_HIRE_HEADING) == 1
    # L2/L3 workers already learn hiring from the buy-info demo: do not double up
    for lvl in ("L2", "L3", "L4"):
        assert role_skill(LEVELS[lvl], "agent_1").count(_HIRE_HEADING) == 0
    assert "propose_contract" not in role_skill(LEVELS["L6"], "agent_1")


def test_hire_demo_teaches_subtree_binding_by_sentence():
    s = role_skill(LEVELS["L0"], "agent_1")
    assert "BINDS the contract" in s
    assert 'propose_contract(to="agent_5", task="date the two premieres", price=120)' in s


def test_buy_info_demo_omits_price_under_central_pricing():
    assert "price=80" in role_skill(LEVELS["L2"], "agent_1")
    assert "price=80" not in role_skill(LEVELS["L3"], "agent_1")


def test_worker_borrowing_demo_targets_any_peer_below_credit_centralization():
    for lvl in ("L0", "L1", "L2", "L3"):
        s = role_skill(LEVELS[lvl], "agent_1")
        assert "propose_loan" in s
        assert "repay_loan" in s
        assert 'propose_loan(to="interface"' not in s


def test_worker_borrowing_demo_targets_interface_under_credit_centralization():
    for lvl in ("L4", "L5"):
        s = role_skill(LEVELS[lvl], "agent_1")
        assert 'propose_loan(to="interface"' in s
        assert "repay_loan" in s


def test_interface_lender_demo_shown_only_at_credit_levels():
    for lvl in ("L0", "L1", "L2", "L3"):
        assert "SOLE lender" not in role_skill(LEVELS[lvl], "interface")
    for lvl in ("L4", "L5"):
        assert "SOLE lender" in role_skill(LEVELS[lvl], "interface")


def test_solo_has_no_borrowing_demo():
    s = role_skill(LEVELS["L6"], "agent_1")
    assert "propose_loan" not in s and "repay_loan" not in s


def test_no_demo_leaks_unformatted_placeholders():
    for lvl in LEVELS:
        for who in ("agent_1", "interface"):
            s = role_skill(LEVELS[lvl], who)
            assert "{{" not in s and "}}" not in s
            assert "{lender}" not in s and "{price_arg}" not in s
