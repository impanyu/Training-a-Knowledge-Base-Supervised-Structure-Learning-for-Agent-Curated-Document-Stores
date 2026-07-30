from ca.config import LEVELS, ExperimentConfig, agent_ids
from ca.infra import Infra
from ca.taskboard import Question


def test_level_matrix_matches_spec():
    assert LEVELS["L0"].world_access == "all" and not LEVELS["L0"].has_interface
    assert not LEVELS["L0"].central_credit
    assert LEVELS["L1"].world_access == "interface" and LEVELS["L1"].retrieve_access == "all"
    assert not LEVELS["L1"].central_credit
    assert LEVELS["L2"].retrieve_access == "interface" and not LEVELS["L2"].central_pricing
    assert not LEVELS["L2"].central_credit
    assert LEVELS["L3"].central_pricing and not LEVELS["L3"].star_comms
    assert not LEVELS["L3"].central_credit
    assert LEVELS["L4"].central_credit and LEVELS["L4"].central_pricing
    assert not LEVELS["L4"].star_comms
    assert LEVELS["L5"].star_comms and LEVELS["L5"].central_pricing and LEVELS["L5"].central_credit
    assert LEVELS["L6"].n_agents == 1
    assert not LEVELS["L6"].has_interface and not LEVELS["L6"].central_credit
    assert set(LEVELS) == {"L0", "L1", "L2", "L3", "L4", "L5", "L6"}


def test_agent_ids():
    assert agent_ids(LEVELS["L0"]) == [f"agent_{i}" for i in range(1, 9)]
    ids = agent_ids(LEVELS["L1"])
    assert ids[0] == "interface" and len(ids) == 8
    assert agent_ids(LEVELS["L6"]) == ["agent_1"]


def test_infra_splits_seed_capital():
    cfg = ExperimentConfig(level=LEVELS["L0"], seed=0, seed_capital_total=801)
    infra = Infra(cfg, [Question("q0001", "?", ["x"], "easy", 10)], retriever=None)
    balances = [infra.ledger.balance(a) for a in infra.agent_ids]
    assert sum(balances) == 801 and max(balances) - min(balances) <= 1
    assert infra.ledger.conservation_ok()
