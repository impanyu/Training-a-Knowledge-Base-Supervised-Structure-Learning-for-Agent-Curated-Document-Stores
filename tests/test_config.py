from ca.config import CONFIGS, ExperimentConfig, agent_ids
from ca.infra import Infra
from fixtures import demo_library, demo_posted

# The mechanisms a configuration may centralize. C0 is the all-decentralized
# baseline; every other multi-agent config must differ from it in EXACTLY one
# of these (v3 single-factor design, spec section 9).
MECHANISMS = ("world_access", "shared_solution_memory", "central_pricing",
              "central_credit", "star_comms", "collective_goal")

# which single mechanism each config is *supposed* to flip
FLIPPED = {"C1": "world_access", "C2": "shared_solution_memory",
           "C3": "central_pricing", "C4": "central_credit",
           "C5": "star_comms", "C6": "collective_goal"}


def _mechanisms(cfg):
    return {m: getattr(cfg, m) for m in MECHANISMS}


def test_config_set_is_c0_through_c7():
    assert set(CONFIGS) == {f"C{i}" for i in range(8)}
    assert all(CONFIGS[k].level == k for k in CONFIGS)


def test_c0_is_the_fully_decentralized_baseline():
    c0 = CONFIGS["C0"]
    assert c0.n_agents == 8 and not c0.has_interface
    assert _mechanisms(c0) == {"world_access": "all", "shared_solution_memory": False,
                               "central_pricing": False, "central_credit": False,
                               "star_comms": False, "collective_goal": False}


def test_each_config_flips_exactly_one_mechanism_vs_c0():
    base = _mechanisms(CONFIGS["C0"])
    for name, mech in FLIPPED.items():
        cfg = CONFIGS[name]
        diff = {m for m, v in _mechanisms(cfg).items() if v != base[m]}
        assert diff == {mech}, f"{name} should flip only {mech}, got {diff}"
        assert cfg.n_agents == 8, f"{name} must hold the agent count fixed at 8"


def test_flipped_values_are_the_centralizing_ones():
    assert CONFIGS["C1"].world_access == "interface"
    assert CONFIGS["C2"].shared_solution_memory is True
    assert CONFIGS["C3"].central_pricing is True
    assert CONFIGS["C4"].central_credit is True
    assert CONFIGS["C5"].star_comms is True
    assert CONFIGS["C6"].collective_goal is True


def test_interface_exists_only_where_a_hub_holds_the_power():
    # C2/C6 centralize infrastructure or the goal function, not an agent, so
    # they stay leaderless; C1/C3/C4/C5 need a hub to hold the flipped power.
    assert [c for c in CONFIGS if CONFIGS[c].has_interface] == ["C1", "C3", "C4", "C5"]


def test_c7_is_the_solo_baseline():
    c7 = CONFIGS["C7"]
    assert c7.n_agents == 1 and not c7.has_interface
    assert _mechanisms(c7) == _mechanisms(CONFIGS["C0"])


def test_retrieve_access_is_gone():
    """Info centralization was deleted in v3: retrieval is infrastructure."""
    assert not hasattr(CONFIGS["C0"], "retrieve_access")


def test_agent_ids():
    assert agent_ids(CONFIGS["C0"]) == [f"agent_{i}" for i in range(1, 9)]
    ids = agent_ids(CONFIGS["C1"])
    assert ids[0] == "interface" and len(ids) == 8
    assert agent_ids(CONFIGS["C7"]) == ["agent_1"]


def test_infra_splits_seed_capital():
    cfg = ExperimentConfig(level=CONFIGS["C0"], seed=0, seed_capital_total=801)
    infra = Infra(cfg, demo_library(), demo_posted(), retriever=None)
    balances = [infra.ledger.balance(a) for a in infra.agent_ids]
    assert sum(balances) == 801 and max(balances) - min(balances) <= 1
    assert infra.ledger.conservation_ok()
