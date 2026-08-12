from ca.config import CONFIGS, ExperimentConfig, LevelConfig, agent_ids

# The mechanisms a configuration may centralize. C0 is the all-decentralized
# baseline; every other multi-agent config must differ from it in EXACTLY one
# of these (single-factor design).
MECHANISMS = ("world_access", "shared_memory", "star_comms")

# which single mechanism each config is *supposed* to flip
FLIPPED = {"C1": "world_access", "C2": "shared_memory", "C5": "star_comms"}


def _mechanisms(cfg):
    return {m: getattr(cfg, m) for m in MECHANISMS}


def test_the_surviving_config_set():
    assert set(CONFIGS) == {"C0", "C1", "C2", "C5", "C7"}
    assert all(CONFIGS[k].level == k for k in CONFIGS)


def test_the_economy_configs_are_deleted():
    """C3 (pricing), C4 (credit) and C6 (collective goal) died with the token
    economy; the survivors keep their v3-v5 identities."""
    for dead in ("C3", "C4", "C6"):
        assert dead not in CONFIGS


def test_c0_is_the_fully_decentralized_baseline():
    c0 = CONFIGS["C0"]
    assert c0.n_agents == 8 and not c0.has_hub
    assert _mechanisms(c0) == {"world_access": "all", "shared_memory": False,
                               "star_comms": False}


def test_each_config_flips_exactly_one_mechanism_vs_c0():
    base = _mechanisms(CONFIGS["C0"])
    for name, mech in FLIPPED.items():
        cfg = CONFIGS[name]
        diff = {m for m, v in _mechanisms(cfg).items() if v != base[m]}
        assert diff == {mech}, f"{name} should flip only {mech}, got {diff}"
        assert cfg.n_agents == 8, f"{name} must hold the agent count fixed at 8"


def test_flipped_values_are_the_centralizing_ones():
    assert CONFIGS["C1"].world_access == "hub"
    assert CONFIGS["C2"].shared_memory is True
    assert CONFIGS["C5"].star_comms is True


def test_hub_exists_only_where_a_hub_holds_the_power():
    # C2 centralizes infrastructure, not an agent, so it stays leaderless;
    # C1/C5 need a hub to hold the flipped power.
    assert [c for c in CONFIGS if CONFIGS[c].has_hub] == ["C1", "C5"]


def test_c7_is_the_solo_baseline():
    c7 = CONFIGS["C7"]
    assert c7.n_agents == 1 and not c7.has_hub
    assert _mechanisms(c7) == _mechanisms(CONFIGS["C0"])


def test_the_economy_fields_are_gone():
    for dead in ("central_pricing", "central_credit", "collective_goal"):
        assert not hasattr(CONFIGS["C0"], dead), dead
    cfg = ExperimentConfig(level=CONFIGS["C0"], seed=0)
    for dead in ("seed_capital_total", "loan_rate"):
        assert not hasattr(cfg, dead), dead


def test_level_config_fields():
    assert [f for f in LevelConfig.__dataclass_fields__] == [
        "level", "n_agents", "has_hub", "world_access", "star_comms",
        "shared_memory"]


def test_retrieve_access_is_gone():
    """Info centralization was deleted in v3: retrieval is infrastructure."""
    assert not hasattr(CONFIGS["C0"], "retrieve_access")


def test_agent_ids():
    assert agent_ids(CONFIGS["C0"]) == [f"agent_{i}" for i in range(1, 9)]
    ids = agent_ids(CONFIGS["C1"])
    assert ids[0] == "hub" and len(ids) == 8
    assert agent_ids(CONFIGS["C7"]) == ["agent_1"]
