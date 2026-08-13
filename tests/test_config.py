from ca.config import CONFIGS, ExperimentConfig, LevelConfig, agent_ids


def test_the_two_arms():
    assert set(CONFIGS) == {"P0", "B0"}
    assert all(CONFIGS[k].level == k for k in CONFIGS)
    assert CONFIGS["P0"].proactive is True
    assert CONFIGS["B0"].proactive is False


def test_the_centralization_spectrum_is_retired():
    for dead in ("C0", "C1", "C2", "C3", "C4", "C5", "C6", "C7"):
        assert dead not in CONFIGS


def test_proactive_is_the_only_arm_flag():
    assert [f for f in LevelConfig.__dataclass_fields__] == ["level", "proactive"]
    for dead in ("has_hub", "world_access", "star_comms", "shared_memory",
                 "n_agents"):
        assert not hasattr(CONFIGS["P0"], dead), dead


def test_the_c_level_knobs_are_gone_from_the_experiment_config():
    cfg = ExperimentConfig(level=CONFIGS["P0"], seed=0)
    for dead in ("hub_turns_per_round", "solo_turns_per_round", "claim_ttl",
                 "list_top_n"):
        assert not hasattr(cfg, dead), dead


def test_experiment_defaults():
    cfg = ExperimentConfig(level=CONFIGS["P0"], seed=0)
    assert cfg.n_agents == 8
    assert cfg.arrival_rate == 0.5
    assert cfg.fifo_k == 10 and cfg.memory_k == 5
    assert cfg.max_rounds == 60 and cfg.checkpoint_every == 20


def test_agent_ids_come_from_the_agent_count():
    assert agent_ids(8) == [f"agent_{i}" for i in range(1, 9)]
    assert agent_ids(2) == ["agent_1", "agent_2"]
    assert agent_ids(1) == ["agent_1"]
