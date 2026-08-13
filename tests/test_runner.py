"""The runner's CLI surface. It takes one --bank; corpus.jsonl,
corpus_emb.npy and the qclusters cache live in its directory."""
import pytest

from ca.runner import build_parser


BASE = ["--level", "P0", "--out", "runs/x", "--bank", "data/v5/bank.json"]


def test_only_the_two_arms_are_selectable():
    ap = build_parser()
    for dead in ("C0", "C1", "C2", "C5", "C7"):
        with pytest.raises(SystemExit):
            ap.parse_args(["--level", dead, "--out", "runs/x", "--bank", "b.json"])
    for alive in ("P0", "B0"):
        ap.parse_args(["--level", alive, "--out", "runs/x", "--bank", "b.json"])


def test_bank_is_required_and_the_dead_flags_are_gone():
    ap = build_parser()
    args = ap.parse_args(BASE)
    assert args.bank == "data/v5/bank.json"
    for dead in ("--solo-turns", "--turns", "--hub-turns", "--index"):
        with pytest.raises(SystemExit):
            ap.parse_args(BASE + [dead, "1"])
    with pytest.raises(SystemExit):
        ap.parse_args(["--level", "P0", "--out", "runs/x"])   # --bank required


def test_the_v7_knobs():
    args = build_parser().parse_args(
        BASE + ["--agents", "4", "--arrival-rate", "1.5", "--seed", "3",
                "--max-rounds", "12", "--checkpoint-every", "4",
                "--resume", "runs/x/checkpoint_0004.json", "--model", "m"])
    assert args.agents == 4 and args.arrival_rate == 1.5
    assert (args.seed, args.max_rounds) == (3, 12)
    assert args.checkpoint_every == 4
    assert args.resume.endswith("checkpoint_0004.json") and args.model == "m"


def test_defaults():
    args = build_parser().parse_args(BASE)
    assert args.agents == 8 and args.arrival_rate == 0.5
    assert args.seed == 0 and args.max_rounds == 60
    assert args.checkpoint_every == 20 and args.resume is None


def test_question_clusters_builds_once_then_loads(tmp_path):
    import unittest.mock as mock

    import ca.runner as runner_mod
    from fixtures import HashEmbedding, demo_bank

    bank = demo_bank()
    with mock.patch.object(runner_mod, "_onnx_ef", HashEmbedding):
        first = runner_mod.question_clusters(bank, tmp_path, 2)
    assert (tmp_path / "qclusters_2.json").exists()
    assert set(first["assignment"].values()) == {0, 1}

    def explode():
        raise AssertionError("a present cache must be loaded, not re-embedded")

    with mock.patch.object(runner_mod, "_onnx_ef", explode):
        second = runner_mod.question_clusters(bank, tmp_path, 2)
    assert second == first
