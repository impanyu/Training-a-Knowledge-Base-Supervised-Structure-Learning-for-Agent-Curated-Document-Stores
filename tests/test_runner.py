"""The runner's CLI surface. v4 takes one --bank (the flat question bank);
--library / --posted are gone with the task trees."""
import json

import pytest
from fixtures import demo_questions

from ca.bank import QuestionBank
from ca.runner import build_parser


BASE = ["--level", "C0", "--index", "idx", "--out", "runs/x"]


def test_bank_is_required_and_the_tree_flags_are_gone():
    ap = build_parser()
    args = ap.parse_args(BASE + ["--bank", "data/v4/bank.json"])
    assert args.bank == "data/v4/bank.json"
    for dead in ("--library", "--posted"):
        with pytest.raises(SystemExit):
            ap.parse_args(BASE + ["--bank", "b.json", dead, "x"])
    with pytest.raises(SystemExit):
        ap.parse_args(BASE)                      # --bank is required


def test_run_knobs_survive_the_port():
    args = build_parser().parse_args(
        BASE + ["--bank", "b.json", "--seed", "3", "--capital", "5000",
                "--max-rounds", "12", "--checkpoint-every", "4",
                "--resume", "runs/x/checkpoint_0004.json", "--model", "m",
                "--solo-turns", "8"])
    assert (args.seed, args.capital, args.max_rounds) == (3, 5000, 12)
    assert args.checkpoint_every == 4 and args.solo_turns == 8
    assert args.resume.endswith("checkpoint_0004.json") and args.model == "m"
    assert args.index == "idx" and args.out == "runs/x"


def test_defaults():
    args = build_parser().parse_args(BASE + ["--bank", "b.json"])
    assert args.seed == 0 and args.capital == 400_000 and args.max_rounds == 60
    assert args.checkpoint_every == 20 and args.solo_turns == 1
    assert args.resume is None and args.turns is None


def test_bank_json_loads_through_the_same_path_the_runner_uses(tmp_path):
    p = tmp_path / "bank.json"
    p.write_text(json.dumps({"questions": [
        {"qid": q.qid, "text": q.text, "answers": q.answers,
         "difficulty": q.difficulty, "price": q.price, "quota": q.quota,
         "topic": q.topic, "source": "hotpot"}          # unknown fields ignored
        for q in demo_questions()]}))
    bank = QuestionBank.from_json(str(p))
    assert len(bank.questions) == 5 and bank.total_units() == 7
    assert bank.get("q0004").topic == "k07"
