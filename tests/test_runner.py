import json

from fixtures import demo_library

from ca.runner import load_library, load_posted, load_questions
from ca.tasktree import TaskLibrary


def test_load_questions(tmp_path):
    p = tmp_path / "pool.jsonl"
    rows = [
        {"qid": "q0001", "text": "t1", "answers": ["a"], "difficulty": "2hop", "price": 1000},
        {"qid": "q0002", "text": "t2", "answers": ["b", "c"], "difficulty": "4hop", "price": 3000},
    ]
    p.write_text("\n".join(json.dumps(r) for r in rows))
    qs = load_questions(str(p))
    assert len(qs) == 2 and qs[1].answers == ["b", "c"] and qs[1].price == 3000


def test_load_library_wraps_a_bare_pool_into_single_leaf_tasks(tmp_path):
    """Interim shim (until T21): a pool.jsonl still yields a runnable library."""
    p = tmp_path / "pool.jsonl"
    p.write_text("\n".join(json.dumps(
        {"qid": f"q{i:04d}", "text": f"question {i}", "answers": ["a"],
         "difficulty": "2hop", "price": 100 * i}) for i in (1, 2)))
    lib = load_library(str(p))
    assert load_posted(lib) == ["t0001", "t0002"]
    assert lib.leaves("t0002") == ["q0002"] and lib.price("t0002") == 200
    assert lib.resolve("question 1").nid == "t0001"


def test_load_library_and_posted_from_one_file(tmp_path):
    p = tmp_path / "library.json"
    demo_library().to_json(str(p), posted=["t0001", "t0004"])
    lib = load_library(str(p))
    assert load_posted(lib) == ["t0001", "t0004"]
    assert lib.price("t0001") == 600


def test_posted_json_overrides_the_library_field(tmp_path):
    lp, pp = tmp_path / "library.json", tmp_path / "posted.json"
    demo_library().to_json(str(lp), posted=["t0001"])
    pp.write_text(json.dumps(["t0004"]))
    lib = TaskLibrary.from_json(str(lp))
    assert load_posted(lib, str(pp)) == ["t0004"]


def test_posted_defaults_to_the_roots_of_the_library(tmp_path):
    p = tmp_path / "library.json"
    demo_library().to_json(str(p))            # no posted list at all
    lib = TaskLibrary.from_json(str(p))
    assert load_posted(lib) == ["t0001", "t0004", "t0005"]   # t0002 has a parent
