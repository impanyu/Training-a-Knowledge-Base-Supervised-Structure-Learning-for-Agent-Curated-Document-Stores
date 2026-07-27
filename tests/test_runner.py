import json
from ca.runner import load_questions


def test_load_questions(tmp_path):
    p = tmp_path / "pool.jsonl"
    rows = [
        {"qid": "q0001", "text": "t1", "answers": ["a"], "difficulty": "2hop", "price": 1000},
        {"qid": "q0002", "text": "t2", "answers": ["b", "c"], "difficulty": "4hop", "price": 3000},
    ]
    p.write_text("\n".join(json.dumps(r) for r in rows))
    qs = load_questions(str(p))
    assert len(qs) == 2 and qs[1].answers == ["b", "c"] and qs[1].price == 3000
