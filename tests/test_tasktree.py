import json

import pytest
from fixtures import demo_library

from ca.taskboard import Question
from ca.tasktree import AmbiguousError, TaskLibrary, TaskNode, UnknownNodeError


def test_resolve_by_short_id():
    assert demo_library().resolve("t0002").nid == "t0002"


def test_resolve_by_sentence_ignores_case_punctuation_and_spacing():
    lib = demo_library()
    assert lib.resolve("  Name the Capital, and   the River!  ").nid == "t0002"


def test_resolve_fuzzy_best_match_above_threshold():
    # a typo'd sentence still lands on the intended node when it wins clearly
    assert demo_library().resolve("answer the french geograhy questions").nid == "t0001"


def test_resolve_ambiguous_lists_both_candidates():
    lib = demo_library()
    with pytest.raises(AmbiguousError) as e:
        lib.resolve("answer the french geograpology questions")
    msg = str(e.value)
    assert "t0001" in msg and "t0005" in msg


def test_resolve_unknown_lists_three_nearest():
    lib = demo_library()
    with pytest.raises(UnknownNodeError) as e:
        lib.resolve("who painted the ceiling of the sistine chapel")
    msg = str(e.value)
    assert msg.count("t0") == 3          # exactly three candidates offered


def test_resolve_unknown_qid_is_not_a_node():
    lib = demo_library()
    with pytest.raises(UnknownNodeError):
        lib.resolve("q0001")             # leaves are not nodes; decompose handles them


def test_leaves_are_recursive_and_ordered():
    lib = demo_library()
    assert lib.leaves("t0001") == ["q0001", "q0002", "q0003"]
    assert lib.leaves("t0002") == ["q0001", "q0002"]
    assert lib.leaves("t0004") == ["q0003", "q0004"]


def test_price_is_the_sum_of_leaf_prices():
    lib = demo_library()
    assert lib.price("t0002") == 300
    assert lib.price("t0001") == 600     # 100 + 200 + 300, recursive
    assert lib.price("t0004") == 700


def test_depth_counts_the_leaf_level():
    lib = demo_library()
    assert lib.depth("t0002") == 2       # node + leaves
    assert lib.depth("t0001") == 3       # root + subtask + leaves


def test_children_view_separates_subtasks_from_leaves():
    rows = demo_library().children_view("t0001")
    assert rows[0] == {"kind": "subtask", "nid": "t0002",
                       "sentence": "name the capital and the river",
                       "leaf_count": 2, "price": 300}
    assert rows[1] == {"kind": "leaf", "qid": "q0003", "text": "2+2?", "price": 300}


def test_parents_are_linked_from_children():
    lib = demo_library()
    assert lib.nodes["t0002"].parent == "t0001"
    assert lib.nodes["t0001"].parent is None


def test_leaves_and_price_reject_unknown_nodes():
    lib = demo_library()
    with pytest.raises(UnknownNodeError):
        lib.leaves("t9999")
    with pytest.raises(UnknownNodeError):
        lib.price("t9999")


def test_json_roundtrip(tmp_path):
    lib = demo_library()
    p = tmp_path / "library.json"
    lib.to_json(str(p))
    back = TaskLibrary.from_json(str(p))
    assert back.leaves("t0001") == lib.leaves("t0001")
    assert back.price("t0001") == 600
    assert back.questions["q0001"].answers == ["Paris"]
    assert back.questions["q0002"].difficulty == "easy"
    assert back.resolve("name the capital and the river").nid == "t0002"
    assert back.nodes["t0002"].parent == "t0001"


def test_from_json_reads_optional_posted_list(tmp_path):
    p = tmp_path / "library.json"
    demo_library().to_json(str(p), posted=["t0004"])
    assert TaskLibrary.from_json(str(p)).posted == ["t0004"]


def test_shared_leaf_belongs_to_two_nodes():
    lib = demo_library()
    assert "q0003" in lib.leaves("t0001") and "q0003" in lib.leaves("t0004")


def test_cycles_are_rejected_rather_than_hanging():
    # a cycle is now caught at construction time, not only when walked later
    nodes = [TaskNode("t0001", "a", ["t0002"]), TaskNode("t0002", "b", ["t0001"])]
    with pytest.raises(ValueError):
        TaskLibrary(nodes, [])


def test_construction_rejects_dangling_leaf_ref():
    nodes = [TaskNode("t0001", "a", ["q9999"])]
    with pytest.raises(ValueError):
        TaskLibrary(nodes, [])


def test_from_json_dangling_leaf_ref_raises_at_load(tmp_path):
    p = tmp_path / "library.json"
    payload = {
        "nodes": [{"nid": "t0001", "sentence": "a", "children": ["q9999"]}],
        "questions": [],
    }
    p.write_text(json.dumps(payload))
    with pytest.raises(ValueError):
        TaskLibrary.from_json(str(p))


def test_from_json_cyclic_children_raises_at_load(tmp_path):
    p = tmp_path / "library.json"
    payload = {
        "nodes": [
            {"nid": "t0001", "sentence": "a", "children": ["t0002"]},
            {"nid": "t0002", "sentence": "b", "children": ["t0001"]},
        ],
        "questions": [],
    }
    p.write_text(json.dumps(payload))
    with pytest.raises(ValueError):
        TaskLibrary.from_json(str(p))


def test_construction_rejects_overlapping_node_and_question_ids():
    nodes = [TaskNode("q0001", "a", [])]
    qs = [Question("q0001", "x?", ["y"], "easy", 10)]
    with pytest.raises(ValueError):
        TaskLibrary(nodes, qs)


def test_depth_on_valid_deep_chain():
    nodes = [TaskNode("t0001", "a", ["t0002"]), TaskNode("t0002", "b", ["t0003"]),
             TaskNode("t0003", "c", ["q0001"])]
    qs = [Question("q0001", "x?", ["y"], "easy", 10)]
    lib = TaskLibrary(nodes, qs)
    assert lib.depth("t0001") == 4


def test_base_subtask_returns_wrapping_subtask_node():
    lib = demo_library()
    assert lib.base_subtask("t0001", "q0001") == "t0002"
    assert lib.base_subtask("t0001", "q0002") == "t0002"


def test_base_subtask_returns_leaf_itself_when_directly_attached():
    lib = demo_library()
    # q0003 hangs directly off t0001, with no wrapping subtask node
    assert lib.base_subtask("t0001", "q0003") == "q0003"


def test_base_subtask_is_relative_to_the_task_it_was_delivered_under():
    lib = demo_library()
    # q0003 is shared: under t0001 it's a bare leaf, under t0004 also a bare leaf
    assert lib.base_subtask("t0004", "q0003") == "q0003"
    assert lib.base_subtask("t0004", "q0004") == "q0004"


def test_base_subtask_rejects_leaf_not_under_task():
    lib = demo_library()
    with pytest.raises(UnknownNodeError):
        lib.base_subtask("t0002", "q0005")


def test_resolve_exact_matches_id_or_exact_sentence_only():
    lib = demo_library()
    assert lib.resolve_exact("t0002").nid == "t0002"
    assert lib.resolve_exact("  Name the Capital, and   the River!  ").nid == "t0002"
    # a near-miss (fuzzy) sentence that resolve() would happily match is
    # NOT resolved exactly
    assert lib.resolve("answer the french geograhy questions").nid == "t0001"
    assert lib.resolve_exact("answer the french geograhy questions") is None
    assert lib.resolve_exact("no such task at all") is None
