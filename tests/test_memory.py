import pytest
from ca.memory import FifoMemory, GoalStack, LongTermMemory


def test_fifo_rolls_over():
    m = FifoMemory(k=2)
    m.add("a1", "r1"); m.add("a2", "r2"); m.add("a3", "r3")
    out = m.render()
    assert "a1" not in out and "a2" in out and "a3" in out


def test_fifo_pair_based_keeps_results_full():
    """K is the pair budget: 4 pairs added to k=3, oldest evicted, 3 remain IN FULL."""
    m = FifoMemory(k=3)
    # Create 4 results, each >2000 chars with a unique marker at the end
    big1 = "x" * 2000 + " MARKER_1"
    big2 = "y" * 2000 + " MARKER_2"
    big3 = "z" * 2000 + " MARKER_3"
    big4 = "w" * 2000 + " MARKER_4"

    m.add("act1", big1)
    m.add("act2", big2)
    m.add("act3", big3)
    m.add("act4", big4)  # oldest (big1) should be evicted

    out = m.render()
    # Oldest pair (act1, big1) is gone
    assert "MARKER_1" not in out
    # Remaining 3 pairs are rendered in FULL
    assert "MARKER_2" in out
    assert "MARKER_3" in out
    assert "MARKER_4" in out
    # Verify storage is indeed full (no truncation)
    assert m.items[0][1] == big2
    assert m.items[1][1] == big3
    assert m.items[2][1] == big4


def test_goal_stack_root_protected():
    g = GoalStack("maximize tokens")
    g.push("do q1")
    assert g.pop() == "do q1"
    with pytest.raises(IndexError):
        g.pop()
    assert "maximize tokens" in g.render()


def test_ltm_scoped_and_ranked():
    ltm = LongTermMemory()
    ltm.write("a", "paris is the capital of france")
    ltm.write("a", "tokyo is in japan")
    ltm.write("b", "secret of b")
    hits = ltm.search("a", "capital france", k=1)
    assert hits == ["paris is the capital of france"]
    assert ltm.search("b", "capital", k=3) == []  # no overlap for b
