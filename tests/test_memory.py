import pytest
from ca.memory import FifoMemory, GoalStack, LongTermMemory


def test_fifo_rolls_over():
    m = FifoMemory(k=2)
    m.add("a1", "r1"); m.add("a2", "r2"); m.add("a3", "r3")
    out = m.render()
    assert "a1" not in out and "a2" in out and "a3" in out


def test_fifo_keeps_recent_results_intact_and_truncates_aged_ones():
    m = FifoMemory(k=5, cap_recent=4000, cap_old=300, recent_n=2)
    big = "x" * 2000
    m.add("retrieve", big)
    assert big in m.render()          # newest entry: full result survives
    m.add("a2", "r2")
    assert big in m.render()          # still inside the recent_n window
    m.add("a3", "r3")
    out = m.render()
    assert big not in out             # aged out -> truncated
    assert "x" * 300 in out           # ... to cap_old
    assert m.items[0][1] == big       # but stored FULL, never lossy


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
