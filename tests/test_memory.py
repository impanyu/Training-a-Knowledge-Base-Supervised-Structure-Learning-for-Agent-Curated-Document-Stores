import pytest
from ca.memory import FifoMemory, GoalStack, LongTermMemory


def test_fifo_rolls_over():
    m = FifoMemory(k=2)
    m.add("a1", "r1"); m.add("a2", "r2"); m.add("a3", "r3")
    out = m.render()
    assert "a1" not in out and "a2" in out and "a3" in out


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
