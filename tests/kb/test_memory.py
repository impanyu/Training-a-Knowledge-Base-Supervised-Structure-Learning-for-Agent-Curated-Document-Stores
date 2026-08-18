from kb.memory import IterationMemory


def test_task_block_is_pinned_first():
    m = IterationMemory()
    m.reset("QUESTION q0001\nWhat is x?")
    out = m.render("Remaining actions this phase: 15.")
    assert out.startswith("QUESTION q0001")
    assert "Remaining actions this phase: 15." in out
    assert "(no actions yet)" in out


def test_pairs_render_full_and_in_order():
    m = IterationMemory()
    m.reset("t")
    m.add("search(query='x')", "- d001: a doc")
    m.add("read(doc_id='d001')", "d001 | summary: a doc")
    out = m.render()
    assert out.index("search") < out.index("read")
    assert "- search(query='x') -> - d001: a doc" in out


def test_reset_clears_fifo_and_replaces_task():
    m = IterationMemory()
    m.reset("iter one")
    m.add("a", "r")
    m.reset("iter two")
    assert m.task == "iter two"
    assert not m.items
    assert "(no actions yet)" in m.render()


def test_set_task_keeps_fifo_across_phase_transition():
    m = IterationMemory()
    m.reset("phase 1 task")
    m.add("answer(text='x')", "gold: y | F1 0.00")
    m.set_task("phase 2 task")
    out = m.render()
    assert out.startswith("phase 2 task")
    assert "F1 0.00" in out          # phase 1 trajectory read straight from FIFO


def test_k_eviction_oldest_first():
    m = IterationMemory(k=3)
    m.reset("t")
    for i in range(5):
        m.add(f"a{i}", f"r{i}")
    out = m.render()
    assert "a0" not in out and "a1" not in out
    assert "a2" in out and "a4" in out
    assert len(m.items) == 3


def test_default_k_covers_both_phase_budgets():
    assert IterationMemory().k == 30    # K >= N1+N2: trajectory never evicted
