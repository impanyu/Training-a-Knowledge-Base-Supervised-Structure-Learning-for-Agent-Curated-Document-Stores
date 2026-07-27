from ca.chat import ChatSystem


def test_send_unread_mark_read():
    cs = ChatSystem()
    cs.send("a", "b", "hi", 1)
    cs.send("c", "b", "yo", 1)
    cs.send("a", "c", "not for b", 1)
    assert [m.text for m in cs.unread("b")] == ["hi", "yo"]
    cs.mark_read("b")
    assert cs.unread("b") == []
    cs.send("a", "b", "again", 2)
    assert [m.text for m in cs.unread("b")] == ["again"]


def test_history_pairwise():
    cs = ChatSystem()
    cs.send("a", "b", "1", 1)
    cs.send("b", "a", "2", 1)
    cs.send("a", "c", "x", 1)
    assert [m.text for m in cs.history("a", "b")] == ["1", "2"]
