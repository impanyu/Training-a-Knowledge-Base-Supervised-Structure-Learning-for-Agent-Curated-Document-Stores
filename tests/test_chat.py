from ca.chat import EXTERNAL, PAGE, ChatSystem


def test_pair_thread_is_shared_by_both_peers():
    cs = ChatSystem()
    cs.send("a", "b", "hi", 1)
    cs.send("b", "a", "yo", 1)
    got_a, _ = cs.read("a", "b")
    got_b, _ = cs.read("b", "a")
    assert [(m.sender, m.text) for m in got_a] == [("a", "hi"), ("b", "yo")]
    assert got_a == got_b


def test_external_threads_are_per_agent():
    cs = ChatSystem()
    cs.send(EXTERNAL, "a", "[q0001] one", 1)
    cs.send(EXTERNAL, "b", "[q0002] two", 1)
    assert [m.text for m in cs.read("a", EXTERNAL)[0]] == ["[q0001] one"]
    assert [m.text for m in cs.read("b", EXTERNAL)[0]] == ["[q0002] two"]


def test_unread_increments_on_send_and_clears_only_for_the_reader():
    cs = ChatSystem()
    cs.send("a", "b", "1", 1)
    cs.send("a", "b", "2", 1)
    cs.send("c", "b", "3", 1)
    assert cs.unread_partners("b") == [("a", 2), ("c", 1)]
    assert cs.unread_partners("a") == []       # the sender has nothing unread
    cs.read("b", "a")
    assert cs.unread_partners("b") == [("c", 1)]   # only the read thread cleared
    cs.read("b", "c")
    assert cs.unread_partners("b") == []


def test_reading_an_older_page_does_not_clear_unread():
    cs = ChatSystem()
    for i in range(7):
        cs.send("a", "b", str(i), 1)
    cs.read("b", "a", page=1)
    assert cs.unread_partners("b") == [("a", 7)]
    cs.read("b", "a", page=0)
    assert cs.unread_partners("b") == []


def test_notification_list_puts_external_first_then_peers_by_name():
    cs = ChatSystem()
    cs.send("agent_9", "me", "x", 1)
    cs.send("agent_2", "me", "x", 1)
    cs.send(EXTERNAL, "me", "[q0001] q", 1)
    assert cs.unread_partners("me") == [(EXTERNAL, 1), ("agent_2", 1), ("agent_9", 1)]


def test_pagination_pages_backwards_in_windows_of_five():
    cs = ChatSystem()
    for i in range(12):
        cs.send("a", "b", str(i), i)
    page0, older0 = cs.read("b", "a", 0)
    page1, older1 = cs.read("b", "a", 1)
    page2, older2 = cs.read("b", "a", 2)
    page3, older3 = cs.read("b", "a", 3)
    assert PAGE == 5
    assert [m.text for m in page0] == ["7", "8", "9", "10", "11"]
    assert [m.text for m in page1] == ["2", "3", "4", "5", "6"]
    assert [m.text for m in page2] == ["0", "1"]
    assert (older0, older1, older2) == (7, 2, 0)
    assert page3 == [] and older3 == 0


def test_history_is_never_truncated_or_consumed():
    cs = ChatSystem()
    for i in range(9):
        cs.send("a", "b", str(i), 1)
    for _ in range(3):
        cs.read("b", "a", 0)
    assert len(cs.threads[("a", "b")]) == 9
    assert [m.text for m in cs.read("b", "a", 1)[0]] == ["0", "1", "2", "3"]


def test_seq_is_globally_increasing():
    cs = ChatSystem()
    cs.send("a", "b", "x", 1)
    cs.send("c", "d", "y", 1)
    cs.send("a", "b", "z", 2)
    assert [m.seq for m in cs.read("b", "a", 0)[0]] == [1, 3]
    assert [m.seq for m in cs.read("d", "c", 0)[0]] == [2]


def test_agent_message_count_excludes_external_traffic():
    cs = ChatSystem()
    cs.send("a", "b", "peer", 1)
    cs.send(EXTERNAL, "a", "[q0001] arrival", 1)
    cs.send("a", EXTERNAL, "[q0001] answer", 2)
    assert cs.n_agent_messages == 1


def test_state_roundtrip_preserves_threads_unread_and_counters():
    cs = ChatSystem()
    for i in range(7):
        cs.send("a", "b", str(i), i)
    cs.send(EXTERNAL, "a", "[q0001] q", 3)
    cs.read("b", "a")
    cs.send("b", "a", "reply", 4)
    state = cs.to_state()

    import json
    state = json.loads(json.dumps(state))      # must survive JSON
    cs2 = ChatSystem()
    cs2.from_state(state)
    assert cs2.unread_partners("a") == [(EXTERNAL, 1), ("b", 1)]
    assert cs2.unread_partners("b") == []
    assert [m.text for m in cs2.read("b", "a", 1)[0]] == \
        [m.text for m in cs.read("b", "a", 1)[0]]
    assert cs2.n_agent_messages == cs.n_agent_messages == 8
    cs2.send("a", "b", "after", 5)             # seq continues, no collision
    assert cs2.read("b", "a", 0)[0][-1].seq == 10
