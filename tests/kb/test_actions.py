from kb.actions import (ACTION_SPECS, EDIT_ACTIONS, MODE_TOOLS, dispatch,
                        tools_for)

from .test_store import three_nodes

FLOW = {"answer", "done"}
SENSE = {"search", "read"}


def test_the_catalog_is_exactly_the_ten():
    assert set(ACTION_SPECS) == FLOW | SENSE | set(EDIT_ACTIONS)
    assert len(ACTION_SPECS) == 10
    assert EDIT_ACTIONS == ("add", "edit", "delete", "link", "link_many",
                            "unlink")


def test_mode_subsets_per_spec():
    assert set(MODE_TOOLS["train_forward"]) == {"search", "read", "answer"}
    assert set(MODE_TOOLS["train_backward"]) == {"search", "read",
                                                 "done"} | set(EDIT_ACTIONS)
    assert set(MODE_TOOLS["test"]) == {"search", "read", "answer"}
    assert "answer" not in MODE_TOOLS["train_backward"]
    for mode in ("train_forward", "test"):     # edits are train Phase 2 ONLY
        assert not any(a in MODE_TOOLS[mode] for a in EDIT_ACTIONS)


def test_tools_for_emits_static_schemas():
    tools = tools_for("train_backward")
    assert [t["name"] for t in tools] == list(MODE_TOOLS["train_backward"])
    for t in tools:
        assert t["description"] and t["input_schema"]["type"] == "object"
    # gating is by schema subset: same spec objects, no per-node variation
    assert tools_for("train_forward")[0]["input_schema"] is ACTION_SPECS["search"]["input_schema"]


def test_search_result_lines_are_id_and_full_text():
    s = three_nodes()
    out = dispatch(s, "search", {"query": "Alpha beta."})
    assert out.splitlines()[0] == "- s0001: Alpha beta."


def test_read_shows_text_and_link_target_texts():
    s = three_nodes()
    out = dispatch(s, "read", {"id": "s0003"})
    assert out.splitlines()[0] == "s0003: Epsilon zeta."
    assert "- s0001: Alpha beta." in out       # link with the target's text
    assert dispatch(s, "read", {"id": "s0001"}).splitlines()[-1] == "(none)"


def test_edit_dispatch_and_result_strings():
    s = three_nodes()
    assert dispatch(s, "add", {"text": "Theta iota."}) == "added s0004"
    assert dispatch(s, "edit", {"id": "s0001", "text": "Alpha rewritten."}) == \
        "edited s0001"
    assert dispatch(s, "link", {"a": "s0001", "b": "s0002"}) == \
        "linked s0001 -> s0002"
    assert dispatch(s, "unlink", {"a": "s0001", "b": "s0002"}) == \
        "unlinked s0001 -> s0002"
    assert dispatch(s, "delete", {"id": "s0004"}) == "deleted s0004"


def test_invalid_ids_become_error_results_not_exceptions():
    s = three_nodes()
    assert dispatch(s, "read", {"id": "s9999"}) == "ERROR: no such note s9999"
    assert dispatch(s, "edit", {"id": "s9999", "text": "x"}) == \
        "ERROR: no such note s9999"
    assert dispatch(s, "delete", {"id": "s9999"}).startswith("ERROR:")
    assert dispatch(s, "add", {"text": "  "}) == \
        "ERROR: note text must not be empty"
    assert dispatch(s, "link", {"a": "s0001", "b": "s0001"}).startswith("ERROR:")
    assert dispatch(s, "read", {}).startswith("ERROR: bad input")
    assert dispatch(s, "flarb", {}) == "ERROR: unknown action flarb"


def test_flow_actions_are_not_dispatchable():
    s = three_nodes()
    assert dispatch(s, "answer", {"text": "x"}).startswith("ERROR")
    assert dispatch(s, "done", {}).startswith("ERROR")
