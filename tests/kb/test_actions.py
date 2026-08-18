from kb.actions import (ACTION_SPECS, EDIT_ACTIONS, MODE_TOOLS, dispatch,
                        tools_for)

from .test_store import two_docs

FLOW = {"answer", "done"}
SENSE = {"search", "read"}


def test_the_catalog_is_exactly_the_thirteen():
    assert set(ACTION_SPECS) == FLOW | SENSE | set(EDIT_ACTIONS)
    assert len(ACTION_SPECS) == 13
    assert {"add_statement", "edit_statement"} <= set(EDIT_ACTIONS)


def test_mode_subsets_per_spec():
    assert set(MODE_TOOLS["train_forward"]) == {"search", "read", "answer"}
    assert set(MODE_TOOLS["train_backprop"]) == {"search", "read", "done"} | set(EDIT_ACTIONS)
    assert set(MODE_TOOLS["test"]) == {"search", "read", "answer"}
    assert "answer" not in MODE_TOOLS["train_backprop"]
    assert not any(a in MODE_TOOLS["train_forward"] for a in EDIT_ACTIONS)


def test_authoring_actions_are_phase2_only():
    for mode in ("train_forward", "test"):
        assert "add_statement" not in MODE_TOOLS[mode]
        assert "edit_statement" not in MODE_TOOLS[mode]
        assert set(MODE_TOOLS[mode]) == {"search", "read", "answer"}  # unchanged
    assert {"add_statement", "edit_statement"} <= set(MODE_TOOLS["train_backprop"])


def test_tools_for_emits_static_schemas():
    tools = tools_for("train_backprop")
    assert [t["name"] for t in tools] == list(MODE_TOOLS["train_backprop"])
    for t in tools:
        assert t["description"] and t["input_schema"]["type"] == "object"
    # gating is by schema subset: same spec objects, no per-doc variation
    assert tools_for("train_forward")[0]["input_schema"] is ACTION_SPECS["search"]["input_schema"]


def test_no_schema_ever_takes_a_summary():
    for spec in ACTION_SPECS.values():
        assert "summary" not in spec["input_schema"]["properties"]


def test_search_result_lines_carry_the_matched_snippet():
    s = two_docs()
    out = dispatch(s, "search", {"query": "Alpha beta."})
    assert out.splitlines()[0] == '- d001: about alpha | match: "Alpha beta."'


def test_read_shows_statements_and_live_link_summaries():
    s = two_docs()
    out = dispatch(s, "read", {"doc_id": "d002"})
    assert "d002 | summary: about epsilon" in out
    assert "- s0003: Epsilon zeta." in out
    assert "- d001: about alpha" in out       # link with the LIVE summary


def test_edit_dispatch_and_result_strings():
    s = two_docs()
    assert dispatch(s, "copy_statement", {"sid": "s0001", "to_doc": "d002"}) == \
        "copied s0001 -> s0004 in d002"
    assert dispatch(s, "move_statement", {"sid": "s0002", "to_doc": "d002"}) == \
        "moved s0002 to d002"
    assert dispatch(s, "delete_statement", {"sid": "s0002"}) == "deleted s0002"
    assert dispatch(s, "link", {"a": "d001", "b": "d002"}) == "linked d001 -> d002"
    assert dispatch(s, "unlink", {"a": "d001", "b": "d002"}) == "unlinked d001 -> d002"
    assert dispatch(s, "create_doc", {"sids": ["s0001"], "link_ids": ["d001"]}) == \
        "created d003 (1 statements, 1 links)"
    assert dispatch(s, "delete_doc", {"doc_id": "d003"}) == \
        "deleted d003 (1 statement instances died)"


def test_authoring_dispatch_and_result_strings():
    s = two_docs()
    assert dispatch(s, "add_statement",
                    {"doc_id": "d002", "text": "Theta iota."}) == \
        "added s0004 to d002"
    assert dispatch(s, "edit_statement",
                    {"sid": "s0001", "text": "Alpha rewritten."}) == \
        "edited s0001"
    assert dispatch(s, "add_statement", {"doc_id": "d999", "text": "x"}) == \
        "ERROR: no such doc d999"
    assert dispatch(s, "edit_statement", {"sid": "s9999", "text": "x"}) == \
        "ERROR: no live statement s9999"
    assert dispatch(s, "add_statement", {"doc_id": "d001", "text": "  "}) == \
        "ERROR: statement text must not be empty"
    assert dispatch(s, "edit_statement", {"sid": "s0002", "text": ""}) == \
        "ERROR: statement text must not be empty"


def test_invalid_ids_become_error_results_not_exceptions():
    s = two_docs()
    assert dispatch(s, "read", {"doc_id": "d999"}) == "ERROR: no such doc d999"
    assert dispatch(s, "copy_statement", {"sid": "s9999", "to_doc": "d001"}) == \
        "ERROR: no live statement s9999"
    assert dispatch(s, "link", {"a": "d001", "b": "d001"}).startswith("ERROR:")
    assert dispatch(s, "read", {}).startswith("ERROR: bad input")
    assert dispatch(s, "flarb", {}) == "ERROR: unknown action flarb"


def test_flow_actions_are_not_dispatchable():
    s = two_docs()
    assert dispatch(s, "answer", {"text": "x"}).startswith("ERROR")
    assert dispatch(s, "done", {}).startswith("ERROR")
