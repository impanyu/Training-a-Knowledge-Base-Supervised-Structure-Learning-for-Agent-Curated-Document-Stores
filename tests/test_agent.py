import ca.agent as agent_mod
from fixtures import arrive, demo_infra

from ca.agent import Agent, Decision, LLMPolicy, ScriptedPolicy
from ca.context import ROOT_GOAL


def make(level="P0", **kw):
    infra = demo_infra(level, **kw)
    infra.round = 1
    return infra.cfg, infra


def test_turn_executes_and_logs():
    cfg, infra = make()
    arrive(infra, "q0005", 1)
    ag = Agent("agent_1", cfg, infra,
               ScriptedPolicy([("deliver_work", {"target_id": "q0005", "content": "4"})]))
    ev = ag.take_turn()
    assert ev["action"] == "deliver_work" and ev["category"] == "solving"
    assert ev["result"] == "delivered q0005: F1 1.00"
    assert len(ag.fifo.items) == 1


def test_tokens_are_measured_not_charged():
    cfg, infra = make()
    ag = Agent("agent_1", cfg, infra,
               ScriptedPolicy([("memory_search", {"query": "capital of France"})],
                              in_tokens=100, out_tokens=20))
    ev = ag.take_turn()
    assert ev["category"] == "solving"
    assert ev["tokens_in"] == 100 and ev["tokens_out"] == 20
    assert "balance_after" not in ev      # nothing is debited any more


def test_record_qa_turn_is_solving_work():
    cfg, infra = make()
    ag = Agent("agent_1", cfg, infra,
               ScriptedPolicy([("record_qa", {"question": "q?", "answer": "a"})]))
    ev = ag.take_turn()
    assert ev["category"] == "solving"
    assert ev["result"] == "recorded to the shared knowledge base"


def test_b0_agents_are_not_offered_record_qa():
    cfg, infra = make("B0")
    ag = Agent("agent_1", cfg, infra, ScriptedPolicy([]))
    assert "record_qa" not in {t["name"] for t in ag._tools}
    assert "record_qa" not in ag._system
    cfg, infra = make("P0")
    ag = Agent("agent_1", cfg, infra, ScriptedPolicy([]))
    assert "record_qa" in {t["name"] for t in ag._tools}


def test_noop_turn_is_an_error_result():
    class _NoopPolicy:
        def decide(self, system, context, tools):
            return Decision("__noop__", {}, 7, 3)

    cfg, infra = make()
    ag = Agent("agent_1", cfg, infra, _NoopPolicy())
    ev = ag.take_turn()
    assert ev["result"].startswith("ERROR")
    assert ev["category"] == "admin"
    assert ev["tokens_in"] == 7 and ev["tokens_out"] == 3


def test_goal_actions_update_local_stack():
    cfg, infra = make()
    ag = Agent("agent_1", cfg, infra, ScriptedPolicy([
        ("push_goal", {"note": "solve q0005"}), ("pop_goal", {})]))
    ag.take_turn()
    assert "solve q0005" in ag.goals.render()
    ag.take_turn()
    assert "solve q0005" not in ag.goals.render()


def test_every_agent_shares_the_one_root_goal():
    for arm in ("P0", "B0"):
        cfg, infra = make(arm)
        root = Agent("agent_2", cfg, infra, ScriptedPolicy([])).goals.render()
        assert f"[0] {ROOT_GOAL} (root, permanent)" in root


def test_the_root_goal_cannot_be_popped():
    cfg, infra = make()
    ag = Agent("agent_1", cfg, infra, ScriptedPolicy([("pop_goal", {})]))
    ev = ag.take_turn()
    assert ev["result"].startswith("ERROR")
    assert ROOT_GOAL in ag.goals.render()


def test_a_turn_does_not_clear_unread_notifications():
    """Only read_chat(page=0) clears unread -- taking a turn must not."""
    cfg, infra = make()
    arrive(infra, "q0005", 1)
    ag = Agent("agent_1", cfg, infra, ScriptedPolicy([("list_agents", {})]))
    ag.take_turn()
    assert infra.chat.unread_partners("agent_1") == [("external", 1)]


def test_the_domain_exemplars_reach_the_system_prompt():
    cfg, infra = make()
    ag2 = Agent("agent_2", cfg, infra, ScriptedPolicy([]))
    assert "capital of France?" in ag2._system            # France expert
    ag1 = Agent("agent_1", cfg, infra, ScriptedPolicy([]))
    assert "sum of 2 and 2?" in ag1._system               # arithmetic expert


def test_llm_policy_retries_and_survives_sdk_errors(monkeypatch):
    captured = {}

    class FakeMessages:
        def create(self, **kw):
            raise RuntimeError("503 overloaded_error")

    class FakeClient:
        def __init__(self, **kw):
            captured.update(kw)
            self.messages = FakeMessages()

    monkeypatch.setattr(agent_mod.anthropic, "Anthropic", FakeClient)
    policy = LLMPolicy("claude-haiku-4-5")
    assert captured["max_retries"] == 5          # SDK-level retries for transient errors
    d = policy.decide("sys", "ctx", [])          # a hard failure must not kill the run
    assert d.name == "__noop__"
    assert d.in_tokens == 0 and d.out_tokens == 0
