import ca.agent as agent_mod
from fixtures import demo_infra

from ca.agent import Agent, Decision, LLMPolicy, ScriptedPolicy


def make(level="C0"):
    infra = demo_infra(level)
    return infra.cfg, infra


def test_turn_executes_and_logs():
    cfg, infra = make()
    ag = Agent("agent_1", cfg, infra, ScriptedPolicy([("claim_question", {"qid": "q0001"})]))
    ev = ag.take_turn()
    assert ev["action"] == "claim_question" and ev["category"] == "admin"
    assert "claimed" in ev["result"]
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


def test_permission_denied_turn_still_reports_its_tokens():
    cfg, infra = make("C1")
    ag = Agent("agent_1", cfg, infra,
               ScriptedPolicy([("claim_question", {"qid": "q0001"})],
                              in_tokens=10, out_tokens=5))
    ev = ag.take_turn()
    assert ev["result"].startswith("ERROR")
    assert ev["category"] == "admin"  # claim_question is admin even when denied
    assert ev["tokens_in"] == 10 and ev["tokens_out"] == 5


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
        ("push_goal", {"note": "solve q0001"}), ("pop_goal", {})]))
    ag.take_turn()
    assert "solve q0001" in ag.goals.render()
    ag.take_turn()
    assert "solve q0001" not in ag.goals.render()


def test_root_goal_is_cooperative_and_solo_at_c7():
    cfg, infra = make()
    root = Agent("agent_1", cfg, infra, ScriptedPolicy([])).goals.render()
    assert ("[0] Cooperate with the other agents to answer as many questions "
            "correctly as possible. (root, permanent)") in root

    cfg7, infra7 = make("C7")
    root7 = Agent("agent_1", cfg7, infra7, ScriptedPolicy([])).goals.render()
    assert "[0] Answer as many questions correctly as possible. (root, permanent)" in root7
    assert "Cooperate" not in root7


def test_the_root_goal_cannot_be_popped():
    cfg, infra = make()
    ag = Agent("agent_1", cfg, infra, ScriptedPolicy([("pop_goal", {})]))
    ev = ag.take_turn()
    assert ev["result"].startswith("ERROR")
    assert "Cooperate" in ag.goals.render()


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


def test_turn_marks_chat_read():
    cfg, infra = make()
    infra.chat.send("agent_2", "agent_1", "ping", 0)
    ag = Agent("agent_1", cfg, infra, ScriptedPolicy([("list_agents", {})]))
    ag.take_turn()
    assert infra.chat.unread("agent_1") == []
