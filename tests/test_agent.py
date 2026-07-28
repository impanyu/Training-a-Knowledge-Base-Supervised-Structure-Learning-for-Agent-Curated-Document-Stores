import ca.agent as agent_mod
from ca.agent import Agent, ScriptedPolicy, Decision, LLMPolicy
from ca.config import LEVELS, ExperimentConfig
from ca.infra import Infra
from ca.retrieval import KeywordBackend
from ca.taskboard import Question

DOCS = [{"title": "Paris", "text": "Paris is the capital of France."}]


def make(level="L0"):
    cfg = ExperimentConfig(level=LEVELS[level], seed=0, seed_capital_total=1000)
    infra = Infra(cfg, [Question("q0001", "capital of France?", ["Paris"], "easy", 100)],
                  retriever=KeywordBackend(DOCS))
    return cfg, infra


def test_turn_executes_and_logs():
    cfg, infra = make()
    ag = Agent("agent_1", cfg, infra, ScriptedPolicy([("claim_question", {"qid": "q0001"})]))
    ev = ag.take_turn()
    assert ev["action"] == "claim_question" and ev["billable"] is False
    assert "claimed" in ev["result"]
    assert len(ag.fifo.items) == 1


def test_billing_on_billable_turn():
    cfg, infra = make()
    ag = Agent("agent_1", cfg, infra,
               ScriptedPolicy([("retrieve", {"query": "capital of France"})]))
    ag.policy.script[0] = ("retrieve", {"query": "capital of France"})
    # simulate LLM cost by wrapping decision tokens
    ag.policy = ScriptedPolicy([("retrieve", {"query": "capital of France"})],
                               in_tokens=100, out_tokens=20)
    start = infra.ledger.balance("agent_1")
    ev = ag.take_turn()
    assert ev["billable"] is True and ev["tokens_in"] == 100
    assert infra.ledger.balance("agent_1") == start - 120
    assert infra.ledger.conservation_ok()


def test_permission_denied_becomes_error_result_unbilled():
    cfg, infra = make("L1")
    ag = Agent("agent_1", cfg, infra, ScriptedPolicy([("claim_question", {"qid": "q0001"})]))
    start = infra.ledger.balance("agent_1")
    ev = ag.take_turn()
    assert ev["result"].startswith("ERROR")
    assert infra.ledger.balance("agent_1") == start


def test_goal_actions_update_local_stack():
    cfg, infra = make()
    ag = Agent("agent_1", cfg, infra, ScriptedPolicy([
        ("push_goal", {"note": "solve q0001"}), ("pop_goal", {})]))
    ag.take_turn()
    assert "solve q0001" in ag.goals.render()
    ag.take_turn()
    assert "solve q0001" not in ag.goals.render()


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
    ag = Agent("agent_1", cfg, infra, ScriptedPolicy([("check_balance", {})]))
    ag.take_turn()
    assert infra.chat.unread("agent_1") == []
