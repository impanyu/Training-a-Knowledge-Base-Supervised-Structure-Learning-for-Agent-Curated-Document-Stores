"""T27: Recorder solution-reuse tallies -- per-agent `solutions` summary block
(answers/decompositions stored, from infra.solutions.stats) plus n_recalls /
n_recall_hits counted live off the event stream in `log()`."""
from fixtures import demo_infra

from ca.recorder import Recorder


def _event(agent, action, result, round_no=1):
    return {"round": round_no, "agent": agent, "action": action, "input": {},
            "result": result, "category": "solving", "tokens_in": 1, "tokens_out": 1,
            "balance_after": 0}


def test_solutions_block_reflects_stats_and_recall_tallies(tmp_path):
    infra = demo_infra("C0")
    infra.solutions.record_decomposition("agent_1", "t0001", ["t0002", "q0003"])
    infra.solutions.record_answer("agent_1", "q0003", "4", f1=1.0)
    rec = Recorder(str(tmp_path))
    rec.log(_event("agent_1", "recall_solutions",
                   'known 1/1 answers under t0001: {"q0003": "4" (F1 1.00)}'))
    rec.log(_event("agent_1", "recall_solutions", "(no stored solutions under t0002)"))
    rec.log(_event("agent_2", "recall_solutions", "(no stored solutions under t0001)"))
    summary = rec.write_summary(infra, rounds_used=1)
    rec.close()

    assert summary["solutions"]["agent_1"] == {
        "answers": 1, "decompositions": 1, "n_recalls": 2, "n_recall_hits": 1,
    }
    assert summary["solutions"]["agent_2"] == {
        "answers": 0, "decompositions": 0, "n_recalls": 1, "n_recall_hits": 0,
    }
    # every agent gets an entry, even one that never called recall_solutions
    assert summary["solutions"]["agent_3"] == {
        "answers": 0, "decompositions": 0, "n_recalls": 0, "n_recall_hits": 0,
    }


def test_non_recall_events_do_not_affect_the_tally(tmp_path):
    infra = demo_infra("C0")
    rec = Recorder(str(tmp_path))
    rec.log(_event("agent_1", "check_balance", "balance: 100"))
    rec.log(_event("agent_1", "decompose", "[t0002] ..."))
    summary = rec.write_summary(infra, rounds_used=1)
    rec.close()
    assert summary["solutions"]["agent_1"]["n_recalls"] == 0
    assert summary["solutions"]["agent_1"]["n_recall_hits"] == 0


def test_shared_c2_bucket_is_reflected_per_agent_via_stats(tmp_path):
    infra = demo_infra("C2")
    infra.solutions.record_answer("agent_1", "q0001", "Paris", f1=1.0)
    rec = Recorder(str(tmp_path))
    summary = rec.write_summary(infra, rounds_used=0)
    rec.close()
    # the shared bucket is the same store no matter which agent's id you ask
    # through, so both agents see the one answer that lives in it
    assert summary["solutions"]["agent_1"]["answers"] == 1
    assert summary["solutions"]["agent_2"]["answers"] == 1
