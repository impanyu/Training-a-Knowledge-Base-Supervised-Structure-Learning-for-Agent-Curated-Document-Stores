"""Train iteration (Phase 1 forward / Phase 2 backprop), test iteration, and
the multi-epoch driver (spec §4/§5).

Train Phase 1 (budget N1): search/read then ONE answer(); its RESULT reveals
gold + F1 and the phase ends immediately (leftover budget forfeited). Budget
exhausted without answer -> auto F1=0, gold still revealed at Phase 2 start.
Train Phase 2 (budget N2): search/read + 7 edits until done() or exhaustion;
dirty summaries/embeddings regenerate now (store.refresh at iteration end).

Test iteration (budget M, frozen KB): answer() ends it, its result is empty
("submitted"), gold never appears; exhaustion -> scored 0 and counted as
"unanswered" (distinct from answered-wrong)."""
import random
import time
from pathlib import Path

from kb.actions import EDIT_ACTIONS, MODE_TOOLS, dispatch, tools_for
from kb.grader import grade
from kb.memory import IterationMemory
from kb.store import save_snapshot

N1, N2, M, K = 15, 15, 15, 30

TRAIN_SYSTEM = (
    "You are training a document knowledge base on supervised question "
    "answering. Your objectives, in order:\n"
    "1. Answer the question correctly.\n"
    "2. Consolidate the verified reasoning into the store's structure, so the "
    "facts this question needed are easier to find next time.\n"
    "3. Generalize the fix to this question's CLASS of questions, not just "
    "this instance.\n"
    "HOW you consolidate is your choice - links, new index docs, copies, "
    "cleanup; "
    "you are judged only by whether a non-reasoning future reader could "
    "answer this class of question more easily.\n"
    "Statements are immutable text: you never write or edit sentences, only "
    "move, copy or delete existing statements by reference. Summaries and "
    "embeddings are maintained for you automatically.\n"
    "When answering, submit exactly one short answer (a name / phrase / "
    'number, or "unknown" if this universe cannot determine it); the result '
    "reveals the gold answer and your F1."
)

READER_SYSTEM = (
    "Answer the question using search and read over a document knowledge "
    "base. Keep the answer short: a name, phrase or number. If the answer "
    'cannot be determined from the documents, answer "unknown". Submit it '
    "with answer()."
)


def _fmt(d) -> str:
    args = ", ".join(f"{k}={v!r}" for k, v in d.inp.items())
    return f"{d.name}({args})"


def _turn(policy, system, mem, remaining, mode):
    ctx = mem.render(f"Remaining actions this phase: {remaining}.")
    return policy.decide(system, ctx, tools_for(mode))


def train_iteration(store, policy, q, log, epoch, n1=N1, n2=N2, k=K) -> dict:
    t0 = time.perf_counter()
    tok_in = tok_out = 0
    mem = IterationMemory(k)
    mem.reset(f"TRAIN QUESTION {q.qid} [phase 1: answer it]\n{q.text}")
    answer, answered_at, f1v, emv = None, None, 0.0, 0.0
    p1_steps = 0
    for step in range(1, n1 + 1):
        p1_steps = step
        d = _turn(policy, TRAIN_SYSTEM, mem, n1 - step + 1, "train_forward")
        if d.name == "answer":
            answer = str(d.inp.get("text", ""))
            f1v, emv = grade(answer, q.golds, q.unanswerable)
            answered_at = step
            result = f'submitted. gold answer: "{q.golds[0]}" | your F1: {f1v:.2f}'
        elif d.name in MODE_TOOLS["train_forward"]:
            result = dispatch(store, d.name, d.inp)
        else:
            result = f"ERROR: {d.name} is not available in this phase"
        mem.add(_fmt(d), result)
        tok_in, tok_out = tok_in + d.in_tokens, tok_out + d.out_tokens
        log.trace({"kind": "train", "epoch": epoch, "qid": q.qid, "phase": 1,
                   "step": step, "action": d.name, "input": d.inp,
                   "result": result, "tokens_in": d.in_tokens,
                   "tokens_out": d.out_tokens})
        if answered_at:
            break                       # leftover Phase-1 budget is forfeited

    mine = (f'your answer: "{answer}"' if answer is not None
            else "your answer: (none - budget exhausted, scored F1 0.00)")
    mem.set_task(
        f"TRAIN QUESTION {q.qid} [phase 2: consolidate]\n{q.text}\n"
        f'{mine}\ngold answer: "{q.golds[0]}" | F1 {f1v:.2f}\n'
        "Consolidate what you verified into the store's structure so this "
        "class of question gets easier; done() when finished.")
    edits = {a: 0 for a in EDIT_ACTIONS}
    p2_steps = 0
    for step in range(1, n2 + 1):
        p2_steps = step
        d = _turn(policy, TRAIN_SYSTEM, mem, n2 - step + 1, "train_backprop")
        if d.name == "done":
            result = "phase complete"
        elif d.name in MODE_TOOLS["train_backprop"]:
            result = dispatch(store, d.name, d.inp)
            if d.name in edits and not result.startswith("ERROR"):
                edits[d.name] += 1
        else:
            result = f"ERROR: {d.name} is not available in this phase"
        mem.add(_fmt(d), result)
        tok_in, tok_out = tok_in + d.in_tokens, tok_out + d.out_tokens
        log.trace({"kind": "train", "epoch": epoch, "qid": q.qid, "phase": 2,
                   "step": step, "action": d.name, "input": d.inp,
                   "result": result, "tokens_in": d.in_tokens,
                   "tokens_out": d.out_tokens})
        if d.name == "done":
            break
    regens = store.refresh()            # batch regeneration, iteration end only

    row = {"epoch": epoch, "qid": q.qid, "template": q.template,
           "category": q.category, "f1": f1v, "em": emv, "answer": answer,
           "answered_at": answered_at, "p1_steps": p1_steps,
           "p2_steps": p2_steps, "edits": edits, "regens": regens,
           "tokens_in": tok_in, "tokens_out": tok_out,
           "seconds": time.perf_counter() - t0}
    log.train(row)
    return row


def test_iteration(store, policy, q, log, split, epoch, m=M, k=K) -> dict:
    t0 = time.perf_counter()
    tok_in = tok_out = 0
    mem = IterationMemory(k)                     # fresh memory each question
    mem.reset(f"QUESTION {q.qid}\n{q.text}")
    answer, steps = None, 0
    for step in range(1, m + 1):
        steps = step
        d = _turn(policy, READER_SYSTEM, mem, m - step + 1, "test")
        if d.name == "answer":
            answer = str(d.inp.get("text", ""))
            result = "submitted"                 # gold never appears in test
        elif d.name in MODE_TOOLS["test"]:
            result = dispatch(store, d.name, d.inp)
        else:
            result = f"ERROR: {d.name} is not available in this phase"
        mem.add(_fmt(d), result)
        tok_in, tok_out = tok_in + d.in_tokens, tok_out + d.out_tokens
        log.trace({"kind": "test", "epoch": epoch, "qid": q.qid, "phase": 0,
                   "step": step, "action": d.name, "input": d.inp,
                   "result": result, "tokens_in": d.in_tokens,
                   "tokens_out": d.out_tokens})
        if answer is not None:
            break
    if answer is None:
        f1v, emv, status = 0.0, 0.0, "unanswered"
    else:
        f1v, emv = grade(answer, q.golds, q.unanswerable)
        status = "answered" if emv == 1.0 else "wrong"

    row = {"epoch": epoch, "split": split, "qid": q.qid, "template": q.template,
           "category": q.category, "flavor": q.eval_flavor, "f1": f1v,
           "em": emv, "steps": steps, "status": status, "answer": answer,
           "tokens_in": tok_in, "tokens_out": tok_out,
           "seconds": time.perf_counter() - t0}
    log.test(row)
    return row


def run_test(store, policy, universe, split, log, epoch=0, m=M, k=K,
             limit=None) -> list[dict]:
    qids = universe.splits[split][:limit]
    return [test_iteration(store, policy, universe.questions[qid], log,
                           split, epoch, m, k) for qid in qids]


def _log_stats(log, store, epoch, train_rows=()):
    """kb_stats row: store shape (incl. approximate statement_tokens) plus
    this epoch's training cost — tokens and wall-clock seconds."""
    log.stats({"epoch": epoch, **store.stats(),
               "summarizer_tokens_in": store.summarizer.tokens_in,
               "summarizer_tokens_out": store.summarizer.tokens_out,
               "train_iterations": len(train_rows),
               "train_tokens_in": sum(r["tokens_in"] for r in train_rows),
               "train_tokens_out": sum(r["tokens_out"] for r in train_rows),
               "train_seconds": sum(r["seconds"] for r in train_rows)})


def run_training(store, policy, universe, log, out_dir, epochs=1, seed=0,
                 n1=N1, n2=N2, m=M, k=K, train_size=None,
                 eval_each_epoch=False, test_policy=None,
                 universe_path=None) -> None:
    """Epoch driver: seeded per-epoch shuffle of the train split, per-epoch KB
    snapshots (epoch 0 = the untrained store = pure RAG baseline), optional
    evaluation on the small EVAL split after every snapshot. Training never
    touches test_in/test_out (T39.1): the full test runs exactly once, after
    training, via kb.test on the snapshot of choice."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    test_policy = test_policy or policy
    train_qids = universe.splits["train"][:train_size]

    def _eval(epoch):
        run_test(store, test_policy, universe, "eval", log, epoch=epoch,
                 m=m, k=k)

    save_snapshot(store, out / "kb_epoch_0.json", universe_path)
    _log_stats(log, store, 0)
    if eval_each_epoch:
        _eval(0)
    for epoch in range(1, epochs + 1):
        order = list(train_qids)
        random.Random(seed * 100003 + epoch).shuffle(order)
        rows = [train_iteration(store, policy, universe.questions[qid], log,
                                epoch, n1, n2, k) for qid in order]
        save_snapshot(store, out / f"kb_epoch_{epoch}.json", universe_path)
        _log_stats(log, store, epoch, rows)
        if eval_each_epoch:
            _eval(epoch)
