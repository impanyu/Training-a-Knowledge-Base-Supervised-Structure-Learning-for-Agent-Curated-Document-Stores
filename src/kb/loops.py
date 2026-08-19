"""Train iteration (Phase 1 forward / Phase 2 backward), test iteration, and
the multi-epoch driver (spec §4/§5).

Train Phase 1 (budget N1): search/read then ONE answer(); its RESULT reveals
gold + F1 and the phase ends immediately (leftover budget forfeited). Budget
exhausted without answer -> auto F1=0, gold still revealed at Phase 2 start.
Train Phase 2 (budget N2): search/read + 5 edits until done() or exhaustion;
dirty nodes re-embed now (store.refresh at iteration end).

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
    "You are training a knowledge base on supervised question answering. "
    "The store is a graph of single-sentence notes: each note is one "
    "self-contained statement, found by semantic search over its text and "
    "linked to other notes by id. Your objectives, in order:\n"
    "1. Answer the question correctly.\n"
    "2. Consolidate the verified reasoning into the graph, so the facts "
    "this question needed are easier to find next time.\n"
    "3. Build an INDEX for this question's TYPE. You just computed an "
    "answer; do not store it. That exact question will not be asked "
    "again - questions of its type will, about other entities - and a "
    "store of verified answers is a cache covering one instance each. "
    "What transfers is an index.\n"
    "   A question type is defined by the KEY the reader must search for "
    "first: a person, a city, a job, a hobby, a family. That first search "
    "is where search is weakest, because the question names a class "
    "rather than a note. An index note is the entry point for one key: "
    "named the way a searcher would phrase that key, so search reliably "
    "lands on it, and LINKED to every note belonging to it. This works "
    "because reading a note returns that note AND the full text of every "
    "note linked to it - so one read of a complete index replaces a "
    "sequence of uncertain searches, for every question of that type, "
    "about any entity.\n"
    "   Everything you can build is one of two things, and both are "
    "edges. (i) An INDEX note pointing at the notes that belong to its "
    "key. (ii) An EDGE between notes that already exist - two facts a "
    "chain has to step between, or two INDEXES, one pointing at the "
    "other. Index-to-index edges are how navigation gets levels: a job "
    "index pointing at the city indexes whose residents hold that job "
    "turns a two-hop question into two reads. Build the level you needed "
    "this time.\n"
    "   Match the index to the type: questions asking for a person's "
    "attributes want a per-person index linking that person's notes; "
    "questions naming a group and asking which member has a property "
    "want an index for the group, linking its members' notes; questions "
    "that walk a relation want the relation's notes linked to the notes "
    "about the people they name; questions that count want the counted "
    "items linked to their owner.\n"
    "   An index is NOT a roster of answers. Do not write out who "
    "satisfies a condition - link to the notes and let the reader read "
    "them. Its value is the COMPLETENESS of its links, not its prose: an "
    "index holding three of thirty members is worse than none, because it "
    "looks authoritative and is not. You will not finish one in a single "
    "iteration, and you are not meant to. Create it when its key first "
    "comes up, and EXTEND it every later time you meet that key: search "
    "for members it is missing and link them. An index grows across "
    "iterations; that accumulation is what training is.\n"
    "4. If you add a note, it must earn its retrieval slot: it either "
    "states a fact the store does not already say, or it is an index "
    "carrying many links. A note that does neither is dead weight - it "
    "consumes a retrieval slot and a read step and leaves the reader "
    "where it started. Never write a note that only describes what it "
    "points at.\n"
    "5. Keep the graph PARSIMONIOUS. Redundant notes compete in search and "
    "bury each other: don't add a note whose content the graph already "
    "carries; edit an existing note rather than duplicating it; delete any "
    "redundancy you create; if the graph already serves this question's "
    "class well, change NOTHING - a clean miss of the edit budget is better "
    "than a redundant note. Organize and navigate; don't duplicate.\n"
    "HOW you consolidate is your choice - links, navigation notes, edits, "
    "cleanup; you are judged only by whether a non-reasoning future reader "
    "could answer this class of question more easily.\n"
    "Any note text you write must be a single short self-contained sentence "
    "with full names, no pronouns. Embeddings are maintained for you "
    "automatically.\n"
    "When answering, submit exactly one short answer (a name / phrase / "
    'number, or "unknown" if this universe cannot determine it); the result '
    "reveals the gold answer and your F1."
)

READER_SYSTEM = (
    "Answer the question using search and read over a knowledge base of "
    "single-sentence notes. Keep the answer short: a name, phrase or "
    'number. If the answer cannot be determined from the notes, answer '
    '"unknown". Submit it with answer().'
)


def _fmt(d) -> str:
    args = ", ".join(f"{k}={v!r}" for k, v in d.inp.items())
    return f"{d.name}({args})"


def _turn(policy, system, mem, remaining, mode):
    ctx = mem.render(f"Remaining actions this phase: {remaining}.")
    return policy.decide(system, ctx, tools_for(mode))


def train_iteration(store, policy, q, log, epoch, n1=N1, n2=N2, k=K) -> dict:
    t0 = time.perf_counter()
    merges0 = store.merges              # dedup merges this iteration (T42)
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
        f"TRAIN QUESTION {q.qid} [phase 2: backward]\n{q.text}\n"
        f'{mine}\ngold answer: "{q.golds[0]}" | F1 {f1v:.2f}\n'
        "Before editing: name this question's TYPE and the KEY a reader "
        "must search for first to answer any question of that type. Then "
        "check whether an index note for that key exists - search for it. "
        "If it does not, create it and link what you can find. If it "
        "does, EXTEND it: find members it is missing and link them. Do "
        "not store the answer you just computed. If the index for this "
        "key is already complete, change nothing. done() when finished.")
    edits = {a: 0 for a in EDIT_ACTIONS}
    p2_steps = 0
    for step in range(1, n2 + 1):
        p2_steps = step
        d = _turn(policy, TRAIN_SYSTEM, mem, n2 - step + 1, "train_backward")
        if d.name == "done":
            result = "phase complete"
        elif d.name in MODE_TOOLS["train_backward"]:
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
           "merges": store.merges - merges0,
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
    """kb_stats row: graph shape (incl. approximate statement_tokens) plus
    this epoch's training cost — tokens and wall-clock seconds."""
    log.stats({"epoch": epoch, **store.stats(),
               "train_iterations": len(train_rows),
               "train_tokens_in": sum(r["tokens_in"] for r in train_rows),
               "train_tokens_out": sum(r["tokens_out"] for r in train_rows),
               "train_seconds": sum(r["seconds"] for r in train_rows)})


def run_training(store, policy, universe, log, out_dir, epochs=1, seed=0,
                 n1=N1, n2=N2, m=M, k=K, train_size=None,
                 eval_each_epoch=False, test_policy=None,
                 universe_path=None, snapshot_every=0) -> None:
    """Epoch driver: seeded per-epoch shuffle of the train split, per-epoch KB
    snapshots (epoch 0 = the untrained store = pure RAG baseline), optional
    evaluation on the small EVAL split after every snapshot. Training never
    touches test_in/test_out (T39.1): the full test runs exactly once, after
    training, via kb.test on the snapshot of choice. snapshot_every=N > 0
    (T39.8) additionally writes kb_iter_XXXX.json every N train iterations,
    numbered across epochs — the same numbering kb.replay --at uses; the
    default 0 keeps per-epoch snapshots only."""
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
    it = 0
    for epoch in range(1, epochs + 1):
        order = list(train_qids)
        random.Random(seed * 100003 + epoch).shuffle(order)
        rows = []
        for qid in order:
            rows.append(train_iteration(store, policy, universe.questions[qid],
                                        log, epoch, n1, n2, k))
            it += 1
            if snapshot_every and it % snapshot_every == 0:
                save_snapshot(store, out / f"kb_iter_{it:04d}.json",
                              universe_path)
        save_snapshot(store, out / f"kb_epoch_{epoch}.json", universe_path)
        _log_stats(log, store, epoch, rows)
        if eval_each_epoch:
            _eval(epoch)
