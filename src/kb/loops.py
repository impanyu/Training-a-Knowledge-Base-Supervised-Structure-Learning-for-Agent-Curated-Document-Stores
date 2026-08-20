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

# N1 (forward) and M (exam) are deliberately equal, so the training
# forward pass and the frozen exam differ only in the store. N2 is
# larger: the backward pass must both diagnose the trajectory and build
# structure, and in the v10L run diagnosis consumed 79% of a 15-action
# budget on failed questions, leaving 0.46 links per iteration.
# 40 gives room to locate an index's members and attach them; with
# link_many the attaching is one action, so the budget goes to search.
N1, N2, M, K = 15, 40, 15, 30

# Two skills, each with goal, means and a worked example. RETRIEVAL_SKILL is
# used by BOTH train Phase 1 and the frozen-store exam, so the forward pass
# and the test are the same policy on the same instructions -- the store is
# then the only thing that differs between them. CURATION_SKILL is Phase 2
# only; it never reaches a reader, so nothing it says can leak into a score.

RETRIEVAL_SKILL = (
    "You answer questions from a knowledge base you cannot change.\n"
    "\nGOAL. Produce the correct short answer within your action budget. "
    "A name, a phrase or a number - or \"unknown\" if this universe cannot "
    "determine it. Answering costs nothing extra; running out of budget "
    "scores zero, so do not keep searching once you can answer.\n"
    "\nMEANS. The store is a graph of single-sentence notes.\n"
    "  search(query) - the five notes whose text is most similar to your "
    "query. It is similarity, not understanding: it finds notes that "
    "SOUND like the query, so query with the words the note itself would "
    "use (a full name, an attribute word), not with the question.\n"
    "  read(id) - that note, AND the full text of every note linked to "
    "it. Links are free information: reading one well-connected note can "
    "return what several searches would have. If a search result looks "
    "like an entry point for a group, reading it is usually worth a step.\n"
    "  answer(text) - submit and end.\n"
    "\nEXAMPLE (illustrative; these notes are not in your store). "
    "Question: what is the hobby of the spouse of Mira Ashford?\n"
    "  search(\"Mira Ashford spouse\") -> n41: Mira Ashford is married to "
    "Tomas Ashford. (also n12, n88, ... less relevant)\n"
    "  read(n41) -> n41 plus its links: n42: Tomas Ashford's hobby is "
    "falconry. n43: Tomas Ashford's job is cooper.\n"
    "  answer(\"falconry\")\n"
    "The second hop cost a read rather than a search because the note was "
    "linked. Where links are absent, search again with the name you just "
    "learned.\n"
    "  A second case. Question: what is the hobby of Tomas Ashford's "
    "second cousin? Searches for the relation return nothing resembling "
    "it, and reading the notes about Tomas Ashford shows a spouse, "
    "parents and children but no such relation anywhere in this "
    "universe. answer(\"unknown\") is then correct and cheap - a question "
    "the store cannot determine is not a question to spend fifteen "
    "actions on.\n"
)

CURATION_SKILL = (
    "You are training a knowledge base on supervised question answering: "
    "you answer a question, are shown the gold answer, and then edit the "
    "store itself. The store is a graph of single-sentence notes, found "
    "by semantic search over their text and linked to each other by id.\n"
    "\nGOAL. Build the INDEX NOTES this question needed. A question names "
    "things - a person, a city, a job, a hobby, a relation of a person - "
    "and each named thing is a KEY. For every key the question named, "
    "there should be one note that stands for it and links to everything "
    "under it. Build the ones that are missing; extend the ones that are "
    "there and incomplete. You do not have to abstract the question or "
    "guess what will be asked next: an index for the city this question "
    "happened to name serves every later question about that city, and "
    "that is where the generalization comes from.\n"
    "  Examples of the mapping, which is meant to be this direct:\n"
    "    who lives in Fenmarch          -> \"Residents of Fenmarch\"\n"
    "    who is Felix Abernathy         -> \"Felix Abernathy\", linked to "
    "every note about him\n"
    "    who are Felix's friends        -> \"Friends of Felix Abernathy\"\n"
    "    friends of Felix's friends     -> \"Friends of Felix Abernathy\" "
    "first, then \"Friends of\" each of those people\n"
    "    which cooper lives in Fenmarch -> \"People whose job is cooper\" "
    "and \"Residents of Fenmarch\"\n"
    "  A multi-hop question names a key at every hop; build each level you "
    "had to walk through, not only the last.\n"
    "\nMEANS. search and read as before, plus add(text), edit(id, text), "
    "delete(id), link(a, b), link_many(a, targets), unlink(a, b), "
    "done(). Four facts about this environment; the rest is your "
    "judgement.\n"
    "  - search ranks by similarity and returns five - the same five "
    "however often you ask, which is what the reader you are building "
    "for has, and which cannot enumerate anything. search_keyword is "
    "different: it returns EVERY note containing the keywords you give, "
    "with a count, so one call closes a set exactly. If more than forty "
    "match, only the count comes back: that key is too broad to index.\n"
    "  - link_many costs ONE action however many targets it names, so "
    "attaching is cheap and finding is what your budget goes on.\n"
    "  - read(id) returns that note AND the full text of every note "
    "linked to it, but not what those in turn link to. One level per "
    "read.\n"
    "  - ONE note holds ONE statement. Several statements are several "
    "notes; never pack two facts into one.\n"
    "\nAN INDEX POINTS; IT DOES NOT LIST. Its members belong in its "
    "LINKS, never in its sentence. \"People whose hobby is model trains "
    "are Lowell Briarcliff, Bennett Quixwood and Thea Ingleside\" is the "
    "wrong shape: it packs several facts into one note, it goes stale the "
    "moment anyone changes hobby, the store already states each of those "
    "facts in its own note, and a reader can only match the whole "
    "sentence rather than the person it wants. Write the note as the name "
    "of the key - \"People whose hobby is model trains\" - and attach the "
    "members with link_many. An index may point at other indexes rather "
    "than at raw notes: \"Residents of Fenmarch\" pointing at a note per "
    "person, each of which points at that person's facts, is often "
    "better than pointing at the residence notes directly, because the "
    "person notes are reused by every other index that names them.\n"
    "\nCOMPLETENESS. An index is a promise that everything under its key "
    "is reachable from it, and a reader believes it: finding your index, "
    "it stops searching. A partial index is therefore not weak but WRONG "
    "- it makes a reader count nine where there are fourteen, "
    "confidently. Use search_keyword to close the set exactly, then "
    "link_many to attach it in one action. An index with no links is "
    "worth nothing; five with one link each are worth less than one with "
    "ten. If the budget runs out mid-index, the next question naming that "
    "key owes the rest.\n"
    "\nEXAMPLE (illustrative; these notes are not in your store). You "
    "were asked which cooper lives in Fenmarch, searched \"cooper "
    "Fenmarch\", got five unrelated notes and ran out of budget; the gold "
    "is Tomas Ashford. The question named two keys, the job and the city. "
    "search_keyword([\"lives in the city of Fenmarch\"]) reports "
    "twenty-nine notes and shows them all, so that set is closed; "
    "add(\"Residents of Fenmarch\") -> n90 and link_many(n90, [...]) "
    "attaches them in one action. Then the same for "
    "search_keyword([\"job is cooper\"]) -> add(\"People whose job is "
    "cooper\") -> n120. Anyone later asking about Fenmarch, or about "
    "coopers, now starts one read from the answer. What you did not "
    "spend the budget on: a note saying Tomas Ashford is the cooper of "
    "Fenmarch - true, allowed, and last, because it serves the one "
    "question that will not be asked again.\n"
    "\nEvery note you add must earn its retrieval slot: it competes with "
    "the whole store in search, and a reader that retrieves it and learns "
    "nothing has paid a step for it. Do not add a note whose content the "
    "store already carries, edit rather than duplicate, delete redundancy "
    "you create, and if every key this question named already has a "
    "complete index, change NOTHING. Note text is one short "
    "self-contained sentence with full names and no pronouns; embeddings "
    "are maintained for you.\n"
)


# Phase 1 and the exam share the retrieval skill; the phase-1 task block adds
# the one difference (its answer() result reveals the gold).
TRAIN_SYSTEM = CURATION_SKILL          # backward compatibility for callers
READER_SYSTEM = RETRIEVAL_SKILL


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
        d = _turn(policy, RETRIEVAL_SKILL, mem, n1 - step + 1, "train_forward")
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
    if answer is None:
        outcome = "The budget ran out before you located the answer. "
    elif f1v < 0.5:
        outcome = "Your answer was wrong. "
    else:
        outcome = "Your answer was right. "
    mem.set_task(
        f"TRAIN QUESTION {q.qid} [phase 2: backward]\n{q.text}\n"
        f'{mine}\ngold answer: "{q.golds[0]}" | F1 {f1v:.2f}\n'
        + outcome
        + "Name the keys this question mentioned - the people, places, "
        "jobs, hobbies and relations it named - and make sure each one "
        "has a complete index note. Build what is missing, extend what "
        "is partial, and change nothing if they are already complete. "
        "done() when finished.")
    edits = {a: 0 for a in EDIT_ACTIONS}
    p2_steps = 0
    for step in range(1, n2 + 1):
        p2_steps = step
        d = _turn(policy, CURATION_SKILL, mem, n2 - step + 1, "train_backward")
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
        d = _turn(policy, RETRIEVAL_SKILL, mem, m - step + 1, "test")
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
