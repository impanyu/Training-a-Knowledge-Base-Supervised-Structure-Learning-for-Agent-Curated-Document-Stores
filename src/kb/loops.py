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
)

CURATION_SKILL = (
    "You are training a knowledge base on supervised question answering: "
    "you answer a question, are shown the gold answer, and then edit the "
    "store itself. The store is a graph of single-sentence notes, found "
    "by semantic search over their text and linked to each other by id.\n"
    "\nGOAL. Leave the store better for the NEXT question of this kind - "
    "asked about a different entity - than it was for this one. "
    "Generalizing is required, not optional: an iteration that leaves the "
    "store no better for a sibling question has failed, even if you "
    "answered correctly. That exact question will not be asked again, so "
    "prefer ACCESS STRUCTURE over the answer you just computed: an "
    "answer serves the one instance that produced it, whereas an index "
    "over some key serves every question entering through that key, and "
    "an edge serves every chain that must make that hop, whatever entity "
    "it concerns. Rank what you build by how many kinds of question it "
    "could serve, and spend the budget in that order.\n"
    "\nMEANS. search and read as before, plus add(text), edit(id, text), "
    "delete(id), link(a, b), link_many(a, targets), unlink(a, b), done(). "
    "link_many costs ONE action however many targets it names, so "
    "populating an index is cheap - collect ids with a few searches, then "
    "attach them in a single step. Reading a note returns "
    "that note AND the full text of every note linked to it, so a "
    "well-linked entry point replaces a sequence of uncertain searches. "
    "Build freely; there are five things you can create and all of them "
    "are worth creating when they earn their place.\n"
    "  NEW NOTES. (1) An INDEX note: one that exists to point at the "
    "notes belonging to some key, named the way a searcher would phrase "
    "that key. (2) A STATEMENT note: one that states a fact the store "
    "does not already carry - an aggregation, a derived relation, "
    "something scattered brought together - worth adding when more than "
    "one question would want it. A note that only records the answer you "
    "were just given is allowed but ranks last: it serves one instance. "
    "ONE note holds ONE statement. If what you want "
    "to record is several statements, add them as several notes and link "
    "them; never pack two facts into one note - a note carrying two "
    "facts matches neither cleanly in search and cannot be edited or "
    "deleted independently.\n"
    "  NEW LINKS. (3) statement to statement: the hop a chain had to "
    "make, so the next reader traverses instead of searching. (4) index "
    "to statement: what makes an index an index. (5) index to index: how "
    "navigation acquires levels, so a two-hop question becomes two "
    "reads.\n"
    "\nFIT THE INDEX TO THE QUESTION TYPE. Read the question you were "
    "just asked and decide what TYPE it is - not this instance, the "
    "shape: an attribute lookup about one person, a join over two "
    "properties, a walk along a relation, a count, an intersection of two "
    "sets. The type tells you what a reader of that type must search for "
    "FIRST, and that is the key your index must be built on, because the "
    "first search is where search is weakest: the question names a class "
    "rather than a note. An attribute lookup enters through a person, so "
    "index that person's notes together. A relation walk enters through "
    "the relation, so link relation notes to the notes about the people "
    "they name. A count enters through the owner, so link the counted "
    "items to it. Build the index the type needs, not the one that is "
    "easiest to name.\n"
    "  A type can have MORE THAN ONE key, and then each key deserves its "
    "own index. A join over job and city can be entered from either side: "
    "an index of residents by city serves the questions that name the "
    "city, and one of holders by job serves the questions that name the "
    "job, and neither substitutes for the other. Link the indexes to "
    "each other too, so a reader landing on one discovers the other "
    "without a further search. If the budget only stretches to one, "
    "build it "
    "completely and leave the other as the debt the next question of "
    "this type pays off; a complete index plus a missing one beats two "
    "half-built ones.\n"
    "\nSUGGESTED, NOT PRESCRIBED - you may find better on the rest. Indexes "
    "are worth returning to: one you extend on a later question covers "
    "more than one you abandon.\n"
    "  LINK WHAT THE READER WILL NEED, not just what matches the key. An "
    "index is not a membership list. A reader arriving through a key "
    "wants the facts it will be asked about, and one read returns the "
    "full text of everything linked - so an index of a city's residents "
    "that links only their residence notes forces the reader to chase "
    "each person separately, while the same index also linking those "
    "people's job, hobby and family notes answers a join, a count and a "
    "lookup in ONE read. Reach one hop further than the key strictly "
    "implies.\n"
    "  COMPLETENESS IS THE POINT. An index with no links is worth "
    "nothing, and one with some is worth less than it looks. An index is "
    "a promise that everything "
    "under its key is reachable from it, and a reader believes that "
    "promise: finding your index, it stops searching. So a partial index "
    "is not merely weak, it is WRONG - it makes a reader count nine "
    "residents when there are fourteen, or miss the member that was the "
    "answer, and it does so confidently. Before you leave an index, "
    "search its key several different ways - the key itself, its "
    "members' likely phrasings, the attribute that defines them - until "
    "a fresh search returns nobody you have not already linked. Then "
    "link them all with link_many, which costs one action. If you truly "
    "cannot finish, the next question on that key must finish it; an "
    "unfinished index is a debt, not an achievement.\n"
    "  Two ways an index fails. Its key can be too broad: \"People in the knowledge base\" or "
    "\"Marriages\" name the whole store, match everything and "
    "discriminate nothing, and no question will ever search for them. A "
    "key must be something a question would actually mention - a named "
    "person, a named city, a named job. And you can spread too thin: "
    "five indexes carrying one link each are worth less than one "
    "carrying ten. Finish the one you started before opening another. Use the gold "
    "answer as a LOCATOR, not as material for a post-mortem: searching "
    "it takes you straight to notes worth linking, which is time spent "
    "building rather than explaining. Do not spend the budget working "
    "out why the attempt failed - spend it leaving structure behind.\n"
    "\nEXAMPLE (illustrative; these notes are not in your store). The "
    "question was: which cooper lives in Fenmarch? You searched \"cooper "
    "Fenmarch\", got five unrelated notes and ran out of budget. The gold "
    "is Tomas Ashford.\n"
    "  Step one, name the TYPE. Not \"which cooper lives in Fenmarch\" but "
    "\"which holder of JOB lives in CITY\" - a join over two properties. "
    "Its instances are every (job, city) pair, hundreds of them, and none "
    "of them is the one you were just asked.\n"
    "  Step two, find the KEYS - plural here. A reader of this type has "
    "two ways in, the job or the city, and search handles neither: both "
    "name a class, not a note. So this type wants two indexes, residents "
    "by city and holders by job, plus an edge between them. Start with "
    "the city, the smaller set; the job index is the debt you settle "
    "next time a job is named.\n"
    "  Step three, locate ALL of them. search(\"lives in Fenmarch\") "
    "returns n44 \"Tomas Ashford lives in Fenmarch.\" and four more; "
    "search(\"Fenmarch resident\") returns four others; "
    "search(\"city of Fenmarch\") returns five, four already seen and one "
    "new. A fourth search returns nobody new, so the set is closed at "
    "fourteen. Stopping at the first five would have built an index that "
    "lies.\n"
    "  (1) new INDEX note: add(\"Residents of Fenmarch\") -> n90. Named as "
    "a searcher would phrase the key, so the next question mentioning "
    "Fenmarch lands on it. Not \"People in the knowledge base\", which "
    "names the whole store and discriminates nothing.\n"
    "  (4) index to statement: link_many(n90, [n44, n51, ... n133]) - "
    "one action, all fourteen residence notes - and then link_many(n90, "
    "[n43, n52, ... n134]) for those same people's JOB notes, which is "
    "what the type will actually be asked about. Now read(n90) returns "
    "twenty-eight statements in one action and \"Tomas Ashford's job is "
    "cooper\" is among them: the question is answered in a single read, "
    "with no search at all. Linking only the residence notes would have "
    "left the reader knowing fourteen names and none of their jobs.\n"
    "  (3) statement to statement: link(n43, n44), joining \"Tomas "
    "Ashford's job is cooper\" to \"Tomas Ashford lives in Fenmarch\" - "
    "the hop this chain needed. Useful, but it repairs one instance; the "
    "index above serves every question about Fenmarch.\n"
    "  (5) index to index: the second key. A question of this type that "
    "names the job rather than the city enters somewhere else entirely, "
    "so add(\"People whose job is cooper\") -> n120 and populate it the "
    "same way, its members' city notes included. Then link(n120, n90) "
    "and link(n90, n120), so a reader arriving at either sees the other "
    "exists. Multi-level navigation is where index-to-index edges pay "
    "most: one \"jobs\" index linking to every per-job index turns "
    "finding the right one into a read instead of a search.\n"
    "  (2) new STATEMENT note: while linking you notice the store never "
    "says how many people live in Fenmarch, and several types of "
    "question would want it: add(\"Fenmarch has fourteen residents.\") -> "
    "n121, link(n90, n121). One fact, one note - the founding year would "
    "be a second note, not a longer sentence.\n"
    "  What came LAST: a note saying Tomas Ashford is the cooper who "
    "lives in Fenmarch. You may add it - it is true and the store does "
    "not state it - but it serves exactly one question, and that "
    "question will not be asked again. The budget is small, so spend it "
    "first on structure many types of question can use.\n"
    "\nEvery note you add must earn its retrieval slot: it competes with "
    "the whole store in search, and a reader that retrieves it and learns "
    "nothing has paid a step for it. Keep the graph parsimonious - do not "
    "add a note whose content the store already carries, edit rather than "
    "duplicate, delete redundancy you create, and if the store already "
    "serves this kind of question well, change NOTHING. Any note text you "
    "write must be ONE short self-contained sentence stating ONE fact, "
    "with full names and no pronouns - split anything longer into "
    "separate notes; embeddings are maintained for you.\n"
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
        + "Leave the store better for the next question of this kind - "
        "asked about a different entity - than it was for this one. Do "
        "not spend the budget explaining this attempt; spend it "
        "building. If the store already serves that kind well, change "
        "nothing. done() when finished.")
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
