# v4: flat questions with quotas (spec)

v4 removes hierarchical task trees. **A task is one question.** The WORLD posts
each question with a **quota** (how many times it may be claimed and paid for);
claiming decrements the remaining quota, and a question with zero remaining is
no longer claimable. Total system demand = sum of quotas.

Rationale (user, 2026-08-01): the tree machinery (decompose, packaged JSON
delivery, sentence addressing, semantic-locality clustering) cost far more
complexity than it bought. Repeats now come from an explicit quota rather than
from the same leaf appearing under several trees, which makes knowledge reuse
and specialization easier to reason about and to measure.

## 1. Data: the question bank

`data/v4/bank.json` (built by `scripts/build_bank.py`):

```json
{"questions": [{"qid": "q0001", "text": "...", "answers": ["..."],
                "difficulty": "2hop", "price": 63000, "quota": 3, "topic": "k07"}],
 "total_units": 1478, "n_topics": 25}
```

- 500 questions, quota uniform in 1..5 (1478 claimable units), prices unchanged
  from the profit-regime calibration (63k / 105k / 157.5k for 2/3/4-hop).
- `topic` is a k-means cluster id over question embeddings. It is **metadata
  only** — never shown to agents — and exists so per-agent specialization stays
  measurable without a task tree.

`src/ca/bank.py` replaces `tasktree.py`:

```python
@dataclass
class Question:            # moved here from taskboard.py
    qid: str; text: str; answers: list[str]
    difficulty: str; price: int; quota: int; topic: str

class QuestionBank:
    questions: dict[str, Question]
    @classmethod
    def from_json(cls, path) -> "QuestionBank"
    def get(self, qid) -> Question          # unknown qid -> BankError listing near ids
    def total_units(self) -> int            # sum of quotas
```

Addressing is **by qid only** — no sentence matching, no fuzzy resolution
(`tasktree.py`, its fuzzy/ambiguity machinery, and `build_tasks.py` are deleted).

## 2. Board: quota accounting

`src/ca/board.py` (replaces `taskboard.py`'s tree logic):

```python
@dataclass
class Claim:       agent: str; round: int
@dataclass
class Result:      qid: str; agent: str; submitted: str; f1: float; em: float; payout: int; round: int

class QuestionBoard:
    def __init__(self, bank, ledger)
    remaining: dict[qid, int]                 # starts at quota
    active: dict[qid, list[Claim]]
    strikes: dict[(qid, agent), int]
    results: list[Result]

    def open_questions(self, viewer=None) -> list[Question]   # remaining > 0; per-viewer stable shuffle
    def claim(self, agent, qid, round) -> Question
    def expire_claims(self, round, ttl) -> list[str]           # returns expired qids
    def deliver(self, agent, qid, answer) -> Result
    def all_done(self) -> bool
```

Rules (inherited from v3's board, adapted):

- `claim`: fails if `remaining[qid] == 0`, if the agent already holds an active
  claim on that qid, or if `strikes[(qid, agent)] >= 2`. On success:
  `remaining -= 1`, append Claim, `strikes += 1`.
- `expire_claims`: an active claim older than `claim_ttl` (8) rounds is dropped
  and `remaining += 1` — **quota is returned to the pool** (a failed attempt
  must not destroy demand). The strike stays, so an agent gets at most two
  attempts per question.
- `deliver`: requires an active claim by that agent. Grades against the gold
  answers, mints `round(price * f1)` to the agent, records a Result, removes the
  claim. Remaining is **not** restored (this unit is consumed).
- Several agents may hold the same question concurrently when quota > 1.

## 3. Memory: one per-agent vector store

`src/ca/memory.py`'s `LongTermMemory` becomes `AgentMemory`, backed by a local
Chroma collection per bucket (chromadb's default ONNX embedder — local, no API
cost). `solutions.py` is deleted; answers live here as ordinary text entries
with metadata.

```python
class AgentMemory:
    def __init__(self, shared: bool = False, persist_dir: str | None = None)
    def write(self, agent, text, *, kind="note", qid=None, f1=None) -> None
    def search(self, agent, query, k=5) -> list[dict]     # semantic, notes + answers
    def answer(self, agent, qid) -> dict | None           # exact, metadata filter
    def n_answers(self, agent) -> int
    def to_state(self) / from_state(state)                # (id, text, metadata) triples; embeddings recomputed on restore
```

An answer is **an ordinary memory entry** — the same store, the same semantic
index as a hand-written note — that merely carries extra metadata so it can
*also* be retrieved by qid:

    [q0042] Which river flows through Orleans? -> "Loire" (F1 1.00)

metadata `{"kind": "answer", "qid": "q0042", "f1": 1.0}`. So
`memory_search("river in France")` finds it by meaning and `answer(agent,
"q0042")` finds it exactly by id.

Memory is **append-only**, like any notebook: answering the same question twice
leaves two entries, and the improvement history is itself information the agent
can see. `answer()` returns the matching entries; the claim line quotes the
best-known one (highest F1, ties broken by recency).

**Writes are automatic** (no agent action records anything):

1. after a graded WORLD delivery — answer + F1;
2. after receiving a qid-bound contract deliverable — answer, F1 unknown.

`memory_write` / `memory_search` remain for free-text notes.

**Reads are automatic**: `claim_question(qid)` attaches any stored answer to the
claim result, including its F1, so the agent can decide whether to re-solve:

    claimed [q0042] Which river flows through Orleans? (2hop, reward 63000, 2 left)
    memory: you previously answered "Loire" (F1 0.30) — low quality, consider re-solving

C2 (`shared_memory`) makes every agent share one bucket: notes *and* answers.

## 4. Actions

Solving: `retrieve`, `work_on`, `deliver_work`.
Admin: `list_questions`, `claim_question`, `send_message`, `read_chat`,
`propose_contract` / `accept_contract` / `reject_contract` / `counter_offer` /
`cancel_contract` / `set_price`, `pay`, `propose_loan` / `accept_loan` /
`repay_loan`, `push_goal` / `pop_goal`, `memory_write`, `memory_search`,
`check_balance`, `list_agents`.

Removed: `decompose`, `recall_solutions`, `list_tasks`, `claim_task`.

- `list_questions(offset)` — `[q0042] <text> (2hop, reward 63000, 3 left)`,
  per-viewer stable shuffle (the v3 de-herding fix is kept), paginated.
- `claim_question(qid)` — see §3 for the auto-recall line.
- `deliver_work(target_id, content)` — `target_id` is a qid (graded WORLD
  delivery; `content` is the **short answer string**, no JSON) or a contract id.
- Contracts bind to a question when `task` is exactly a qid; a bound contract's
  deliverable is that question's answer text.

## 5. Configurations

Unchanged single-factor spectrum C0–C7; only C2's wording changes: **shared
long-term memory** (notes and answers), flag `shared_memory`.

## 6. Metrics

- `demand_absorbed` = delivered units / `total_units` (replaces task_completion_rate)
- `n_answered`, `total_f1`, `accuracy_per_ktok_solving`, `coordination_overhead`,
  `coverage` (minted/burned), credit block, `gini`, `bankrupt_rate` — unchanged
- `specialization` — per-agent Herfindahl over **topics** of delivered questions
- `memory_hit_rate` = claims whose result carried a stored answer / all claims
- `improvement_rate` = repeat deliveries whose F1 beat the stored F1 / repeat
  deliveries that had a stored answer (new: quotas make second attempts possible)
