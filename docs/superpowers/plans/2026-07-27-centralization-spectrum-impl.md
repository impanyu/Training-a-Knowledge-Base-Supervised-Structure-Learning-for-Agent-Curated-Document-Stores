# Centralization Spectrum Experiment Platform — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the multi-agent economy testbed described in `docs/superpowers/specs/2026-07-27-centralization-spectrum-design.md`: an infrastructure layer (ledger, chat, contracts/escrow, task board, memory, retrieval), LLM agents that act via a filtered action registry, six centralization configs L0–L5, round-robin scheduling, full tracing, and metrics.

**Architecture:** Pure-Python infrastructure (`src/ca/`) with an in-process action registry. Each agent = one Anthropic Messages API tool-use call per round (one action per turn, forced via `tool_choice: any` + `disable_parallel_tool_use`). Config levels differ ONLY in action visibility/permission rules. Deterministic tests use scripted (non-LLM) policies.

**Tech Stack:** Python ≥3.11, `anthropic` SDK, `bm25s` (retrieval), `datasets` (HF, data prep only), `pytest`.

## Global Constraints

- Agent model default: `claude-haiku-4-5` (design decision; validation runs may pass `claude-sonnet-5` via CLI flag). Never hardcode model elsewhere.
- Billing rule: a turn is billed iff the executed action is billable — billable = `retrieve`, `work_on`, and `deliver_work` targeting a WORLD question. Bill = that turn's LLM `input_tokens + output_tokens`.
- Ledger conservation invariant: `sum(balances) + sum(escrow) == seed_total + minted - burned`. Must hold after every operation.
- IDs: agents `interface`, `agent_1`…; questions `q0001`…; contracts `c0001`…. `deliver_work` billability keys off the `q`/`c` prefix.
- Bankruptcy: balance ≤ 0 → billable actions error; free actions still allowed.
- Config levels differ only via `LevelConfig` fields — no `if level == "L3"` scattered in handlers.
- All randomness through a seeded `random.Random`; no global `random` calls.
- Every trace event appended to JSONL immediately (crash-safe).

---

### Task 1: Project scaffolding

**Files:**
- Create: `pyproject.toml`, `src/ca/__init__.py`, `tests/__init__.py`, `.gitignore`

**Interfaces:**
- Produces: importable package `ca` under `src/`, pytest configured with `pythonpath=["src"]`.

- [ ] **Step 1: Write files**

`pyproject.toml`:
```toml
[project]
name = "centralized-agents"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = ["anthropic>=0.40", "bm25s>=0.2"]

[project.optional-dependencies]
data = ["datasets>=2.19"]
dev = ["pytest>=8"]

[tool.pytest.ini_options]
pythonpath = ["src"]
testpaths = ["tests"]
```

`src/ca/__init__.py`:
```python
"""Centralization-spectrum multi-agent economy testbed."""
```

`tests/__init__.py`: empty file.

`.gitignore`:
```
__pycache__/
*.egg-info/
.pytest_cache/
data/
runs/
.venv/
```

- [ ] **Step 2: Verify pytest runs**

Run: `cd /Users/ypan12/git_repo/centralized_agents && python -m pytest`
Expected: `no tests ran` (exit code 5 is fine).

- [ ] **Step 3: Commit**

```bash
git add pyproject.toml src tests .gitignore
git commit -m "chore: scaffold package layout"
```

---

### Task 2: Ledger (economy core)

**Files:**
- Create: `src/ca/economy.py`
- Test: `tests/test_economy.py`

**Interfaces:**
- Produces: `Ledger(seed_capital: dict[str,int])` with methods `balance(a)->int`, `is_bankrupt(a)->bool`, `burn(a, n)`, `mint(a, n)`, `transfer(frm, to, n)`, `lock_escrow(key, frm, n)`, `release_escrow(key, to)`, `refund_escrow(key, frm)`, `conservation_ok()->bool`, attr `escrow: dict[str,int]`; exception `InsufficientFunds`.

- [ ] **Step 1: Write the failing tests**

`tests/test_economy.py`:
```python
import pytest
from ca.economy import Ledger, InsufficientFunds


def make():
    return Ledger({"a": 100, "b": 50})


def test_seed_and_balance():
    led = make()
    assert led.balance("a") == 100
    assert led.conservation_ok()


def test_burn_can_go_negative_then_bankrupt():
    led = make()
    led.burn("b", 60)
    assert led.balance("b") == -10
    assert led.is_bankrupt("b")
    assert not led.is_bankrupt("a")
    assert led.conservation_ok()


def test_mint_and_transfer():
    led = make()
    led.mint("a", 30)          # WORLD payment
    led.transfer("a", "b", 20)
    assert led.balance("a") == 110 and led.balance("b") == 70
    assert led.conservation_ok()


def test_transfer_insufficient_raises():
    led = make()
    with pytest.raises(InsufficientFunds):
        led.transfer("b", "a", 51)


def test_escrow_lifecycle():
    led = make()
    led.lock_escrow("c1", "a", 40)
    assert led.balance("a") == 60 and led.escrow["c1"] == 40
    assert led.conservation_ok()
    led.release_escrow("c1", "b")
    assert led.balance("b") == 90 and "c1" not in led.escrow
    assert led.conservation_ok()


def test_escrow_refund_and_insufficient():
    led = make()
    with pytest.raises(InsufficientFunds):
        led.lock_escrow("c1", "b", 51)
    led.lock_escrow("c2", "a", 10)
    led.refund_escrow("c2", "a")
    assert led.balance("a") == 100
    assert led.conservation_ok()
```

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest tests/test_economy.py -q`
Expected: FAIL / ImportError (`ca.economy` missing).

- [ ] **Step 3: Implement**

`src/ca/economy.py`:
```python
"""Token ledger: balances, burn (LLM cost), mint (WORLD rewards), escrow."""


class InsufficientFunds(Exception):
    pass


class Ledger:
    def __init__(self, seed_capital: dict[str, int]):
        self.balances: dict[str, int] = dict(seed_capital)
        self.seed_total = sum(seed_capital.values())
        self.minted = 0
        self.burned = 0
        self.escrow: dict[str, int] = {}

    def balance(self, agent: str) -> int:
        return self.balances[agent]

    def is_bankrupt(self, agent: str) -> bool:
        return self.balances[agent] <= 0

    def burn(self, agent: str, amount: int) -> None:
        if amount < 0:
            raise ValueError("negative burn")
        self.balances[agent] -= amount  # may go negative -> bankruptcy
        self.burned += amount

    def mint(self, agent: str, amount: int) -> None:
        if amount < 0:
            raise ValueError("negative mint")
        self.balances[agent] += amount
        self.minted += amount

    def transfer(self, frm: str, to: str, amount: int) -> None:
        if amount <= 0:
            raise ValueError("transfer amount must be positive")
        if self.balances[frm] < amount:
            raise InsufficientFunds(f"{frm} has {self.balances[frm]} < {amount}")
        self.balances[frm] -= amount
        self.balances[to] += amount

    def lock_escrow(self, key: str, frm: str, amount: int) -> None:
        if self.balances[frm] < amount:
            raise InsufficientFunds(f"{frm} has {self.balances[frm]} < {amount}")
        self.balances[frm] -= amount
        self.escrow[key] = amount

    def release_escrow(self, key: str, to: str) -> None:
        self.balances[to] += self.escrow.pop(key)

    def refund_escrow(self, key: str, frm: str) -> None:
        self.balances[frm] += self.escrow.pop(key)

    def conservation_ok(self) -> bool:
        total = sum(self.balances.values()) + sum(self.escrow.values())
        return total == self.seed_total + self.minted - self.burned
```

- [ ] **Step 4: Run tests**

Run: `python -m pytest tests/test_economy.py -q`
Expected: 6 passed.

- [ ] **Step 5: Commit**

```bash
git add src/ca/economy.py tests/test_economy.py
git commit -m "feat: ledger with burn/mint/transfer/escrow and conservation invariant"
```

---

### Task 3: Answer grader (EM / F1)

**Files:**
- Create: `src/ca/grader.py`
- Test: `tests/test_grader.py`

**Interfaces:**
- Produces: `normalize(s)->str`, `exact_match(pred, golds)->float` (0/1), `f1(pred, golds)->float` (best over gold aliases).

- [ ] **Step 1: Write the failing tests**

`tests/test_grader.py`:
```python
from ca.grader import exact_match, f1, normalize


def test_normalize():
    assert normalize("The  Answer, is: Blue!") == "answer is blue"


def test_exact_match_with_aliases():
    assert exact_match("W. Somerset Maugham", ["William Somerset Maugham", "W. Somerset Maugham"]) == 1.0
    assert exact_match("Maugham", ["W. Somerset Maugham"]) == 0.0


def test_f1_partial():
    score = f1("Somerset Maugham", ["W. Somerset Maugham"])
    assert 0.0 < score < 1.0
    assert f1("", ["x"]) == 0.0
    assert f1("exact", ["exact"]) == 1.0
```

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest tests/test_grader.py -q` — Expected: ImportError.

- [ ] **Step 3: Implement**

`src/ca/grader.py` (standard SQuAD-style scoring):
```python
"""SQuAD-style answer scoring: normalization, exact match, token F1."""
import re
import string
from collections import Counter


def normalize(s: str) -> str:
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def _f1_single(pred: str, gold: str) -> float:
    pt, gt = normalize(pred).split(), normalize(gold).split()
    if not pt or not gt:
        return float(pt == gt)
    common = Counter(pt) & Counter(gt)
    overlap = sum(common.values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(pt)
    recall = overlap / len(gt)
    return 2 * precision * recall / (precision + recall)


def exact_match(pred: str, golds: list[str]) -> float:
    return float(any(normalize(pred) == normalize(g) for g in golds))


def f1(pred: str, golds: list[str]) -> float:
    return max((_f1_single(pred, g) for g in golds), default=0.0)
```

- [ ] **Step 4: Run tests** — `python -m pytest tests/test_grader.py -q` — Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add src/ca/grader.py tests/test_grader.py
git commit -m "feat: EM/F1 grader with alias support"
```

---

### Task 4: Chat system

**Files:**
- Create: `src/ca/chat.py`
- Test: `tests/test_chat.py`

**Interfaces:**
- Produces: `ChatMessage` dataclass (`mid, sender, recipient, text, round_no`); `ChatSystem` with `send(sender, recipient, text, round_no)->int`, `unread(agent)->list[ChatMessage]`, `mark_read(agent)`, `history(a, b, limit=20)->list[ChatMessage]`.

- [ ] **Step 1: Write the failing tests**

`tests/test_chat.py`:
```python
from ca.chat import ChatSystem


def test_send_unread_mark_read():
    cs = ChatSystem()
    cs.send("a", "b", "hi", 1)
    cs.send("c", "b", "yo", 1)
    cs.send("a", "c", "not for b", 1)
    assert [m.text for m in cs.unread("b")] == ["hi", "yo"]
    cs.mark_read("b")
    assert cs.unread("b") == []
    cs.send("a", "b", "again", 2)
    assert [m.text for m in cs.unread("b")] == ["again"]


def test_history_pairwise():
    cs = ChatSystem()
    cs.send("a", "b", "1", 1)
    cs.send("b", "a", "2", 1)
    cs.send("a", "c", "x", 1)
    assert [m.text for m in cs.history("a", "b")] == ["1", "2"]
```

- [ ] **Step 2: Run to verify failure** — `python -m pytest tests/test_chat.py -q` — ImportError.

- [ ] **Step 3: Implement**

`src/ca/chat.py`:
```python
"""Point-to-point messaging with per-agent unread cursors."""
from collections import defaultdict
from dataclasses import dataclass


@dataclass
class ChatMessage:
    mid: int
    sender: str
    recipient: str
    text: str
    round_no: int


class ChatSystem:
    def __init__(self):
        self.messages: list[ChatMessage] = []
        self._cursor: dict[str, int] = defaultdict(int)  # agent -> index read up to

    def send(self, sender: str, recipient: str, text: str, round_no: int) -> int:
        mid = len(self.messages)
        self.messages.append(ChatMessage(mid, sender, recipient, text, round_no))
        return mid

    def unread(self, agent: str) -> list[ChatMessage]:
        return [m for m in self.messages[self._cursor[agent]:] if m.recipient == agent]

    def mark_read(self, agent: str) -> None:
        self._cursor[agent] = len(self.messages)

    def history(self, a: str, b: str, limit: int = 20) -> list[ChatMessage]:
        pair = [m for m in self.messages
                if {m.sender, m.recipient} == {a, b}]
        return pair[-limit:]
```

- [ ] **Step 4: Run tests** — `python -m pytest tests/test_chat.py -q` — 2 passed.

- [ ] **Step 5: Commit**

```bash
git add src/ca/chat.py tests/test_chat.py
git commit -m "feat: chat system with unread cursors and pairwise history"
```

---

### Task 5: Internal contracts with escrow

**Files:**
- Create: `src/ca/contracts.py`
- Test: `tests/test_contracts.py`

**Interfaces:**
- Consumes: `Ledger` from Task 2.
- Produces: `Contract` dataclass (`cid, proposer, contractor, task, price, status, awaiting, deliverable`); `ContractError`; `ContractSystem(ledger)` with `propose(proposer, contractor, task, price)->Contract`, `counter(agent, cid, price)`, `accept(agent, cid)`, `reject(agent, cid)`, `cancel(agent, cid)`, `deliver(agent, cid, content)->Contract`, `get(cid)->Contract`, `pending_for(agent)->list[Contract]`. Statuses: `proposed/accepted/delivered/rejected/cancelled`. `deliver` atomically releases escrow to contractor and stores the deliverable (caller forwards it to proposer's chat).

- [ ] **Step 1: Write the failing tests**

`tests/test_contracts.py`:
```python
import pytest
from ca.economy import Ledger
from ca.contracts import ContractSystem, ContractError


def setup():
    led = Ledger({"boss": 100, "worker": 10})
    return led, ContractSystem(led)


def test_propose_accept_deliver_flow():
    led, cs = setup()
    c = cs.propose("boss", "worker", "find X", 30)
    assert c.status == "proposed" and c.awaiting == "worker"
    cs.accept("worker", c.cid)
    assert led.balance("boss") == 70 and led.escrow[c.cid] == 30
    done = cs.deliver("worker", c.cid, "X is 42")
    assert done.status == "delivered" and done.deliverable == "X is 42"
    assert led.balance("worker") == 40 and c.cid not in led.escrow
    assert led.conservation_ok()


def test_counter_offer_flips_awaiting():
    led, cs = setup()
    c = cs.propose("boss", "worker", "t", 10)
    cs.counter("worker", c.cid, 20)
    assert c.price == 20 and c.awaiting == "boss"
    cs.accept("boss", c.cid)          # proposer accepts worker's counter
    assert led.escrow[c.cid] == 20    # escrow always from proposer (payer)
    assert led.conservation_ok()


def test_wrong_party_rejected():
    led, cs = setup()
    c = cs.propose("boss", "worker", "t", 10)
    with pytest.raises(ContractError):
        cs.accept("boss", c.cid)       # boss is not the awaited party
    with pytest.raises(ContractError):
        cs.deliver("boss", c.cid, "z")  # not contractor, not accepted


def test_accept_fails_if_payer_broke():
    led, cs = setup()
    c = cs.propose("worker", "boss", "t", 999)  # worker is payer, has 10
    with pytest.raises(ContractError):
        cs.accept("boss", c.cid)
    assert c.status == "proposed"


def test_cancel_refunds_escrow():
    led, cs = setup()
    c = cs.propose("boss", "worker", "t", 30)
    cs.accept("worker", c.cid)
    cs.cancel("boss", c.cid)
    assert c.status == "cancelled" and led.balance("boss") == 100
    assert led.conservation_ok()


def test_pending_for():
    led, cs = setup()
    c1 = cs.propose("boss", "worker", "t1", 5)
    c2 = cs.propose("boss", "worker", "t2", 5)
    cs.accept("worker", c2.cid)
    pend = cs.pending_for("worker")
    assert {p.cid for p in pend} == {c1.cid, c2.cid}  # one awaiting reply, one to deliver
```

- [ ] **Step 2: Run to verify failure** — `python -m pytest tests/test_contracts.py -q` — ImportError.

- [ ] **Step 3: Implement**

`src/ca/contracts.py`:
```python
"""Internal subcontracts: propose/counter/accept(escrow)/deliver(atomic settle)."""
from dataclasses import dataclass, field

from ca.economy import InsufficientFunds, Ledger


class ContractError(Exception):
    pass


@dataclass
class Contract:
    cid: str
    proposer: str      # payer (发包方)
    contractor: str    # worker (承包方)
    task: str
    price: int
    status: str = "proposed"
    awaiting: str = ""          # who must respond while status == proposed
    deliverable: str | None = None


class ContractSystem:
    def __init__(self, ledger: Ledger):
        self.ledger = ledger
        self.contracts: dict[str, Contract] = {}
        self._n = 0

    def _next_id(self) -> str:
        self._n += 1
        return f"c{self._n:04d}"

    def get(self, cid: str) -> Contract:
        if cid not in self.contracts:
            raise ContractError(f"unknown contract {cid}")
        return self.contracts[cid]

    def propose(self, proposer: str, contractor: str, task: str, price: int) -> Contract:
        if price <= 0:
            raise ContractError("price must be positive")
        if proposer == contractor:
            raise ContractError("cannot contract with yourself")
        c = Contract(self._next_id(), proposer, contractor, task, price,
                     awaiting=contractor)
        self.contracts[c.cid] = c
        return c

    def _require(self, c: Contract, status: str, agent: str | None = None):
        if c.status != status:
            raise ContractError(f"{c.cid} is {c.status}, not {status}")
        if agent is not None and c.awaiting != agent:
            raise ContractError(f"{c.cid} awaits {c.awaiting}, not {agent}")

    def counter(self, agent: str, cid: str, price: int) -> Contract:
        c = self.get(cid)
        self._require(c, "proposed", agent)
        if price <= 0:
            raise ContractError("price must be positive")
        c.price = price
        c.awaiting = c.proposer if agent == c.contractor else c.contractor
        return c

    def accept(self, agent: str, cid: str) -> Contract:
        c = self.get(cid)
        self._require(c, "proposed", agent)
        try:
            self.ledger.lock_escrow(c.cid, c.proposer, c.price)
        except InsufficientFunds as e:
            raise ContractError(f"payer cannot fund escrow: {e}") from e
        c.status = "accepted"
        c.awaiting = ""
        return c

    def reject(self, agent: str, cid: str) -> Contract:
        c = self.get(cid)
        self._require(c, "proposed", agent)
        c.status = "rejected"
        c.awaiting = ""
        return c

    def cancel(self, agent: str, cid: str) -> Contract:
        c = self.get(cid)
        if agent not in (c.proposer, c.contractor):
            raise ContractError("not a party to this contract")
        if c.status == "accepted":
            self.ledger.refund_escrow(c.cid, c.proposer)
        elif c.status != "proposed":
            raise ContractError(f"cannot cancel a {c.status} contract")
        c.status = "cancelled"
        c.awaiting = ""
        return c

    def deliver(self, agent: str, cid: str, content: str) -> Contract:
        c = self.get(cid)
        if c.status != "accepted":
            raise ContractError(f"{c.cid} is {c.status}, not accepted")
        if agent != c.contractor:
            raise ContractError("only the contractor may deliver")
        # atomic: payment + deliverable recorded together
        self.ledger.release_escrow(c.cid, c.contractor)
        c.deliverable = content
        c.status = "delivered"
        return c

    def pending_for(self, agent: str) -> list[Contract]:
        out = []
        for c in self.contracts.values():
            if c.status == "proposed" and c.awaiting == agent:
                out.append(c)
            elif c.status == "accepted" and c.contractor == agent:
                out.append(c)
        return out
```

- [ ] **Step 4: Run tests** — `python -m pytest tests/test_contracts.py -q` — 6 passed.

- [ ] **Step 5: Commit**

```bash
git add src/ca/contracts.py tests/test_contracts.py
git commit -m "feat: contract system with escrow and atomic delivery settlement"
```

---

### Task 6: Task board (WORLD contracts)

**Files:**
- Create: `src/ca/taskboard.py`
- Test: `tests/test_taskboard.py`

**Interfaces:**
- Consumes: `Ledger` (Task 2), `f1`/`exact_match` (Task 3).
- Produces: `Question` dataclass (`qid, text, answers, difficulty, price, status, claimed_by, submitted, score, em, payout`); `BoardError`; `TaskBoard(questions: list[Question], ledger)` with `list_open()->list[Question]`, `get(qid)`, `claim(agent, qid)`, `deliver(agent, qid, answer)->tuple[float,int]` (grades F1, mints `round(price*f1)`, closes question, one shot), `all_done()->bool`, `results()->list[Question]`.

- [ ] **Step 1: Write the failing tests**

`tests/test_taskboard.py`:
```python
import pytest
from ca.economy import Ledger
from ca.taskboard import TaskBoard, Question, BoardError


def setup():
    qs = [Question("q0001", "capital of France?", ["Paris"], "easy", 100),
          Question("q0002", "2+2?", ["4", "four"], "easy", 100)]
    led = Ledger({"a": 10, "b": 10})
    return led, TaskBoard(qs, led)


def test_claim_hides_from_open_list():
    led, tb = setup()
    tb.claim("a", "q0001")
    assert [q.qid for q in tb.list_open()] == ["q0002"]
    with pytest.raises(BoardError):
        tb.claim("b", "q0001")


def test_deliver_pays_by_f1_and_closes():
    led, tb = setup()
    tb.claim("a", "q0001")
    score, payout = tb.deliver("a", "q0001", "Paris")
    assert score == 1.0 and payout == 100
    assert led.balance("a") == 110
    assert led.conservation_ok()
    with pytest.raises(BoardError):
        tb.deliver("a", "q0001", "Paris")  # one shot


def test_deliver_requires_claimer():
    led, tb = setup()
    tb.claim("a", "q0001")
    with pytest.raises(BoardError):
        tb.deliver("b", "q0001", "Paris")


def test_wrong_answer_pays_zero_and_all_done():
    led, tb = setup()
    tb.claim("a", "q0001")
    tb.deliver("a", "q0001", "London")
    assert not tb.all_done()
    tb.claim("a", "q0002")
    tb.deliver("a", "q0002", "4")
    assert tb.all_done()
```

- [ ] **Step 2: Run to verify failure** — `python -m pytest tests/test_taskboard.py -q` — ImportError.

- [ ] **Step 3: Implement**

`src/ca/taskboard.py`:
```python
"""WORLD's task board: open questions are external contracts at posted prices."""
from dataclasses import dataclass, field

from ca.economy import Ledger
from ca.grader import exact_match, f1


class BoardError(Exception):
    pass


@dataclass
class Question:
    qid: str
    text: str
    answers: list[str]
    difficulty: str
    price: int
    status: str = "open"            # open / claimed / closed
    claimed_by: str | None = None
    submitted: str | None = None
    score: float = 0.0              # F1
    em: float = 0.0
    payout: int = 0


class TaskBoard:
    def __init__(self, questions: list[Question], ledger: Ledger):
        self.questions: dict[str, Question] = {q.qid: q for q in questions}
        self.ledger = ledger

    def get(self, qid: str) -> Question:
        if qid not in self.questions:
            raise BoardError(f"unknown question {qid}")
        return self.questions[qid]

    def list_open(self) -> list[Question]:
        return [q for q in self.questions.values() if q.status == "open"]

    def claim(self, agent: str, qid: str) -> Question:
        q = self.get(qid)
        if q.status != "open":
            raise BoardError(f"{qid} is {q.status}")
        q.status = "claimed"
        q.claimed_by = agent
        return q

    def deliver(self, agent: str, qid: str, answer: str) -> tuple[float, int]:
        q = self.get(qid)
        if q.status != "claimed":
            raise BoardError(f"{qid} is {q.status}, claim it first")
        if q.claimed_by != agent:
            raise BoardError(f"{qid} was claimed by {q.claimed_by}")
        q.submitted = answer
        q.score = f1(answer, q.answers)
        q.em = exact_match(answer, q.answers)
        q.payout = round(q.price * q.score)
        q.status = "closed"
        if q.payout > 0:
            self.ledger.mint(agent, q.payout)
        return q.score, q.payout

    def all_done(self) -> bool:
        return all(q.status == "closed" for q in self.questions.values())

    def results(self) -> list[Question]:
        return list(self.questions.values())
```

- [ ] **Step 4: Run tests** — `python -m pytest tests/test_taskboard.py -q` — 4 passed.

- [ ] **Step 5: Commit**

```bash
git add src/ca/taskboard.py tests/test_taskboard.py
git commit -m "feat: task board with claim/deliver, F1-scaled payouts, one-shot close"
```

---

### Task 7: Agent memories (FIFO, goal stack, long-term)

**Files:**
- Create: `src/ca/memory.py`
- Test: `tests/test_memory.py`

**Interfaces:**
- Produces: `FifoMemory(k)` with `add(action, result)`, `render()->str`; `GoalStack(root)` with `push(note)`, `pop()->str` (root not poppable), `render()->str`; `LongTermMemory` (all agents) with `write(agent, content)`, `search(agent, query, k=3)->list[str]`.

- [ ] **Step 1: Write the failing tests**

`tests/test_memory.py`:
```python
import pytest
from ca.memory import FifoMemory, GoalStack, LongTermMemory


def test_fifo_rolls_over():
    m = FifoMemory(k=2)
    m.add("a1", "r1"); m.add("a2", "r2"); m.add("a3", "r3")
    out = m.render()
    assert "a1" not in out and "a2" in out and "a3" in out


def test_goal_stack_root_protected():
    g = GoalStack("maximize tokens")
    g.push("do q1")
    assert g.pop() == "do q1"
    with pytest.raises(IndexError):
        g.pop()
    assert "maximize tokens" in g.render()


def test_ltm_scoped_and_ranked():
    ltm = LongTermMemory()
    ltm.write("a", "paris is the capital of france")
    ltm.write("a", "tokyo is in japan")
    ltm.write("b", "secret of b")
    hits = ltm.search("a", "capital france", k=1)
    assert hits == ["paris is the capital of france"]
    assert ltm.search("b", "capital", k=3) == []  # no overlap for b
```

- [ ] **Step 2: Run to verify failure** — `python -m pytest tests/test_memory.py -q` — ImportError.

- [ ] **Step 3: Implement**

`src/ca/memory.py`:
```python
"""Per-agent short-term (FIFO, goal stack) and long-term memory."""
from collections import defaultdict, deque


class FifoMemory:
    def __init__(self, k: int = 10):
        self.items: deque[tuple[str, str]] = deque(maxlen=k)

    def add(self, action: str, result: str) -> None:
        self.items.append((action, result))

    def render(self) -> str:
        if not self.items:
            return "(no recent actions)"
        return "\n".join(f"- {a} -> {r}" for a, r in self.items)


class GoalStack:
    def __init__(self, root: str):
        self._root = root
        self._stack: list[str] = []

    def push(self, note: str) -> None:
        self._stack.append(note)

    def pop(self) -> str:
        if not self._stack:
            raise IndexError("cannot pop the root goal")
        return self._stack.pop()

    def render(self) -> str:
        lines = [f"[0] {self._root} (root, permanent)"]
        lines += [f"[{i+1}] {n}" for i, n in enumerate(self._stack)]
        lines[-1] += "   <- current focus"
        return "\n".join(lines)


class LongTermMemory:
    def __init__(self):
        self._store: dict[str, list[str]] = defaultdict(list)

    def write(self, agent: str, content: str) -> None:
        self._store[agent].append(content)

    def search(self, agent: str, query: str, k: int = 3) -> list[str]:
        qtok = set(query.lower().split())
        scored = []
        for entry in self._store[agent]:
            overlap = len(qtok & set(entry.lower().split()))
            if overlap > 0:
                scored.append((overlap, entry))
        scored.sort(key=lambda t: -t[0])
        return [e for _, e in scored[:k]]
```

- [ ] **Step 4: Run tests** — `python -m pytest tests/test_memory.py -q` — 3 passed.

- [ ] **Step 5: Commit**

```bash
git add src/ca/memory.py tests/test_memory.py
git commit -m "feat: FIFO memory, goal stack, keyword long-term memory"
```

---

### Task 8: Retrieval backend

**Files:**
- Create: `src/ca/retrieval.py`
- Test: `tests/test_retrieval.py`

**Interfaces:**
- Produces: `Doc` TypedDict-ish dict `{"title": str, "text": str}`; `Bm25Backend(docs: list[dict])` with `search(query, k=5)->list[dict]`; `Bm25Backend.load(index_dir)` classmethod (mmap-load a saved index + docs). Test uses in-memory construction only.

- [ ] **Step 1: Write the failing tests**

`tests/test_retrieval.py`:
```python
from ca.retrieval import Bm25Backend

DOCS = [
    {"title": "Paris", "text": "Paris is the capital and largest city of France."},
    {"title": "Tokyo", "text": "Tokyo is the capital of Japan."},
    {"title": "Berlin", "text": "Berlin is the capital of Germany."},
]


def test_search_ranks_relevant_doc_first():
    r = Bm25Backend(DOCS)
    hits = r.search("capital of France", k=2)
    assert hits[0]["title"] == "Paris"
    assert len(hits) == 2


def test_k_larger_than_corpus():
    r = Bm25Backend(DOCS)
    assert len(r.search("capital", k=10)) == 3
```

- [ ] **Step 2: Run to verify failure** — `python -m pytest tests/test_retrieval.py -q` — ImportError.

- [ ] **Step 3: Implement**

`src/ca/retrieval.py`:
```python
"""Pluggable retrieval; default backend = bm25s over a paragraph corpus."""
import json
from pathlib import Path

import bm25s


class Bm25Backend:
    def __init__(self, docs: list[dict], retriever: "bm25s.BM25 | None" = None):
        self.docs = docs
        if retriever is None:
            retriever = bm25s.BM25()
            retriever.index(bm25s.tokenize([d["text"] for d in docs], show_progress=False))
        self.retriever = retriever

    def search(self, query: str, k: int = 5) -> list[dict]:
        k = min(k, len(self.docs))
        idx, _scores = self.retriever.retrieve(
            bm25s.tokenize([query], show_progress=False), k=k, show_progress=False
        )
        return [self.docs[int(i)] for i in idx[0]]

    def save(self, index_dir: str) -> None:
        p = Path(index_dir)
        p.mkdir(parents=True, exist_ok=True)
        self.retriever.save(str(p / "bm25"))
        with open(p / "docs.jsonl", "w") as f:
            for d in self.docs:
                f.write(json.dumps(d, ensure_ascii=False) + "\n")

    @classmethod
    def load(cls, index_dir: str) -> "Bm25Backend":
        p = Path(index_dir)
        docs = [json.loads(line) for line in open(p / "docs.jsonl")]
        retriever = bm25s.BM25.load(str(p / "bm25"), mmap=True)
        return cls(docs, retriever=retriever)
```

Note: if the `bm25s` API surface differs in the installed version (e.g. tokenize kwargs), fix against the actual library error output — the test corpus is tiny so iteration is fast. Do not switch libraries.

- [ ] **Step 4: Run tests** — `pip install bm25s && python -m pytest tests/test_retrieval.py -q` — 2 passed.

- [ ] **Step 5: Commit**

```bash
git add src/ca/retrieval.py tests/test_retrieval.py
git commit -m "feat: bm25s retrieval backend with save/load"
```

---

### Task 9: Level configs + Infra aggregate

**Files:**
- Create: `src/ca/config.py`, `src/ca/infra.py`
- Test: `tests/test_config.py`

**Interfaces:**
- Produces:
  - `LevelConfig` frozen dataclass: `level, n_agents, has_interface, world_access ("all"|"interface"), retrieve_access ("all"|"interface"), interface_no_counter (bool), star_comms (bool)`; dict `LEVELS` with keys `L0..L5` per the spec §9 matrix.
  - `ExperimentConfig` dataclass: `level: LevelConfig, seed: int, seed_capital_total: int, fifo_k: int = 10, max_rounds: int = 60, model: str = "claude-haiku-4-5", max_tokens_per_turn: int = 1024`.
  - `agent_ids(level)->list[str]`: L0 → `agent_1..agent_8`; L1–L4 → `["interface", "agent_1"..."agent_7"]`; L5 → `["agent_1"]`.
  - `Infra(cfg, questions, retriever)` holding `.ledger .chat .contracts .board .ltm .retriever .scratchpads (dict[agent][task_id]->list[str]) .round .agent_ids .cfg` — seed capital split equally (remainder to first agent).

- [ ] **Step 1: Write the failing tests**

`tests/test_config.py`:
```python
from ca.config import LEVELS, ExperimentConfig, agent_ids
from ca.infra import Infra
from ca.taskboard import Question


def test_level_matrix_matches_spec():
    assert LEVELS["L0"].world_access == "all" and not LEVELS["L0"].has_interface
    assert LEVELS["L1"].world_access == "interface" and LEVELS["L1"].retrieve_access == "all"
    assert LEVELS["L2"].retrieve_access == "interface" and not LEVELS["L2"].interface_no_counter
    assert LEVELS["L3"].interface_no_counter and not LEVELS["L3"].star_comms
    assert LEVELS["L4"].star_comms and LEVELS["L4"].interface_no_counter
    assert LEVELS["L5"].n_agents == 1


def test_agent_ids():
    assert agent_ids(LEVELS["L0"]) == [f"agent_{i}" for i in range(1, 9)]
    ids = agent_ids(LEVELS["L1"])
    assert ids[0] == "interface" and len(ids) == 8
    assert agent_ids(LEVELS["L5"]) == ["agent_1"]


def test_infra_splits_seed_capital():
    cfg = ExperimentConfig(level=LEVELS["L0"], seed=0, seed_capital_total=801)
    infra = Infra(cfg, [Question("q0001", "?", ["x"], "easy", 10)], retriever=None)
    balances = [infra.ledger.balance(a) for a in infra.agent_ids]
    assert sum(balances) == 801 and max(balances) - min(balances) <= 1
    assert infra.ledger.conservation_ok()
```

- [ ] **Step 2: Run to verify failure** — `python -m pytest tests/test_config.py -q` — ImportError.

- [ ] **Step 3: Implement**

`src/ca/config.py`:
```python
"""Centralization levels L0-L5. Levels differ ONLY through these fields."""
from dataclasses import dataclass, field


@dataclass(frozen=True)
class LevelConfig:
    level: str
    n_agents: int
    has_interface: bool
    world_access: str        # "all" | "interface"  (list/claim/deliver to WORLD)
    retrieve_access: str     # "all" | "interface"
    interface_no_counter: bool  # contracts proposed BY interface: no counter_offer
    star_comms: bool         # non-interface agents may only interact with interface


LEVELS: dict[str, LevelConfig] = {
    "L0": LevelConfig("L0", 8, False, "all", "all", False, False),
    "L1": LevelConfig("L1", 8, True, "interface", "all", False, False),
    "L2": LevelConfig("L2", 8, True, "interface", "interface", False, False),
    "L3": LevelConfig("L3", 8, True, "interface", "interface", True, False),
    "L4": LevelConfig("L4", 8, True, "interface", "interface", True, True),
    "L5": LevelConfig("L5", 1, False, "all", "all", False, False),
}


def agent_ids(level: LevelConfig) -> list[str]:
    if level.has_interface:
        return ["interface"] + [f"agent_{i}" for i in range(1, level.n_agents)]
    return [f"agent_{i}" for i in range(1, level.n_agents + 1)]


@dataclass
class ExperimentConfig:
    level: LevelConfig
    seed: int
    seed_capital_total: int
    fifo_k: int = 10
    max_rounds: int = 60
    model: str = "claude-haiku-4-5"
    max_tokens_per_turn: int = 1024
```

`src/ca/infra.py`:
```python
"""Aggregate of all authoritative world state."""
from collections import defaultdict

from ca.chat import ChatSystem
from ca.config import ExperimentConfig, agent_ids
from ca.contracts import ContractSystem
from ca.economy import Ledger
from ca.memory import LongTermMemory
from ca.taskboard import Question, TaskBoard


class Infra:
    def __init__(self, cfg: ExperimentConfig, questions: list[Question], retriever):
        self.cfg = cfg
        self.agent_ids = agent_ids(cfg.level)
        n = len(self.agent_ids)
        base, rem = divmod(cfg.seed_capital_total, n)
        seed_capital = {a: base for a in self.agent_ids}
        seed_capital[self.agent_ids[0]] += rem
        self.ledger = Ledger(seed_capital)
        self.chat = ChatSystem()
        self.contracts = ContractSystem(self.ledger)
        self.board = TaskBoard(questions, self.ledger)
        self.ltm = LongTermMemory()
        self.retriever = retriever
        self.scratchpads: dict[str, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
        self.round = 0
```

- [ ] **Step 4: Run tests** — `python -m pytest tests/test_config.py -q` — 3 passed.

- [ ] **Step 5: Commit**

```bash
git add src/ca/config.py src/ca/infra.py tests/test_config.py
git commit -m "feat: L0-L5 level configs and Infra aggregate"
```

---

### Task 10: Action registry, handlers, permission gating

**Files:**
- Create: `src/ca/actions.py`
- Test: `tests/test_actions.py`

**Interfaces:**
- Consumes: everything from Tasks 2–9.
- Produces:
  - `ACTION_SPECS: dict[str, dict]` — name → `{description, input_schema}` (Anthropic tool format, `input_schema` with `type:"object"`, `properties`, `required`).
  - `is_billable(name, inp)->bool` — `retrieve`, `work_on`, or `deliver_work` whose `target_id` starts with `"q"`.
  - `visible_tools(cfg_level, agent_id)->list[dict]` — Anthropic `tools` list, hiding actions the agent can never use at this level (world actions and `retrieve` for non-interface when gated).
  - `permission_error(infra, agent_id, name, inp)->str|None` — returns error string for statically forbidden calls (world/retrieve gating, star-comms target check, bankruptcy on billable, `counter_offer` on interface-proposed contracts at L3+).
  - `dispatch(infra, agent_id, name, inp)->str` — executes and returns result string; catches domain exceptions into `"ERROR: ..."` strings.

- [ ] **Step 1: Write the failing tests**

`tests/test_actions.py`:
```python
import pytest
from ca.actions import dispatch, is_billable, permission_error, visible_tools, ACTION_SPECS
from ca.config import LEVELS, ExperimentConfig
from ca.infra import Infra
from ca.retrieval import Bm25Backend
from ca.taskboard import Question

DOCS = [{"title": "Paris", "text": "Paris is the capital of France."}]


def make(level="L0", capital=1000):
    cfg = ExperimentConfig(level=LEVELS[level], seed=0, seed_capital_total=capital)
    qs = [Question("q0001", "capital of France?", ["Paris"], "easy", 100)]
    return Infra(cfg, qs, retriever=Bm25Backend(DOCS))


def test_billability():
    assert is_billable("retrieve", {"query": "x"})
    assert is_billable("work_on", {"task_id": "q0001", "thought": "t"})
    assert is_billable("deliver_work", {"target_id": "q0001", "content": "Paris"})
    assert not is_billable("deliver_work", {"target_id": "c0001", "content": "x"})
    assert not is_billable("send_message", {"to": "a", "text": "x"})


def test_world_gating_by_level():
    i0 = make("L0")
    assert permission_error(i0, "agent_1", "claim_question", {"qid": "q0001"}) is None
    i1 = make("L1")
    assert permission_error(i1, "agent_1", "claim_question", {"qid": "q0001"}) is not None
    assert permission_error(i1, "interface", "claim_question", {"qid": "q0001"}) is None


def test_retrieve_gating_and_star_comms():
    i2 = make("L2")
    assert permission_error(i2, "agent_1", "retrieve", {"query": "x"}) is not None
    assert permission_error(i2, "interface", "retrieve", {"query": "x"}) is None
    i4 = make("L4")
    assert permission_error(i4, "agent_1", "send_message", {"to": "agent_2", "text": "hi"}) is not None
    assert permission_error(i4, "agent_1", "send_message", {"to": "interface", "text": "hi"}) is None
    assert permission_error(i4, "interface", "send_message", {"to": "agent_2", "text": "hi"}) is None


def test_no_counter_on_interface_contracts_at_L3():
    i3 = make("L3")
    c = i3.contracts.propose("interface", "agent_1", "solve q0001", 50)
    err = permission_error(i3, "agent_1", "counter_offer", {"contract_id": c.cid, "price": 80})
    assert err is not None
    # agent-to-agent contracts still counterable at L3
    c2 = i3.contracts.propose("agent_1", "agent_2", "sub", 10)
    assert permission_error(i3, "agent_2", "counter_offer", {"contract_id": c2.cid, "price": 20}) is None


def test_bankrupt_blocks_billable_only():
    i0 = make("L0", capital=8)  # 1 token each
    i0.ledger.burn("agent_1", 5)
    assert permission_error(i0, "agent_1", "retrieve", {"query": "x"}) is not None
    assert permission_error(i0, "agent_1", "send_message", {"to": "agent_2", "text": "s"}) is None


def test_dispatch_full_answer_flow():
    i0 = make("L0")
    out = dispatch(i0, "agent_1", "list_questions", {})
    assert "q0001" in out
    dispatch(i0, "agent_1", "claim_question", {"qid": "q0001"})
    out = dispatch(i0, "agent_1", "retrieve", {"query": "capital of France"})
    assert "Paris" in out
    dispatch(i0, "agent_1", "work_on", {"task_id": "q0001", "thought": "answer is Paris"})
    out = dispatch(i0, "agent_1", "deliver_work", {"target_id": "q0001", "content": "Paris"})
    assert "100" in out  # payout mentioned
    assert i0.ledger.balance("agent_1") > 125  # 125 seed + 100 payout
    assert i0.ledger.conservation_ok()


def test_dispatch_contract_flow_delivers_to_chat():
    i0 = make("L0")
    dispatch(i0, "agent_1", "propose_contract", {"to": "agent_2", "task": "find capital", "price": 30})
    dispatch(i0, "agent_2", "accept_contract", {"contract_id": "c0001"})
    dispatch(i0, "agent_2", "deliver_work", {"target_id": "c0001", "content": "it is Paris"})
    unread = i0.chat.unread("agent_1")
    assert any("Paris" in m.text for m in unread)
    assert i0.ledger.conservation_ok()


def test_dispatch_error_string_not_exception():
    i0 = make("L0")
    out = dispatch(i0, "agent_1", "claim_question", {"qid": "q9999"})
    assert out.startswith("ERROR")


def test_visible_tools_filtered():
    names_l0 = {t["name"] for t in visible_tools(LEVELS["L0"], "agent_1")}
    assert "claim_question" in names_l0 and "retrieve" in names_l0
    names_l2 = {t["name"] for t in visible_tools(LEVELS["L2"], "agent_1")}
    assert "claim_question" not in names_l2 and "retrieve" not in names_l2
    names_l2i = {t["name"] for t in visible_tools(LEVELS["L2"], "interface")}
    assert "claim_question" in names_l2i and "retrieve" in names_l2i
    assert set(ACTION_SPECS) >= names_l0
```

- [ ] **Step 2: Run to verify failure** — `python -m pytest tests/test_actions.py -q` — ImportError.

- [ ] **Step 3: Implement**

`src/ca/actions.py`:
```python
"""Action registry: specs (tool schemas), permission gating, dispatch."""
from ca.config import LevelConfig
from ca.contracts import ContractError
from ca.infra import Infra
from ca.taskboard import BoardError


def _schema(props: dict, required: list[str]) -> dict:
    return {"type": "object", "properties": props, "required": required}


_S = {"type": "string"}
_I = {"type": "integer"}

ACTION_SPECS: dict[str, dict] = {
    # -------- billable (answer-related) --------
    "retrieve": {
        "description": "Search the external knowledge corpus. COSTS TOKENS.",
        "input_schema": _schema({"query": _S}, ["query"]),
    },
    "work_on": {
        "description": "Record one reasoning step about a task in your private scratchpad. COSTS TOKENS.",
        "input_schema": _schema({"task_id": _S, "thought": _S}, ["task_id", "thought"]),
    },
    "deliver_work": {
        "description": ("Deliver work. target_id starting with 'q' = submit final answer to the WORLD "
                        "(graded, paid by quality, one shot, COSTS TOKENS). target_id starting with 'c' "
                        "= deliver an accepted contract (escrow released to you, free)."),
        "input_schema": _schema({"target_id": _S, "content": _S}, ["target_id", "content"]),
    },
    # -------- free (coordination) --------
    "list_questions": {
        "description": "List open questions on the task board with prices.",
        "input_schema": _schema({}, []),
    },
    "claim_question": {
        "description": "Exclusively claim an open question (others can no longer see it).",
        "input_schema": _schema({"qid": _S}, ["qid"]),
    },
    "send_message": {
        "description": "Send a chat message to another agent.",
        "input_schema": _schema({"to": _S, "text": _S}, ["to", "text"]),
    },
    "read_chat": {
        "description": "Read recent chat history with another agent.",
        "input_schema": _schema({"with_agent": _S}, ["with_agent"]),
    },
    "propose_contract": {
        "description": "Offer to PAY another agent `price` tokens to do `task` for you.",
        "input_schema": _schema({"to": _S, "task": _S, "price": _I}, ["to", "task", "price"]),
    },
    "accept_contract": {
        "description": "Accept a contract offer (price is locked in escrow from the payer).",
        "input_schema": _schema({"contract_id": _S}, ["contract_id"]),
    },
    "reject_contract": {
        "description": "Reject a contract offer.",
        "input_schema": _schema({"contract_id": _S}, ["contract_id"]),
    },
    "counter_offer": {
        "description": "Counter a pending contract with a new price.",
        "input_schema": _schema({"contract_id": _S, "price": _I}, ["contract_id", "price"]),
    },
    "cancel_contract": {
        "description": "Cancel a contract you are party to (escrow refunded to payer).",
        "input_schema": _schema({"contract_id": _S}, ["contract_id"]),
    },
    "pay": {
        "description": "Freely transfer tokens to another agent (tips, deposits, aid).",
        "input_schema": _schema({"to": _S, "amount": _I}, ["to", "amount"]),
    },
    "push_goal": {
        "description": "Push a sub-goal note onto your goal stack.",
        "input_schema": _schema({"note": _S}, ["note"]),
    },
    "pop_goal": {
        "description": "Pop the top goal off your goal stack (root goal cannot be popped).",
        "input_schema": _schema({}, []),
    },
    "memory_write": {
        "description": "Save a note to your private long-term memory.",
        "input_schema": _schema({"content": _S}, ["content"]),
    },
    "memory_search": {
        "description": "Search your private long-term memory.",
        "input_schema": _schema({"query": _S}, ["query"]),
    },
    "check_balance": {
        "description": "Check your current token balance.",
        "input_schema": _schema({}, []),
    },
    "list_agents": {
        "description": "List all agents in the system.",
        "input_schema": _schema({}, []),
    },
}

_WORLD_ACTIONS = {"list_questions", "claim_question"}
_TARGETED = {"send_message", "propose_contract", "pay"}  # star-comms checked actions


def is_billable(name: str, inp: dict) -> bool:
    if name in ("retrieve", "work_on"):
        return True
    if name == "deliver_work":
        return str(inp.get("target_id", "")).startswith("q")
    return False


def visible_tools(level: LevelConfig, agent_id: str) -> list[dict]:
    is_iface = agent_id == "interface"
    out = []
    for name, spec in ACTION_SPECS.items():
        if level.world_access == "interface" and not is_iface and name in _WORLD_ACTIONS:
            continue
        if level.retrieve_access == "interface" and not is_iface and name == "retrieve":
            continue
        out.append({"name": name, **spec})
    return out


def permission_error(infra: Infra, agent_id: str, name: str, inp: dict) -> str | None:
    level = infra.cfg.level
    is_iface = agent_id == "interface"
    # world access (incl. deliver to WORLD)
    world_call = name in _WORLD_ACTIONS or (
        name == "deliver_work" and str(inp.get("target_id", "")).startswith("q"))
    if world_call and level.world_access == "interface" and not is_iface:
        return "only the interface agent may interact with the task board"
    if name == "retrieve" and level.retrieve_access == "interface" and not is_iface:
        return "only the interface agent may retrieve external information"
    # star comms
    if level.star_comms and not is_iface:
        if name in _TARGETED and inp.get("to") != "interface":
            return "at this configuration you may only interact with the interface agent"
        if name == "read_chat" and inp.get("with_agent") != "interface":
            return "at this configuration you may only interact with the interface agent"
    # pricing centralization: no countering interface-proposed contracts
    if name == "counter_offer" and level.interface_no_counter:
        try:
            c = infra.contracts.get(str(inp.get("contract_id", "")))
        except ContractError:
            return None  # let dispatch produce the unknown-contract error
        if c.proposer == "interface":
            return "the interface agent's contract prices are non-negotiable: accept or reject"
    # bankruptcy freezes billable actions
    if is_billable(name, inp) and infra.ledger.is_bankrupt(agent_id):
        return "you are bankrupt (balance <= 0): answer-related actions are frozen"
    return None


def dispatch(infra: Infra, agent_id: str, name: str, inp: dict) -> str:
    try:
        return _HANDLERS[name](infra, agent_id, inp)
    except (BoardError, ContractError, KeyError, ValueError, IndexError) as e:
        return f"ERROR: {e}"


# ---------------- handlers ----------------

def _h_retrieve(infra, a, inp):
    hits = infra.retriever.search(inp["query"], k=5)
    return "\n\n".join(f"[{d['title']}] {d['text']}" for d in hits) or "(no results)"


def _h_work_on(infra, a, inp):
    infra.scratchpads[a][inp["task_id"]].append(inp["thought"])
    return f"noted on scratchpad for {inp['task_id']} ({len(infra.scratchpads[a][inp['task_id']])} entries)"


def _h_deliver_work(infra, a, inp):
    tid, content = inp["target_id"], inp["content"]
    if tid.startswith("q"):
        score, payout = infra.board.deliver(a, tid, content)
        return f"answer to {tid} graded: F1={score:.2f}, paid {payout} tokens"
    c = infra.contracts.deliver(a, tid, content)
    infra.chat.send(a, c.proposer, f"[deliverable for {c.cid}] {content}", infra.round)
    return f"delivered {c.cid}; escrow of {c.price} tokens released to you"


def _h_list_questions(infra, a, inp):
    qs = infra.board.list_open()
    if not qs:
        return "(no open questions)"
    return "\n".join(f"{q.qid} [{q.difficulty}, reward {q.price}]: {q.text}" for q in qs)


def _h_claim_question(infra, a, inp):
    q = infra.board.claim(a, inp["qid"])
    return f"claimed {q.qid}: {q.text} (reward up to {q.price})"


def _h_send_message(infra, a, inp):
    if inp["to"] not in infra.agent_ids:
        return f"ERROR: unknown agent {inp['to']}"
    infra.chat.send(a, inp["to"], inp["text"], infra.round)
    return f"sent to {inp['to']}"


def _h_read_chat(infra, a, inp):
    msgs = infra.chat.history(a, inp["with_agent"])
    return "\n".join(f"[r{m.round_no}] {m.sender}: {m.text}" for m in msgs) or "(no history)"


def _h_propose_contract(infra, a, inp):
    if inp["to"] not in infra.agent_ids:
        return f"ERROR: unknown agent {inp['to']}"
    c = infra.contracts.propose(a, inp["to"], inp["task"], int(inp["price"]))
    infra.chat.send(a, inp["to"],
                    f"[contract offer {c.cid}] task: {c.task} | price: {c.price}", infra.round)
    return f"proposed {c.cid} to {inp['to']} at {c.price}"


def _h_accept_contract(infra, a, inp):
    c = infra.contracts.accept(a, inp["contract_id"])
    other = c.proposer if a == c.contractor else c.contractor
    infra.chat.send(a, other, f"[{c.cid} accepted] price {c.price} in escrow", infra.round)
    return f"accepted {c.cid}; {c.price} locked in escrow from {c.proposer}"


def _h_reject_contract(infra, a, inp):
    c = infra.contracts.reject(a, inp["contract_id"])
    other = c.proposer if a != c.proposer else c.contractor
    infra.chat.send(a, other, f"[{c.cid} rejected]", infra.round)
    return f"rejected {c.cid}"


def _h_counter_offer(infra, a, inp):
    c = infra.contracts.counter(a, inp["contract_id"], int(inp["price"]))
    infra.chat.send(a, c.awaiting, f"[{c.cid} counter-offer] new price: {c.price}", infra.round)
    return f"countered {c.cid} at {c.price}; awaiting {c.awaiting}"


def _h_cancel_contract(infra, a, inp):
    c = infra.contracts.cancel(a, inp["contract_id"])
    other = c.proposer if a != c.proposer else c.contractor
    infra.chat.send(a, other, f"[{c.cid} cancelled]", infra.round)
    return f"cancelled {c.cid}"


def _h_pay(infra, a, inp):
    infra.ledger.transfer(a, inp["to"], int(inp["amount"]))
    infra.chat.send(a, inp["to"], f"[payment] {inp['amount']} tokens", infra.round)
    return f"paid {inp['amount']} to {inp['to']}"


def _h_push_goal(infra, a, inp):
    return "__PUSH_GOAL__"  # handled by Agent (owns the stack); see agent.py


def _h_pop_goal(infra, a, inp):
    return "__POP_GOAL__"


def _h_memory_write(infra, a, inp):
    infra.ltm.write(a, inp["content"])
    return "saved to long-term memory"


def _h_memory_search(infra, a, inp):
    hits = infra.ltm.search(a, inp["query"])
    return "\n".join(f"- {h}" for h in hits) or "(no matching memories)"


def _h_check_balance(infra, a, inp):
    return f"balance: {infra.ledger.balance(a)} tokens"


def _h_list_agents(infra, a, inp):
    return ", ".join(infra.agent_ids)


_HANDLERS = {
    "retrieve": _h_retrieve, "work_on": _h_work_on, "deliver_work": _h_deliver_work,
    "list_questions": _h_list_questions, "claim_question": _h_claim_question,
    "send_message": _h_send_message, "read_chat": _h_read_chat,
    "propose_contract": _h_propose_contract, "accept_contract": _h_accept_contract,
    "reject_contract": _h_reject_contract, "counter_offer": _h_counter_offer,
    "cancel_contract": _h_cancel_contract, "pay": _h_pay,
    "push_goal": _h_push_goal, "pop_goal": _h_pop_goal,
    "memory_write": _h_memory_write, "memory_search": _h_memory_search,
    "check_balance": _h_check_balance, "list_agents": _h_list_agents,
}
```

- [ ] **Step 4: Run tests** — `python -m pytest tests/test_actions.py -q` — 9 passed. Also run the full suite: `python -m pytest -q`.

- [ ] **Step 5: Commit**

```bash
git add src/ca/actions.py tests/test_actions.py
git commit -m "feat: action registry with billing rule, level gating, dispatch"
```

---

### Task 11: Context rendering (system prompt + per-turn view)

**Files:**
- Create: `src/ca/context.py`
- Test: `tests/test_context.py`

**Interfaces:**
- Consumes: `Infra`, `FifoMemory`, `GoalStack`.
- Produces: `system_prompt(cfg_level, agent_id, all_agent_ids)->str` (stable per agent per run); `render_turn(infra, agent_id, fifo, goals)->str` (balance, full goal stack, pending contracts, unread messages, FIFO). `render_turn` does NOT mark messages read — the Agent does that after a successful render (Task 12).

- [ ] **Step 1: Write the failing tests**

`tests/test_context.py`:
```python
from ca.config import LEVELS, ExperimentConfig
from ca.context import render_turn, system_prompt
from ca.infra import Infra
from ca.memory import FifoMemory, GoalStack
from ca.taskboard import Question


def make(level="L1"):
    cfg = ExperimentConfig(level=LEVELS[level], seed=0, seed_capital_total=800)
    return Infra(cfg, [Question("q0001", "?", ["x"], "easy", 100)], retriever=None)


def test_system_prompt_mentions_identity_goal_and_rules():
    infra = make("L4")
    sp = system_prompt(infra.cfg.level, "agent_1", infra.agent_ids)
    assert "agent_1" in sp
    assert "maximize" in sp.lower()
    assert "interface" in sp  # star-comms rule explained
    sp_i = system_prompt(infra.cfg.level, "interface", infra.agent_ids)
    assert "you are the interface" in sp_i.lower()


def test_render_turn_contains_state():
    infra = make("L0")
    infra.chat.send("agent_2", "agent_1", "hello there", 1)
    infra.contracts.propose("agent_2", "agent_1", "subtask", 20)
    fifo, goals = FifoMemory(3), GoalStack("maximize token balance")
    goals.push("finish q0001")
    fifo.add("check_balance", "balance: 100")
    out = render_turn(infra, "agent_1", fifo, goals)
    assert "100" in out            # balance
    assert "finish q0001" in out   # goal stack
    assert "hello there" in out    # unread
    assert "c0001" in out          # pending contract
    assert "check_balance" in out  # fifo
    # render must NOT consume unread
    assert infra.chat.unread("agent_1")
```

- [ ] **Step 2: Run to verify failure** — `python -m pytest tests/test_context.py -q` — ImportError.

- [ ] **Step 3: Implement**

`src/ca/context.py`:
```python
"""Builds the LLM context: stable system prompt + per-turn dynamic view."""
from ca.config import LevelConfig
from ca.infra import Infra
from ca.memory import FifoMemory, GoalStack

_BASE = """You are {agent_id}, an autonomous agent in a multi-agent economy.
Agents in the system: {peers}.

YOUR PERMANENT ROOT GOAL: maximize your token balance.
Tokens are both your money and your fuel: answer-related actions (retrieve,
work_on, and delivering answers to the WORLD) consume tokens equal to the LLM
cost of that turn. Coordination actions (chat, contracts, payments, memory,
goals) are free. If your balance drops to 0 or below you are BANKRUPT and can
no longer perform answer-related actions.

You earn tokens ONLY from: (a) delivering correct answers to the WORLD's
questions (paid = price x answer quality F1, one attempt per question), or
(b) payments from other agents (contract escrow settlements or transfers).

Each turn you must choose EXACTLY ONE action (tool call). Think about
profitability: estimate what a question will cost to answer vs its reward.
You may subcontract work to other agents via contracts and negotiate prices.
{level_rules}"""

_INTERFACE_EXTRA = """
YOU ARE THE INTERFACE AGENT: the only agent allowed to take questions from
the task board and deliver answers to the WORLD. Other agents can work for
you via contracts. Your profit = WORLD rewards minus what you pay them."""


def _level_rules(level: LevelConfig, is_iface: bool) -> str:
    rules = []
    if level.world_access == "interface":
        rules.append("Only the interface agent can list/claim questions and deliver answers to the WORLD.")
    if level.retrieve_access == "interface":
        rules.append("Only the interface agent can retrieve external information; others must ask it via chat/contracts.")
    if level.interface_no_counter:
        rules.append("Contract prices set by the interface agent are FINAL (no counter-offers on its contracts).")
    if level.star_comms:
        rules.append("You may only message/contract/pay the interface agent."
                     if not is_iface else
                     "Other agents can only interact with you, not with each other.")
    if not rules:
        return "\nThis is a fully decentralized configuration: every agent has equal access to everything."
    return "\nConfiguration rules:\n" + "\n".join(f"- {r}" for r in rules)


def system_prompt(level: LevelConfig, agent_id: str, all_ids: list[str]) -> str:
    is_iface = agent_id == "interface"
    sp = _BASE.format(agent_id=agent_id,
                      peers=", ".join(all_ids),
                      level_rules=_level_rules(level, is_iface))
    if is_iface:
        sp += _INTERFACE_EXTRA
    return sp


def render_turn(infra: Infra, agent_id: str, fifo: FifoMemory, goals: GoalStack) -> str:
    parts = [f"== ROUND {infra.round} ==",
             f"Balance: {infra.ledger.balance(agent_id)} tokens"]
    parts.append("Goal stack (bottom -> top):\n" + goals.render())
    pend = infra.contracts.pending_for(agent_id)
    if pend:
        lines = []
        for c in pend:
            role = "you must respond" if c.status == "proposed" else "you must deliver"
            lines.append(f"- {c.cid} [{c.status}, {role}] with "
                         f"{c.proposer if agent_id != c.proposer else c.contractor}: "
                         f"{c.task} @ {c.price}")
        parts.append("Contracts needing your attention:\n" + "\n".join(lines))
    unread = infra.chat.unread(agent_id)
    if unread:
        parts.append("Unread messages:\n" +
                     "\n".join(f"- from {m.sender}: {m.text}" for m in unread[-10:]))
    parts.append("Your recent actions:\n" + fifo.render())
    parts.append("Choose exactly one action now.")
    return "\n\n".join(parts)
```

- [ ] **Step 4: Run tests** — `python -m pytest tests/test_context.py -q` — 2 passed.

- [ ] **Step 5: Commit**

```bash
git add src/ca/context.py tests/test_context.py
git commit -m "feat: system prompt template and per-turn context rendering"
```

---

### Task 12: Agent turn loop, LLM policy, scripted policy

**Files:**
- Create: `src/ca/agent.py`
- Test: `tests/test_agent.py`

**Interfaces:**
- Consumes: Tasks 10–11.
- Produces:
  - `Decision` dataclass: `name: str, inp: dict, in_tokens: int, out_tokens: int`.
  - `Policy` protocol: `decide(system: str, context: str, tools: list[dict]) -> Decision`.
  - `ScriptedPolicy(script: list[tuple[str, dict]])` — pops next; returns `Decision(name, inp, 0, 0)`; when exhausted returns `Decision("check_balance", {}, 0, 0)`.
  - `LLMPolicy(model, max_tokens)` — Anthropic call (see code); malformed/no-tool response → `Decision("__noop__", {}, in, out)`.
  - `Agent(agent_id, cfg, infra, policy)` with `.fifo .goals` and `take_turn() -> dict` (the trace event). Turn algorithm: render → decide → permission check → dispatch (goal actions handled locally) → bill if billable → mark chat read → push FIFO → return event dict.

- [ ] **Step 1: Write the failing tests**

`tests/test_agent.py`:
```python
from ca.agent import Agent, ScriptedPolicy, Decision
from ca.config import LEVELS, ExperimentConfig
from ca.infra import Infra
from ca.retrieval import Bm25Backend
from ca.taskboard import Question

DOCS = [{"title": "Paris", "text": "Paris is the capital of France."}]


def make(level="L0"):
    cfg = ExperimentConfig(level=LEVELS[level], seed=0, seed_capital_total=1000)
    infra = Infra(cfg, [Question("q0001", "capital of France?", ["Paris"], "easy", 100)],
                  retriever=Bm25Backend(DOCS))
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


def test_turn_marks_chat_read():
    cfg, infra = make()
    infra.chat.send("agent_2", "agent_1", "ping", 0)
    ag = Agent("agent_1", cfg, infra, ScriptedPolicy([("check_balance", {})]))
    ag.take_turn()
    assert infra.chat.unread("agent_1") == []
```

- [ ] **Step 2: Run to verify failure** — `python -m pytest tests/test_agent.py -q` — ImportError.

- [ ] **Step 3: Implement**

`src/ca/agent.py`:
```python
"""Agent = short-term memory + a policy that picks one action per turn."""
import json
from dataclasses import dataclass
from typing import Protocol

import anthropic

from ca.actions import dispatch, is_billable, permission_error, visible_tools
from ca.config import ExperimentConfig
from ca.context import render_turn, system_prompt
from ca.infra import Infra
from ca.memory import FifoMemory, GoalStack


@dataclass
class Decision:
    name: str
    inp: dict
    in_tokens: int
    out_tokens: int


class Policy(Protocol):
    def decide(self, system: str, context: str, tools: list[dict]) -> Decision: ...


class ScriptedPolicy:
    """Deterministic policy for tests: replays a fixed action list."""

    def __init__(self, script: list[tuple[str, dict]], in_tokens: int = 0, out_tokens: int = 0):
        self.script = list(script)
        self.in_tokens, self.out_tokens = in_tokens, out_tokens

    def decide(self, system, context, tools) -> Decision:
        if not self.script:
            return Decision("check_balance", {}, self.in_tokens, self.out_tokens)
        name, inp = self.script.pop(0)
        return Decision(name, inp, self.in_tokens, self.out_tokens)


class LLMPolicy:
    def __init__(self, model: str, max_tokens: int = 1024):
        self.client = anthropic.Anthropic()
        self.model = model
        self.max_tokens = max_tokens

    def decide(self, system, context, tools) -> Decision:
        resp = self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=system,
            messages=[{"role": "user", "content": context}],
            tools=tools,
            tool_choice={"type": "any", "disable_parallel_tool_use": True},
        )
        usage = resp.usage
        for block in resp.content:
            if block.type == "tool_use":
                return Decision(block.name, dict(block.input),
                                usage.input_tokens, usage.output_tokens)
        return Decision("__noop__", {}, usage.input_tokens, usage.output_tokens)


class Agent:
    def __init__(self, agent_id: str, cfg: ExperimentConfig, infra: Infra, policy: Policy):
        self.id = agent_id
        self.cfg = cfg
        self.infra = infra
        self.policy = policy
        self.fifo = FifoMemory(cfg.fifo_k)
        self.goals = GoalStack("maximize token balance")
        self._system = system_prompt(cfg.level, agent_id, infra.agent_ids)
        self._tools = visible_tools(cfg.level, agent_id)

    def take_turn(self) -> dict:
        context = render_turn(self.infra, self.id, self.fifo, self.goals)
        d = self.policy.decide(self._system, context, self._tools)
        billable = False
        if d.name == "__noop__":
            result = "ERROR: no valid action produced this turn"
        elif d.name == "push_goal":
            self.goals.push(str(d.inp.get("note", "")))
            result = "goal pushed"
        elif d.name == "pop_goal":
            try:
                result = f"popped goal: {self.goals.pop()}"
            except IndexError as e:
                result = f"ERROR: {e}"
        else:
            err = permission_error(self.infra, self.id, d.name, d.inp)
            if err:
                result = f"ERROR: {err}"
            else:
                result = dispatch(self.infra, self.id, d.name, d.inp)
                billable = is_billable(d.name, d.inp) and not result.startswith("ERROR")
        if billable:
            self.infra.ledger.burn(self.id, d.in_tokens + d.out_tokens)
        self.infra.chat.mark_read(self.id)  # rendered messages are now "seen"
        self.fifo.add(f"{d.name}({json.dumps(d.inp, ensure_ascii=False)[:120]})", result[:300])
        return {
            "round": self.infra.round, "agent": self.id,
            "action": d.name, "input": d.inp, "result": result,
            "billable": billable, "tokens_in": d.in_tokens, "tokens_out": d.out_tokens,
            "balance_after": self.infra.ledger.balance(self.id),
        }
```

Note the billing rule refinement (consistent with spec §14 "计费回合基础设施侧失败不扣费"): a billable action whose dispatch returned an `ERROR:` string is NOT billed.

- [ ] **Step 4: Run tests** — `python -m pytest tests/test_agent.py -q` — 5 passed. Full suite: `python -m pytest -q`.

- [ ] **Step 5: Commit**

```bash
git add src/ca/agent.py tests/test_agent.py
git commit -m "feat: agent turn loop with LLM and scripted policies, turn billing"
```

---

### Task 13: Scheduler + recorder + deterministic end-to-end test

**Files:**
- Create: `src/ca/recorder.py`, `src/ca/scheduler.py`
- Test: `tests/test_e2e_scripted.py`

**Interfaces:**
- Produces:
  - `Recorder(out_dir)` with `.log(event: dict)` (append JSONL to `trace.jsonl`, flush immediately), `.write_summary(infra, agents)` (write `summary.json`: per-question results, final balances, per-agent billable/free token totals, rounds used, conservation check).
  - `Scheduler(infra, agents, cfg, recorder, rng: random.Random)` with `.run()->dict` (the summary): rounds 1..max_rounds, shuffled order per round, stop early when `board.all_done()`.

- [ ] **Step 1: Write the failing tests**

`tests/test_e2e_scripted.py`:
```python
import json
import random

from ca.agent import Agent, ScriptedPolicy
from ca.config import LEVELS, ExperimentConfig
from ca.infra import Infra
from ca.recorder import Recorder
from ca.retrieval import Bm25Backend
from ca.scheduler import Scheduler
from ca.taskboard import Question

DOCS = [{"title": "Paris", "text": "Paris is the capital of France."}]


def build(level, scripts, tmp_path, n_questions=1):
    cfg = ExperimentConfig(level=LEVELS[level], seed=7, seed_capital_total=1000, max_rounds=10)
    qs = [Question(f"q{i:04d}", "capital of France?", ["Paris"], "easy", 100)
          for i in range(1, n_questions + 1)]
    infra = Infra(cfg, qs, retriever=Bm25Backend(DOCS))
    agents = [Agent(a, cfg, infra, ScriptedPolicy(scripts.get(a, []), in_tokens=10, out_tokens=5))
              for a in infra.agent_ids]
    rec = Recorder(str(tmp_path))
    return infra, Scheduler(infra, agents, cfg, rec, random.Random(cfg.seed))


def test_solo_answer_flow_L5(tmp_path):
    scripts = {"agent_1": [
        ("list_questions", {}),
        ("claim_question", {"qid": "q0001"}),
        ("retrieve", {"query": "capital of France"}),
        ("deliver_work", {"target_id": "q0001", "content": "Paris"}),
    ]}
    infra, sched = build("L5", scripts, tmp_path)
    summary = sched.run()
    assert summary["questions"][0]["score"] == 1.0
    assert summary["conservation_ok"] is True
    # billable turns: retrieve + deliver_work = 2 * 15 tokens burned
    assert summary["tokens"]["agent_1"]["billable"] == 30
    trace = [json.loads(l) for l in open(tmp_path / "trace.jsonl")]
    assert len(trace) >= 4


def test_subcontract_flow_L1(tmp_path):
    # interface claims, subcontracts to agent_1, agent_1 delivers, interface answers WORLD
    scripts = {
        "interface": [
            ("claim_question", {"qid": "q0001"}),
            ("propose_contract", {"to": "agent_1", "task": "find the capital of France", "price": 40}),
            ("check_balance", {}),                      # waits while agent_1 works
            ("check_balance", {}),
            ("deliver_work", {"target_id": "q0001", "content": "Paris"}),
        ],
        "agent_1": [
            ("accept_contract", {"contract_id": "c0001"}),
            ("retrieve", {"query": "capital of France"}),
            ("deliver_work", {"target_id": "c0001", "content": "The answer is Paris"}),
        ],
    }
    infra, sched = build("L1", scripts, tmp_path)
    summary = sched.run()
    assert summary["questions"][0]["score"] == 1.0
    assert summary["conservation_ok"] is True
    # agent_1 earned the 40-token escrow minus its own burn
    assert summary["balances"]["agent_1"] > 1000 // 8
    assert summary["rounds_used"] <= 10


def test_stops_at_max_rounds(tmp_path):
    infra, sched = build("L5", {}, tmp_path)  # nobody answers
    summary = sched.run()
    assert summary["rounds_used"] == 10
    assert summary["questions"][0]["status"] != "closed"
```

Note on ordering in `test_subcontract_flow_L1`: within a round the order is shuffled, so the interface's script inserts wait turns (`check_balance`) to tolerate either ordering; agent_1's `accept_contract` errors harmlessly in round 1 if it runs before the proposal — to make it deterministic regardless of shuffle, the ScriptedPolicy replays strictly in order and errored turns consume the scripted step. If the seed ordering makes the flow fail, adjust the wait-turn count, not the assertions.

- [ ] **Step 2: Run to verify failure** — `python -m pytest tests/test_e2e_scripted.py -q` — ImportError.

- [ ] **Step 3: Implement**

`src/ca/recorder.py`:
```python
"""Crash-safe JSONL trace + end-of-run summary."""
import json
from collections import defaultdict
from pathlib import Path


class Recorder:
    def __init__(self, out_dir: str):
        self.dir = Path(out_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self._f = open(self.dir / "trace.jsonl", "a")
        self._tokens = defaultdict(lambda: {"billable": 0, "free": 0})

    def log(self, event: dict) -> None:
        spent = event["tokens_in"] + event["tokens_out"]
        key = "billable" if event["billable"] else "free"
        self._tokens[event["agent"]][key] += spent
        self._f.write(json.dumps(event, ensure_ascii=False) + "\n")
        self._f.flush()

    def write_summary(self, infra, rounds_used: int) -> dict:
        summary = {
            "level": infra.cfg.level.level,
            "seed": infra.cfg.seed,
            "rounds_used": rounds_used,
            "questions": [{
                "qid": q.qid, "status": q.status, "difficulty": q.difficulty,
                "price": q.price, "claimed_by": q.claimed_by,
                "submitted": q.submitted, "score": q.score, "em": q.em,
                "payout": q.payout,
            } for q in infra.board.results()],
            "balances": {a: infra.ledger.balance(a) for a in infra.agent_ids},
            "tokens": {a: dict(self._tokens[a]) for a in infra.agent_ids},
            "bankrupt": [a for a in infra.agent_ids if infra.ledger.is_bankrupt(a)],
            "n_contracts": len(infra.contracts.contracts),
            "contract_prices": [c.price for c in infra.contracts.contracts.values()
                                if c.status == "delivered"],
            "minted": infra.ledger.minted,
            "burned": infra.ledger.burned,
            "conservation_ok": infra.ledger.conservation_ok(),
        }
        with open(self.dir / "summary.json", "w") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        return summary

    def close(self):
        self._f.close()
```

`src/ca/scheduler.py`:
```python
"""Synchronous round-robin scheduler with seeded per-round shuffling."""
import random

from ca.agent import Agent
from ca.config import ExperimentConfig
from ca.infra import Infra
from ca.recorder import Recorder


class Scheduler:
    def __init__(self, infra: Infra, agents: list[Agent], cfg: ExperimentConfig,
                 recorder: Recorder, rng: random.Random):
        self.infra = infra
        self.agents = agents
        self.cfg = cfg
        self.recorder = recorder
        self.rng = rng

    def run(self) -> dict:
        rounds_used = 0
        for r in range(1, self.cfg.max_rounds + 1):
            self.infra.round = r
            rounds_used = r
            order = list(self.agents)
            self.rng.shuffle(order)
            for agent in order:
                event = agent.take_turn()
                self.recorder.log(event)
            if self.infra.board.all_done():
                break
        summary = self.recorder.write_summary(self.infra, rounds_used)
        self.recorder.close()
        return summary
```

- [ ] **Step 4: Run tests** — `python -m pytest tests/test_e2e_scripted.py -q` — 3 passed (if the L1 shuffle ordering under seed 7 breaks the scripted flow, add/remove one `("check_balance", {})` wait turn in the interface script until green — the invariant assertions must not be weakened). Full suite: `python -m pytest -q`.

- [ ] **Step 5: Commit**

```bash
git add src/ca/recorder.py src/ca/scheduler.py tests/test_e2e_scripted.py
git commit -m "feat: round-robin scheduler and JSONL recorder with e2e scripted tests"
```

---

### Task 14: Metrics module

**Files:**
- Create: `src/ca/metrics.py`
- Test: `tests/test_metrics.py`

**Interfaces:**
- Consumes: summary dict shape from Task 13.
- Produces: `gini(values: list[int|float])->float`; `compute_metrics(summary: dict)->dict` with keys: `total_f1, total_em, n_answered, accuracy_per_ktok_billable, accuracy_per_ktok_all, coordination_overhead, rounds_used, bankrupt_rate, gini_final, n_contracts, mean_contract_price`.

- [ ] **Step 1: Write the failing tests**

`tests/test_metrics.py`:
```python
import pytest
from ca.metrics import compute_metrics, gini


def test_gini():
    assert gini([1, 1, 1, 1]) == pytest.approx(0.0)
    assert gini([0, 0, 0, 10]) == pytest.approx(0.75, abs=0.01)


def test_compute_metrics():
    summary = {
        "questions": [
            {"score": 1.0, "em": 1.0, "status": "closed"},
            {"score": 0.5, "em": 0.0, "status": "closed"},
            {"score": 0.0, "em": 0.0, "status": "open"},
        ],
        "balances": {"a": 100, "b": 0},
        "tokens": {"a": {"billable": 1000, "free": 500}, "b": {"billable": 0, "free": 500}},
        "bankrupt": ["b"],
        "rounds_used": 9,
        "n_contracts": 2,
        "contract_prices": [30, 50],
    }
    m = compute_metrics(summary)
    assert m["total_f1"] == 1.5 and m["total_em"] == 1.0 and m["n_answered"] == 2
    assert m["accuracy_per_ktok_billable"] == pytest.approx(1.5 / 1.0)     # per 1000 billable
    assert m["accuracy_per_ktok_all"] == pytest.approx(1.5 / 2.0)          # per 1000 all
    assert m["coordination_overhead"] == pytest.approx(1000 / 2000)
    assert m["bankrupt_rate"] == 0.5
    assert m["mean_contract_price"] == 40
```

- [ ] **Step 2: Run to verify failure** — `python -m pytest tests/test_metrics.py -q` — ImportError.

- [ ] **Step 3: Implement**

`src/ca/metrics.py`:
```python
"""Headline and auxiliary metrics computed from a run summary."""


def gini(values) -> float:
    vals = sorted(values)
    n = len(vals)
    total = sum(vals)
    if n == 0 or total == 0:
        return 0.0
    cum = 0.0
    for i, v in enumerate(vals, start=1):
        cum += i * v
    return (2 * cum) / (n * total) - (n + 1) / n


def compute_metrics(summary: dict) -> dict:
    qs = summary["questions"]
    total_f1 = sum(q["score"] for q in qs)
    total_em = sum(q["em"] for q in qs)
    n_answered = sum(1 for q in qs if q["status"] == "closed")
    billable = sum(t["billable"] for t in summary["tokens"].values())
    free = sum(t["free"] for t in summary["tokens"].values())
    all_tok = billable + free
    prices = summary.get("contract_prices", [])
    return {
        "total_f1": total_f1,
        "total_em": total_em,
        "n_answered": n_answered,
        "accuracy_per_ktok_billable": total_f1 / (billable / 1000) if billable else 0.0,
        "accuracy_per_ktok_all": total_f1 / (all_tok / 1000) if all_tok else 0.0,
        "coordination_overhead": free / all_tok if all_tok else 0.0,
        "rounds_used": summary["rounds_used"],
        "bankrupt_rate": len(summary["bankrupt"]) / len(summary["balances"]),
        "gini_final": gini([max(b, 0) for b in summary["balances"].values()]),
        "n_contracts": summary["n_contracts"],
        "mean_contract_price": sum(prices) / len(prices) if prices else 0.0,
    }
```

- [ ] **Step 4: Run tests** — `python -m pytest tests/test_metrics.py -q` — 2 passed.

- [ ] **Step 5: Commit**

```bash
git add src/ca/metrics.py tests/test_metrics.py
git commit -m "feat: metrics (accuracy-per-ktok, coordination overhead, gini)"
```

---

### Task 15: Runner CLI + data prep script

**Files:**
- Create: `src/ca/runner.py`, `scripts/prepare_data.py`
- Test: `tests/test_runner.py` (loading only; no live LLM in tests)

**Interfaces:**
- Produces:
  - `load_questions(path)->list[Question]` — JSONL with fields `qid,text,answers,difficulty,price`.
  - `python -m ca.runner --level L0 --questions data/pool.jsonl --index data/index --seed 0 --capital 200000 --max-rounds 60 --model claude-haiku-4-5 --out runs/L0_s0` — builds everything with `LLMPolicy`, runs, prints `compute_metrics` result.
  - `scripts/prepare_data.py --hotpot-n 150 --musique-n 50 --out data/` — builds `pool.jsonl`, pooled paragraph corpus, and a saved bm25s index.

- [ ] **Step 1: Write the failing test**

`tests/test_runner.py`:
```python
import json
from ca.runner import load_questions


def test_load_questions(tmp_path):
    p = tmp_path / "pool.jsonl"
    rows = [
        {"qid": "q0001", "text": "t1", "answers": ["a"], "difficulty": "2hop", "price": 1000},
        {"qid": "q0002", "text": "t2", "answers": ["b", "c"], "difficulty": "4hop", "price": 3000},
    ]
    p.write_text("\n".join(json.dumps(r) for r in rows))
    qs = load_questions(str(p))
    assert len(qs) == 2 and qs[1].answers == ["b", "c"] and qs[1].price == 3000
```

- [ ] **Step 2: Run to verify failure** — `python -m pytest tests/test_runner.py -q` — ImportError.

- [ ] **Step 3: Implement runner**

`src/ca/runner.py`:
```python
"""CLI entry point: run one (level, seed) experiment."""
import argparse
import json
import random

from ca.agent import Agent, LLMPolicy
from ca.config import LEVELS, ExperimentConfig
from ca.infra import Infra
from ca.metrics import compute_metrics
from ca.recorder import Recorder
from ca.retrieval import Bm25Backend
from ca.scheduler import Scheduler
from ca.taskboard import Question


def load_questions(path: str) -> list[Question]:
    out = []
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            out.append(Question(r["qid"], r["text"], r["answers"],
                                r["difficulty"], r["price"]))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--level", required=True, choices=list(LEVELS))
    ap.add_argument("--questions", required=True)
    ap.add_argument("--index", required=True, help="bm25s index dir from prepare_data")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--capital", type=int, default=200_000)
    ap.add_argument("--max-rounds", type=int, default=60)
    ap.add_argument("--model", default="claude-haiku-4-5")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    cfg = ExperimentConfig(level=LEVELS[args.level], seed=args.seed,
                           seed_capital_total=args.capital,
                           max_rounds=args.max_rounds, model=args.model)
    infra = Infra(cfg, load_questions(args.questions),
                  retriever=Bm25Backend.load(args.index))
    agents = [Agent(a, cfg, infra, LLMPolicy(cfg.model, cfg.max_tokens_per_turn))
              for a in infra.agent_ids]
    sched = Scheduler(infra, agents, cfg, Recorder(args.out), random.Random(args.seed))
    summary = sched.run()
    metrics = compute_metrics(summary)
    print(json.dumps(metrics, indent=2))
    with open(f"{args.out}/metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Implement data prep**

`scripts/prepare_data.py`:
```python
"""Build the mixed HotpotQA+MuSiQue pool, pooled paragraph corpus, bm25s index.

Corpus strategy (standard open-retrieval practice, cf. IRCoT): pool the
per-question paragraph sets (gold + distractors) across all sampled questions
into one deduplicated corpus. Gold paragraphs are guaranteed present.
"""
import argparse
import json
import random
import sys
from pathlib import Path

from datasets import load_dataset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from ca.retrieval import Bm25Backend  # noqa: E402

PRICES = {"2hop": 1000, "3hop": 2000, "4hop": 3000}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hotpot-n", type=int, default=150)
    ap.add_argument("--musique-n", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="data")
    args = ap.parse_args()
    rng = random.Random(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    pool, corpus, seen = [], [], set()

    def add_doc(title, text):
        key = (title, text[:80])
        if key not in seen:
            seen.add(key)
            corpus.append({"title": title, "text": text})

    # ---- HotpotQA (distractor config: paragraphs travel with the question) ----
    hp = load_dataset("hotpotqa/hotpot_qa", "distractor", split="validation",
                      trust_remote_code=True)
    idxs = rng.sample(range(len(hp)), args.hotpot_n)
    for n, i in enumerate(idxs):
        ex = hp[i]
        pool.append({"qid": f"q{len(pool)+1:04d}", "text": ex["question"],
                     "answers": [ex["answer"]], "difficulty": "2hop",
                     "price": PRICES["2hop"], "source": "hotpotqa"})
        for title, sents in zip(ex["context"]["title"], ex["context"]["sentences"]):
            add_doc(title, " ".join(sents))

    # ---- MuSiQue (answerable) ----
    mq = load_dataset("dgslibisey/MuSiQue", split="validation")
    idxs = rng.sample(range(len(mq)), args.musique_n)
    for i in idxs:
        ex = mq[i]
        hops = "4hop" if ex["id"].startswith("4hop") else (
               "3hop" if ex["id"].startswith("3hop") else "2hop")
        answers = [ex["answer"]] + list(ex.get("answer_aliases") or [])
        pool.append({"qid": f"q{len(pool)+1:04d}", "text": ex["question"],
                     "answers": answers, "difficulty": hops,
                     "price": PRICES[hops], "source": "musique"})
        for p in ex["paragraphs"]:
            add_doc(p["title"], p["paragraph_text"])

    with open(out / "pool.jsonl", "w") as f:
        for r in pool:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"pool: {len(pool)} questions; corpus: {len(corpus)} paragraphs")

    Bm25Backend(corpus).save(str(out / "index"))
    print(f"index saved to {out}/index")


if __name__ == "__main__":
    main()
```

Note: if HF field names differ from the above (dataset schemas occasionally shift), print `hp.features` / `mq.features` and adapt the field access — do not change the output format (`pool.jsonl` schema and corpus dict shape are contracts consumed by Task 15's runner).

- [ ] **Step 5: Run tests** — `python -m pytest tests/test_runner.py -q` — 1 passed. Full suite green: `python -m pytest -q`.

- [ ] **Step 6: Commit**

```bash
git add src/ca/runner.py scripts/prepare_data.py tests/test_runner.py
git commit -m "feat: experiment runner CLI and HotpotQA+MuSiQue data prep"
```

---

### Task 16: Live smoke test (manual gate)

**Files:**
- Create: `scripts/smoke.sh`

**Interfaces:**
- Consumes: everything. Requires `ANTHROPIC_API_KEY` (or `ant auth login` profile) and network.

- [ ] **Step 1: Write the smoke script**

`scripts/smoke.sh`:
```bash
#!/usr/bin/env bash
# Tiny live run: 3 questions, L5 single agent, ~10 LLM calls. Costs cents.
set -euo pipefail
cd "$(dirname "$0")/.."
pip install -e ".[data,dev]" -q
python scripts/prepare_data.py --hotpot-n 3 --musique-n 0 --out data/smoke
python -m ca.runner --level L5 --questions data/smoke/pool.jsonl \
  --index data/smoke/index --seed 0 --capital 30000 --max-rounds 15 \
  --out runs/smoke_L5
echo "=== metrics ==="
cat runs/smoke_L5/metrics.json
```

- [ ] **Step 2: Run it**

Run: `bash scripts/smoke.sh`
Expected: completes without exception; `runs/smoke_L5/trace.jsonl` shows the agent claiming/retrieving/answering; `metrics.json` printed with nonzero `accuracy_per_ktok_billable` if any answer was right. Inspect the trace manually for sane behavior (this is the human review gate before pilot runs).

- [ ] **Step 3: Then a 2-agent config sanity check**

Run: `python -m ca.runner --level L1 --questions data/smoke/pool.jsonl --index data/smoke/index --seed 0 --capital 60000 --max-rounds 20 --out runs/smoke_L1`
Expected: trace shows interface claiming questions; watch for whether subcontracting emerges (not required at this scale).

- [ ] **Step 4: Commit**

```bash
git add scripts/smoke.sh
git commit -m "chore: live smoke test script"
```

---

## Out of Plan (deliberate)

- Pilot calibration of `capital`, prices, `max_rounds`, `fifo_k` — experiment-phase work using the runner, not code work.
- Prompt caching (Haiku 4.5 min cacheable prefix is 4096 tokens; revisit if system prompt grows), FanOutQA arm, dense retrieval, memory-centralization config, GAIA-class tasks — all spec §16.
- Experiment sweep orchestration (6 levels × seeds) — a 10-line shell loop over `ca.runner` once pilot params are fixed.

## Self-Review Notes

- Spec coverage check: §3 architecture→Tasks 9–13; §4 unified contracts→Tasks 5–6,10; §5 action catalog + billing→Task 10; §6 runtime/context→Tasks 11–12; §7 scheduling→Task 13; §8 economy→Tasks 2,6,9; §9 matrix→Task 9 (tested in Task 10); §10 data→Task 15; §11 metrics→Task 14; §12–13 baselines/protocol→experiment phase (out of code scope); §14 error handling→Tasks 10,12 (error strings, no-bill on failed billable); §15 testing→every task + Tasks 13,16.
- Type consistency: `Decision`, `Question`, `Contract`, summary dict keys cross-checked across Tasks 12–15.
