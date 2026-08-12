# v6 — Cooperation without an economy

The token economy is deleted. Agents still talk, still ask each other for
information, still share memory under C2 — but nothing is bought, sold, priced,
lent or paid for. The root goal is cooperative by default.

Motivation: across v3–v5 the market never formed. With homogeneous agents and
equal board access the bargaining zone is empty at any price (buyer ceiling
P/2, seller floor 2P), so trade was never the mechanism producing cooperation —
it was only producing accounting. Removing it makes the remaining
centralization factors (task access, memory, communication topology) the sole
independent variables.

## 1. What is deleted

Modules removed entirely: `economy.py` (Ledger), `contracts.py`, `loans.py`,
and their tests.

Concepts removed: balance, seed capital, escrow, mint, burn, conservation
invariant, bankruptcy (and the bankruptcy action freeze), contracts of every
status, loans, interest, question price, contract price, central pricing queue,
credit gating, gini, coverage.

Actions removed (11): `propose_contract`, `accept_contract`,
`reject_contract`, `counter_offer`, `cancel_contract`, `set_price`, `pay`,
`propose_loan`, `accept_loan`, `repay_loan`, `check_balance`.

Config fields removed: `seed_capital_total`, `loan_rate`, `central_pricing`,
`central_credit`, `collective_goal`. CLI flag `--capital` removed.

## 2. What survives

`QuestionBank` keeps `price` in the JSON (harmless metadata, useful for
post-hoc difficulty slicing) but nothing reads it as money. Grading is
unchanged: `f1`/`exact_match` per delivery, recorded, never converted to
tokens.

Token counting survives as **pure measurement**. Every turn still records
`tokens_in`/`tokens_out` and a `solving`/`admin` category; no balance is
debited. `coordination_overhead = admin / (admin + solving)` becomes the
headline efficiency metric.

## 3. Configs

C3, C4 and C6 are deleted. C6's collective goal is now every agent's default
root goal, so it would be identical to C0.

| | agents | hub | factor |
|---|---|---|---|
| C0 | 8 | no | decentralized baseline |
| C1 | 8 | yes | `world_access="hub"` — only the hub may list/claim/deliver |
| C2 | 8 | no | `shared_memory=True` — one long-term memory for everyone |
| C5 | 8 | yes | `star_comms=True` — non-hub agents may only message the hub |
| C7 | 1 | no | solo baseline |

The gap in the naming is deliberate: C0/C1/C2/C5/C7 keep the identities they
had in v3–v5 so earlier results stay referenceable.

`LevelConfig` keeps `level, n_agents, has_hub, world_access, star_comms,
shared_memory`.

## 4. Root goal

`GoalStack` is rooted, for every agent at every multi-agent config, at:

> Cooperate with the other agents to answer as many questions correctly as
> possible.

At C7 (solo, no peers) the root is:

> Answer as many questions correctly as possible.

The root is permanent and cannot be popped (existing `GoalStack` behaviour).
The system prompt states it as a shared objective — no agent has a private
score to maximize.

## 5. Action catalog (11)

Solving: `memory_search`, `deliver_work` (question target).
Admin: `list_questions`, `claim_question`, `release_question`, `memory_write`,
`send_message`, `read_chat`, `push_goal`, `pop_goal`, `list_agents`.

`deliver_work(target_id, content)` now only ever targets a question id;
`_is_contract_target` and the contract delivery branch are deleted, so
`classify` reduces to `name in {"memory_search", "deliver_work"}` → solving.

**`release_question(qid)` (new).** Returns a question the caller holds to the
open board: the claim is dropped, the question becomes claimable by anyone
again, no penalty. Errors if the caller does not hold `qid`. This is the only
way to hand work off now that contracts are gone, and claims still never
expire.

Cooperation runs entirely through `send_message` / `read_chat`: an agent asks a
peer what it knows, the peer answers from its memory. Under C5 that traffic
must route through the hub; under C1 only the hub can touch the board, so
non-hub agents can only contribute by answering the hub's questions over chat.

## 6. Ripples

- **`infra.py`** — drops `ledger`, `contracts`, `loans`. Holds: `bank`, `board`,
  `chat`, `memory`, `cfg`, `agent_ids`, `round`.
- **`board.py`** — `QuestionBoard(bank)`, no ledger. `deliver` grades, closes,
  records a result; no payout. `release(agent, qid)` added.
- **`context.py`** — balance line, pending-contracts block, loans block and
  pricing-queue block deleted. `render_turn` emits: round, goal stack, active
  claims, unread messages, repetition warning, FIFO.
- **`skills.py`** — contractor/hire-peer/loan demos replaced by an **ask-a-peer**
  demo (message a peer for a fact, peer replies from memory, asker delivers) and
  a **release** demo. No price talk anywhere.
- **`recorder.py`** — timeseries drops balances/escrow/minted/burned/bankrupt/
  contracts/loans. Keeps round, tokens, solving/admin totals,
  coordination_overhead (+by agent), answered, board {open, active_claims,
  closed}, total/remaining units, demand_absorbed, memory block, n_claims,
  memory_hit_rate, improvement_rate, and adds `n_messages`.
- **`metrics.py`** — drops `gini`, `bankrupt_rate`, `mean_contract_price`,
  `n_contracts`, loan fields, coverage. Keeps n_answered, total/mean F1 and EM,
  demand_absorbed, accuracy_per_ktok_{solving,all}, coordination_overhead,
  admin_solving_ratio, specialization, memory_hit_rate, improvement_rate,
  answers_in_memory_total; adds `n_messages` and `messages_per_answer`.
- **`checkpoint.py`** — ledger/contract/loan capture and restore deleted;
  board claims, closed set, results, memory, chat, goal stacks, FIFO and
  recorder tallies still restore byte-exactly.
- **`agent.py`** — no balance check before acting, no bankruptcy freeze.
- **`runner.py`** — `--capital` gone.
- **scripts/smoke.sh** — updated for the surviving configs.

## 7. Test intents to preserve (rewritten in v6 terms)

- solo answer flow at C7
- C1: a non-hub agent cannot list/claim/deliver; the hub can
- C2: an answer written by one agent is retrievable by another; at C0 it is not
  (corpus content is identical in both, so the assertion must target
  notes/answers, not corpus rows)
- C5: a non-hub agent may only message the hub
- release_question returns the question to the open pool and another agent can
  then claim it
- a peer-assist round trip: agent A messages B, B replies with a fact from its
  memory, A delivers a correct answer
- timeseries writes one line per round
- checkpoint resume is byte-exact with a seeded memory
- no module imports `economy`, `contracts` or `loans`
