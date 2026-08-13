# v7 — Proactive agent cluster

A pivot, not an iteration: the centralization spectrum (C0–C7) is retired. v7
is a single organizational form — a **cluster of always-on domain experts over
one shared knowledge base** — and the experimental questions move to proactive
behaviour: what does an agent do with idle time, and does speculative
self-questioning pay off when real questions arrive?

## 0. The experiment: proactive vs passive

The headline comparison is **proactive vs passive answering efficiency**, so
v7 ships two arms differing ONLY in what idle time is for:

- **`P0` (proactive)** — as specified below: idle turns are spent inventing
  likely questions, answering them, and banking the results via `record_qa`.
- **`B0` (passive baseline)** — identical world, routing, threads, KB and
  actions EXCEPT: `record_qa` is absent from the catalog, and the system
  prompt contains no proactive protocol — agents answer external questions
  when they arrive and otherwise wait (they may still message peers, read
  chat, take notes).

Same bank, same seeds, same arrival schedule (stream depends only on
bank+seed+N, not the arm), so every external question arrives at the same
round to the same agent in both arms. Efficiency is then read off:
latency, F1/EM, coverage, and tokens_per_answer / total tokens (the proactive
arm pre-pays compute while idle — the question is whether that buys lower
latency and higher F1 per external answer, and at what token price).

CLI: `--level P0|B0`.

## 1. The world

- One shared long-term memory (the knowledge base, KB): corpus-seeded
  `AgentMemory` with a single physical bucket for ALL agents (v6's C2
  semantics, now unconditional). Notes, self-QA and graded answers all land
  in it; every agent searches the same store.
- N agents (CLI `--agents`, default 8). No hub, no star comms, no world-access
  gating: the C-level machinery is deleted. One level id `P0` survives for
  config/checkpoint validation.
- **Domains.** The 1000 bank questions are embedded (same ONNX model as the
  corpus) and k-means-clustered into N clusters (seed 0, deterministic),
  cached at `data/v5/qclusters_{N}.json` as {centroids, assignment}. Agent i
  owns cluster i. The system prompt describes the agent's domain with 5
  exemplar questions nearest its centroid.
- **External question stream.** Questions arrive over the run in a seeded
  shuffled order (topics thereby mixed), with per-round arrival count drawn
  Poisson(λ), λ = CLI `--arrival-rate` (default 0.5). Each arriving question
  is routed to the agent owning its cluster and appended to that agent's
  `external` chat thread as a message from `external`:
  `[q0042] <question text>`. Grading golds stay hidden.
- A question is `pending` from arrival until its assignee delivers; there is
  no claiming, no expiry, no board. `list_questions`/`claim_question`/
  `release_question` are deleted.

## 2. Message box (chat rework)

WeChat-model threads:

- One thread per partner pair. Peers share a thread ({a, b} unordered);
  `external` threads are per-agent (agent ↔ external). Messages:
  {from, text, round, seq}. Full history kept.
- `send_message(to, text)` appends to the pair thread and increments the
  recipient's unread counter for that partner. Agents may message any peer.
- `read_chat(with_agent, page=0)` returns the newest 5 messages of that
  thread (page 0); page k returns the k-th older page. Reading page 0 clears
  the caller's unread counter for that partner. It never consumes or deletes
  messages.
- **Notification list** (short-term memory, rendered every turn): one line per
  partner with unread messages — `New messages: external (2), agent_3 (1)` —
  content NOT shown; the agent must call read_chat. No unread → no block.

## 3. Goal stack

Unchanged mechanics (root pinned, push/pop). Root for every agent:

> Answer questions as well as you can — external questions first, then
> questions you pose yourself.

Protocol taught in the system prompt (not hard-coded): on an external
question, push it, research (memory_search), deliver_work, pop. Idle → invent
the next most-likely-to-be-asked question in your domain (informed by what
external has already asked in your thread), push it, research, `record_qa`,
pop.

## 4. Actions (9)

Solving: `memory_search`, `deliver_work`, `record_qa`.
Admin: `memory_write`, `send_message`, `read_chat`, `push_goal`, `pop_goal`,
`list_agents`.

- `deliver_work(target_id="q0042", content="<short answer>")` — only the
  assignee, only while pending. Grades (F1/EM vs golds), records the result
  with `latency = round_delivered − round_arrived`, appends the answer to the
  external thread, auto-stores the graded answer in the KB (kind="answer",
  qid/f1 metadata, existing machinery), closes the question. A repeat or
  foreign qid errors.
- `record_qa(question, answer)` (new) — proactive product: stores
  "Q: <question>\nA: <answer>" in the shared KB with kind="selfqa" and the
  caller in metadata. This is how speculative work becomes visible to
  everyone (including the tally).
- `memory_write` stays for free-form notes (kind="note").

## 5. Deletions

Modules: `board.py` (replaced by `stream.py`), C-level flags in `config.py`
(`has_hub`, `world_access`, `star_comms`, `shared_memory`,
`hub_turns_per_round`, `solo_turns_per_round`), hub/star/world gating in
`actions.py` + `context.py` + `skills.py`, `claim`/`release`/`list_questions`
actions, claim auto-recall marker machinery, `--solo-turns` CLI flag.
`CONFIGS` shrinks to `{"P0": LevelConfig("P0", proactive=True), "B0":
LevelConfig("B0", proactive=False)}` (n_agents default 8, `--agents`
overrides). The `proactive` flag gates exactly two things: `record_qa` in the
action catalog, and the proactive-protocol block in the system prompt +
skills. Nothing else may branch on it.

## 6. New module: `stream.py`

`QuestionStream(bank, n_agents, seed, arrival_rate)`:
- owns the shuffled arrival order (seeded), the Poisson draw per round
  (dedicated `random.Random`, independent of the scheduler rng), the routing
  table (from the cluster cache), the pending set {qid: (agent, round_in)},
  and the results list (qid, agent, submitted, f1, em, round_in, round_out,
  latency, topic, difficulty).
- `tick(round)` → list of (qid, agent) arrivals, called by the scheduler at
  round start, which appends the question messages + unread notifications.
- exhausts gracefully: when the order runs out, no more arrivals.
- full `to_state`/`from_state` (order position, rng state, pending, results)
  for byte-exact resume.

## 7. Scheduler / runner

Round = arrivals first (`stream.tick`), then every agent takes one turn
(rng-shuffled order as today). Runner: `--agents`, `--arrival-rate`,
`--bank`, `--seed`, `--max-rounds`, `--model`, `--out`, `--checkpoint-every`,
`--resume`. Cluster cache is computed at startup if missing (embed 1000
question texts, KMeans(n_clusters=N, seed 0), write cache; skip when cached).

## 8. Recorder / metrics

Timeseries per round: arrivals_total, pending, answered_total, coverage
(answered/arrived), mean_latency, total_f1/em, per-agent {answered, f1_sum,
selfqa, notes}, kb_answers, kb_selfqa, n_messages, tokens
solving/admin + coordination_overhead. Summary/metrics: mean_f1, mean_em,
coverage, mean+median latency, selfqa_total, selfqa_per_agent,
proactive_ratio (selfqa turns / all solving turns), n_messages,
messages_per_answer, coordination_overhead, specialization (per-agent topic
Herfindahl over delivered questions — topics still hidden metadata),
tokens_per_answer. Deleted: demand_absorbed (superseded by coverage),
memory_hit_rate/improvement_rate (claim machinery gone).

## 9. Context & skills

`render_turn`: round, goal stack, notification list, repetition warning,
FIFO. (No balance, no board, no claims.) `system_prompt`: identity + peer
list, root goal, domain description with the 5 exemplars, the priority rule,
the proactive protocol, message-box mechanics (notifications, read_chat
paging), KB mechanics (born knowing the corpus; record_qa/memory_write are
visible to everyone; deliver_work auto-records). `skills.py` demos: answer an
external question end-to-end; a proactive idle cycle (invent → research →
record_qa); ask a peer whose domain borders yours; read an older page of a
long thread.

## 10. Test intents

- routing determinism: same bank+seed+N → identical routing table and arrival
  schedule; different seeds → different orders.
- stream: Poisson arrivals appear as external messages with unread set;
  exhaustion stops arrivals; tick is idempotent per round.
- chat: pair thread shared by both peers; unread increments on send, clears
  only for the reader, only on page 0; pagination returns older messages;
  history never truncated.
- deliver: grades + latency recorded + thread append + KB auto-store; foreign
  or closed qid errors; second delivery errors.
- record_qa lands in the shared KB and is retrievable by a different agent.
- notification list renders only for partners with unread; disappears after
  read_chat.
- goal stack root fixed and unpoppable; push/pop as before.
- checkpoint resume byte-exact including stream + threads + unread state.
- one timeseries line per round; scripted e2e: arrival → notification → read
  → search → deliver → graded result with latency; idle agent does
  record_qa; no module references boards/claims/levels beyond P0.
