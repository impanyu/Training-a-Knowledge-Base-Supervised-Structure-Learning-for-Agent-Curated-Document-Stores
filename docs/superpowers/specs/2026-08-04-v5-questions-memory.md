# v5: questions as tasks, corpus as memory (spec)

Two simplifications (user, 2026-08-04):

1. **A task is one question.** The job layer (bundles, JSON maps, all-or-nothing)
   is deleted. Claim a question, deliver its short answer, get graded, paid.
2. **The corpus IS long-term memory.** Every agent's memory starts pre-loaded
   with the full paragraph corpus. There is NO separate retrieval index and NO
   `retrieve` action: `memory_search` is the single knowledge query, over
   corpus + accumulated notes + past answers alike. Knowledge has exactly one
   substrate. Under C2 (shared memory) the whole system shares one store; in
   every other config each agent has its own (identical at birth, diverging
   through experience).

## 1. Data (`data/v5/`, already built)

- `bank.json` — 1000 questions (700 HotpotQA + 300 MuSiQue; ~86% 2hop),
  prices 18k/30k/45k, `topic` labels from 40 k-means clusters (metadata for
  the specialization metric only).
- `corpus.jsonl` — ~12k deduplicated paragraphs `{title, text}`.
- `corpus_emb.npy` — float32 paragraph embeddings (chroma default ONNX),
  computed ONCE and reused to seed every memory bucket cheaply.

## 2. Bank & board

`QuestionBank`: questions only (`get`, `total_units` = len). `Job` deleted.
`QuestionBoard` (rework of JobBoard): per-question claim/deliver.
- `open_questions(viewer)` — unclaimed & unclosed; per-viewer stable shuffle
- `claim(agent, qid, round)` — one claimant at a time; 2 strikes per
  (question, agent) apply only where a TTL is configured (default: none)
- `deliver(agent, qid, answer_text)` — requires the claim; grades (F1/EM),
  mints round(price x F1), closes the question. One graded attempt per claim.
- Result rows keep `topic` for metrics.

## 3. Memory seeding

`AgentMemory.seed_corpus(paras, embeddings, agents)` adds every paragraph as an
entry `{kind: "corpus", title}` with PRECOMPUTED embeddings (no re-embedding)
to each listed bucket (one call seeds the shared bucket when shared=True).
- Corpus entries are EXCLUDED from `to_state()`; `from_state` assumes a
  freshly-seeded store and re-adds only notes/answers. Seq counter must stay
  deterministic across save/restore (corpus consumes ids first).
- `memory_search` returns corpus hits formatted like retrieval used to
  (`[title] text`), notes and answers in their existing formats.

## 4. Actions (18 total)

Solving: `memory_search` (evidence gathering is solving now), `deliver_work`
(question target). Admin: `list_questions`, `claim_question` (auto-recall of a
stored ANSWER with F1 stays — corpus hits do NOT count as stored answers),
chat, contracts (bind to qid), pay, loans, goals, `memory_write`,
`check_balance`, `list_agents`. Deleted: `retrieve`, `list_jobs`, `claim_job`.
`memory_write` stays admin.

## 5. Ripples

- context.py: system prompt rewritten (pipeline: list -> claim -> memory_search
  -> deliver ONE short answer; "your memory was born knowing the corpus");
  active-claims line becomes `- [q0042] <text> (2hop, reward 18000)` (live
  state; the question TEXT is cheap for single questions, unlike jobs).
- skills.py: demos rewritten; memory demo now teaches searching the corpus
  and writing intermediate findings.
- infra.py: no retriever; takes corpus + embeddings, seeds memory at init.
- runner.py: `--bank data/v5/bank.json` (corpus/emb paths derived from it).
- recorder/metrics: `memory_hit_rate` (claims carrying a stored answer) and
  `improvement_rate` unchanged; `demand_absorbed` = closed/1000;
  specialization over topics unchanged; drop delegation_rate.
- checkpoint: fidelity must hold with seeded stores (corpus excluded, re-seeded
  deterministically on resume).
