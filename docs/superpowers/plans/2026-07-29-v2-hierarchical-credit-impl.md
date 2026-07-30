# V2 Implementation Plan — Hierarchical Tasks, Credit, All-Action Billing

> Executed via subagent-driven development. Unlike the v1 plan, tasks here carry
> precise interfaces + test requirements rather than full code (implementers on
> capable models, TDD mandatory, per-task review). Spec: docs/superpowers/specs/
> 2026-07-27-centralization-spectrum-design.md (v2 sections).

**Goal:** Upgrade the testbed to spec v2: 7-level spectrum (credit centralization),
hierarchical task trees with packaged delivery, loans at 1%/round, all-action
billing with solving/admin labels, shared subtask library with sentence identity.

## Global Constraints (v2)

- EVERY action bills its turn's input+output tokens from the agent's balance.
  `solving` label = {retrieve, work_on, decompose, deliver_work-to-WORLD};
  `admin` = everything else. Event dict gains `category` field; recorder splits
  token tallies by category; coordination_overhead = admin/(admin+solving).
- Bankruptcy (balance ≤ 0) freezes SOLVING actions only; admin actions remain
  (and still bill — debt deepens; borrowing enables recovery).
- Loans: rate fixed at cfg.loan_rate = 0.01 per round. propose_loan(to, amount)
  → accept_loan(loan_id) transfers principal lender→borrower. Every round the
  scheduler auto-transfers interest = 1% of outstanding principal
  borrower→lender; if borrower can't pay fully, unpaid interest capitalizes
  (adds to principal). repay_loan(loan_id, amount) anytime. Lender loss on
  borrower's terminal insolvency is implicit (no clawback).
- Levels L0..L6 per spec §2/§9: new flag `central_credit` (L4, L5 True):
  lender must be interface. L5 = star comms (old L4 semantics + loans/pay only
  with interface). L6 = solo (old L5). `interface_turns_per_round: int = 1`
  config knob (used by scheduler; >1 reserved for future experiment).
- Task trees: nodes have short id (t####) AND unique one-sentence summary;
  every task/node action parameter accepts either (normalize: lowercase, strip
  punctuation/whitespace; exact → fuzzy best-match with difflib ratio ≥ 0.85,
  ambiguous/low → error listing 3 nearest candidates).
- Tree shape: depth ≤ 4 including root, ≤ 3 children per node, leaves are qids.
- Packaged delivery to WORLD: deliver_work(task, answers_json) where content
  parses as JSON {qid: answer} covering ALL leaves of the task (missing/extra
  qids → ERROR, no state change, attempt not consumed). Grade each leaf F1,
  pay Σ R(leaf)×F1, close task (one successful-parse attempt only: a
  well-formed but incomplete map is rejected without consuming the attempt;
  a well-formed complete map consumes it regardless of scores).
- Node-bound contracts: if propose_contract's task text resolves to a known
  subtask node, contract stores node_id; deliver requires JSON covering that
  node's leaves (coverage check only, no grading). Free-text contracts unchanged.
- Repeat pay per (task, leaf): same question in two tasks pays twice.
- Conservation invariant unchanged and asserted per round (loans are internal
  transfers; interest likewise).
- All existing v1 invariants (escrow atomicity, claim TTL + 2-strike — now at
  task level, pagination, repetition warning) carry over to tasks.

---

### T17: All-action billing + category labels
Files: src/ca/agent.py, src/ca/actions.py (is_billable → classify(name,inp)->"solving"|"admin"),
src/ca/recorder.py (per-category tallies; keys solving/admin), src/ca/metrics.py
(coordination_overhead=admin share; accuracy_per_ktok_billable → accuracy_per_ktok_solving,
keep _all), src/ca/config.py (nothing new). Update ALL affected tests (billing tests,
e2e balances, metrics tests). Bankrupt gate: permission_error blocks solving when bankrupt.
Commit: `feat(v2): all-action billing with solving/admin labels`

### T18: Loan system
Files: create src/ca/loans.py (Loan dataclass: lid, lender, borrower, principal,
status proposed/active/repaid; LoanSystem(ledger): propose/accept/repay/interest_tick
returning per-loan interest events; capitalize unpaid interest), src/ca/config.py
(loan_rate=0.01), src/ca/scheduler.py (tick at round start; record interest events
in trace via recorder.log synthetic events agent="__world__"? No — log as event
dict {round, agent: borrower, action: "__interest__", ...} appended directly),
src/ca/actions.py (3 actions + specs + handlers + visibility), src/ca/infra.py
(wire LoanSystem). Tests: lifecycle, capitalization when broke, conservation,
repay partial/full, unknown ids.
Commit: `feat(v2): loan system at 1%/round with interest capitalization`

### T19: Seven-level spectrum + credit gating
Files: src/ca/config.py (LEVELS L0..L6, central_credit flag, star level renamed,
interface_turns_per_round), src/ca/actions.py (permission_error: central_credit →
propose_loan lender must be interface / accept path checks; star extends to loans),
src/ca/scheduler.py (interface extra turns if knob >1 — slot list), src/ca/skills.py
(loan demo block gated; credit-central variant), src/ca/context.py (loans owed/held
in render_turn). Update tests incl. matrix test to 7 levels.
Commit: `feat(v2): L0-L6 spectrum with credit centralization`

### T20: Task trees — model, board, actions
Files: create src/ca/tasktree.py (TaskNode: id, sentence, children ids, leaf qids
under it; TaskLibrary: from_json(path)/to_json, resolve(text)->node (id or sentence,
fuzzy), leaves(node_id)->list[qid], children(node_id)); rewrite src/ca/taskboard.py
as task-based board (tasks = posted node ids with price=Σ leaf prices; claim_task,
deliver_task(agent, node, answers: dict)-> (per-leaf scores, total_payout);
TTL/strikes at task level; list open tasks w/ sentence+leafcount+price); src/ca/actions.py
(list_tasks(offset), claim_task(task), decompose(node), deliver_work routing:
target resolves to task node → WORLD packaged path (JSON parse + coverage), contract
→ node-bound coverage check if bound; drop claim_question/list_questions), src/ca/context.py
(active claimed tasks w/ progress: which leaves have scratchpad notes; expiry countdown),
src/ca/runner.py (load TaskLibrary + posted tasks from data dir), src/ca/skills.py
(demos rewritten around decompose→subcontract-subtree→package flow). Question objects
still hold answers/price for grading. Keep q-ids visible only after decompose reveals leaves.
Tests: resolve fuzzy/ambiguous, decompose reveal, packaged delivery happy/missing-qid/
bad-json (attempt not consumed on rejects), per-(task,leaf) repeat pay across two tasks
sharing a leaf, node-bound contract coverage, claim strikes/TTL on tasks.
Commit: `feat(v2): hierarchical task trees with packaged delivery`

### T21: Task library builder
Files: create scripts/build_tasks.py: inputs pool questions (reuse prepare_data
download logic → 120 Q: 90 hotpot + 30 musique) → embed question texts
(chromadb default embedder via a throwaway collection or sentence-transformers);
bottom-up agglomerative grouping with branching ≤3 (recursive: cluster into
ceil(n/3) groups sized ≤3 at each level, depth ≤3 internal levels); LLM
one-sentence summaries per node (gpt-5-mini, temperature default; enforce
uniqueness: if duplicate sentence, ask again with "must differ from: ...");
semantic-locality validation (mean intra-subtree cosine sim > inter-sibling sim;
re-cluster violating nodes once, else warn); select ~30 posted tasks (mix of
levels/sizes incl. some whole trees + some mid nodes, overlapping); price=Σ leaf;
outputs data dir: library.json (nodes), posted.json (task ids), pool.jsonl
(questions w/ prices), corpus → Chroma index. Deterministic given --seed except
LLM summaries (cache them to summaries.json for reproducibility).
Smoke-tests offline pieces (clustering shape, uniqueness enforcement logic) with
stub embeddings/summarizer; LLM path exercised in the live build.
Commit: `feat(v2): task library builder (clustering + sentence summaries)`

### T22: Metrics v2
Files: src/ca/metrics.py + recorder summary: specialization Herfindahl per agent
over base-subtask of answered leaves (needs library at metrics time — summary
stores per-delivery leaf→agent attribution), task completion counts, credit stats
(loans made/active/repaid, interest paid, debt outstanding, bankrupt-with-debt),
admin/solving ratio. Tests with synthetic summaries.
Commit: `feat(v2): metrics — specialization, credit, admin ratio`

### T23: v2 scripted e2e
tests/test_e2e_v2.py: (a) L0 full flow: claim task → decompose → node-bound
subcontract → contractor answers leaves → delivers JSON → payer merges + packages
task delivery → graded payout; conservation. (b) Loan lifecycle w/ interest tick +
capitalization + repay; L4 credit gating (peer loan blocked, interface loan OK).
(c) star L5 extends to loans. Deterministic ScriptedPolicy; tiny 2-level library
fixture built inline.
Commit: `test(v2): end-to-end scripted flows`

### T24: v2 calibration pilot (live, manual gate)
Build real library (data/v2/), run sentinel configs {L0, L5, L6} gpt-5-mini,
30-task board, turns 480; verify economics (no instant mass bankruptcy under
all-action billing — else recalibrate seed capital ×2–3 and/or leaf prices),
task completion nonzero, decompose/subcontract activity. Then report for
main-experiment parameter lock.
