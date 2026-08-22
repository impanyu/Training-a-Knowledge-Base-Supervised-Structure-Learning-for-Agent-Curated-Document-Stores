# v8 — Closed-loop agent–environment dynamics

New paper direction. The object of study is the dynamical system induced by
an LLM acting on a closed environment: state → prompt → sampled action →
deterministic transition → state. No short-term memory, no history in the
prompt: the loop is exactly a Markov chain on the environment's state space,
with the LLM as its transition kernel. We characterize that chain — fixed
points, cycles, attractors, mixing — first exactly (by censusing the kernel
state-by-state), then empirically (closed-loop rollouts).

Phase A (this spec): single agent, global observation, small discrete space.
Phase B (design only, §8): distributed agents on local state.

The v7 `src/ca` package is untouched. v8 is a sibling package `src/dyn` with
its own tests and CLIs. Only `ca.providers`-style LLM plumbing concepts are
reused (reimplemented locally; no import from `ca`).

## 1. Environments

An environment is: a state space S (k binary sites), an action set
A = {flip(site_i)} ∪ {wait}, a deterministic `apply(s, a) → s'`, and a
deterministic `render(s) → str` (canonical site order, so one state maps to
exactly one prompt string).

- **`bits-k`** — semantically bleached control. Sites are switches
  "X1".."Xk" with values 0/1. Render: `X1=0  X2=1 ...`.
- **`room-k`** — the SAME state lattice wearing semantics. Fixed item list
  (first k used, k ≤ 8): door open/closed, lamp on/off, window open/shut,
  plant watered/dry, book open/closed, tap running/off, bed made/unmade,
  clock running/stopped. Render: "The door is open. The lamp is off. ..."
  Action names are semantic toggles ("toggle the lamp"), positionally
  isomorphic to bits-k's flips.

Same |S| = 2^k, same |A| = k+1 → every analysis is directly comparable
across the semantic/abstract pair; the DIFFERENCE between their kernels is
the measured effect of meaning.

## 2. The kernel

`Kernel.sample(state) → action`: one LLM call. Prompt = framing preamble +
render(state) + "choose exactly one action". The action is forced through a
single tool `act` whose parameter is an enum over A (tool_choice required).
A malformed/failed call records the pseudo-action `invalid` (it is part of
the kernel; `apply` treats it as wait). Framings (factor `--framing`):

- `neutral` — "You observe a small system of toggles. You may change it or
  leave it. There is no goal."
- `inhabit` — "This is your room. You live here. Do as you see fit."

Temperature is a CLI factor. Nothing else varies. No memory, no history.

## 3. Census mode (`python -m dyn.census`)

For EVERY state s in S (k ≤ 6 → ≤ 64 states), draw `--samples` M actions →
empirical π(a|s). Output `census.json`: {env, k, framing, temperature,
model, samples, counts[s][a], a few raw completions per state for
debugging}. Resumable: skips states already counted in an existing output
(append-safe). Cost: |S|·M calls.

## 4. Analysis (`python -m dyn.analyze`, offline, numpy only)

From census.json build P(s'|s) = Σ_a π(a|s)·[apply(s,a)=s'] and report
`chain.json` + printed summary:

- **Fixed points / recurrent structure**: absorbing states; recurrent
  classes via SCC of the support graph (support = probability > eps).
- **Greedy skeleton**: the deterministic map s ↦ apply(s, argmax π(·|s));
  its functional graph's cycle inventory (cycle states, lengths, basin
  sizes) — this is the "limit cycle" answer in the deterministic limit.
- **Stationary distribution** μ (power iteration), its entropy, and the
  top-10 states by mass.
- **Spectrum**: eigenvalues of P; spectral gap 1−|λ2| (mixing speed);
  complex eigenvalues near the unit circle (quasi-cycles: report period
  2π/arg λ for |λ| > 0.9).
- **Kernel statistics**: per-state action entropy H(π(·|s)) (mean/min/max),
  wait mass, invalid mass, per-site flip asymmetry (P(flip i | site=0) vs
  site=1 — the "which way does the LLM push each bit" field), and a
  drift/potential summary: expected Hamming motion toward all-0 / all-1 /
  the semantically "orderly" state (room-k declares an orderly
  configuration: door closed, lamp off, window shut, plant watered, book
  closed, tap off, bed made, clock running).
- **Cross-census comparison** (`--compare a.json b.json`): total-variation
  distance between kernels state-by-state (bits vs room at matched k =
  the semantics effect; neutral vs inhabit = the framing effect).

## 5. Rollout mode (`python -m dyn.rollout`)

Closed-loop trajectories: `--episodes E --steps L --init random|all0|all1`.
Each step: sample kernel once, apply, log. `rollouts.jsonl`: one line per
step {episode, t, state, action, state_next}. Analysis (in `dyn.analyze
--rollouts`): first-recurrence time per episode, terminal behaviour
(absorbed at fixed point / cycling with detected period / still wandering),
visited-state entropy over sliding windows (does the trajectory's entropy
production decay?), and — when a census for the same config exists —
agreement checks (per-step log-likelihood of observed actions under the
censused kernel).

## 6. Package layout & tests

```
src/dyn/env.py       BitsEnv/RoomEnv (shared lattice core), apply/render/orderly
src/dyn/kernel.py    LLM kernel (OpenAI client, tool-enum forcing, retries)
src/dyn/census.py    census CLI (resumable)
src/dyn/rollout.py   rollout CLI
src/dyn/markov.py    P construction + all §4 analyses (pure numpy)
src/dyn/analyze.py   analyze CLI (census/rollouts/compare)
tests/dyn/           offline only: a StubKernel (callable π table) replaces
                     the LLM everywhere; no network in tests
```

Test intents: render determinism & bijective state↔index codec; apply
correctness incl. wait/invalid; bits/room isomorphism (same lattice, mapped
action names); census resume (partial file → completed without re-sampling
counted states); markov: hand-built kernels with known answers — an
absorbing kernel (μ mass 1, gap, basin), a pure 4-cycle kernel (complex λ
period ≈ 4, greedy cycle inventory), a uniform kernel (μ uniform, max
entropy); rollout logging + first-recurrence; analyze end-to-end on stub
census/rollout files; compare TV distance on known kernels.

## 7. Live smoke (after implementation, by the orchestrator, not in tests)

census bits-3 (8 states × 12 samples) + census room-3 + one rollout room-5
(1 episode × 30 steps), gpt-5-mini, temperature 1.0, neutral framing.
Purpose: plumbing + first look at wait/invalid mass and flip asymmetry.

## 8. Phase B design sketch (not implemented in this task)

Ring of k sites, one agent per site. Agent i's prompt renders only the
window (i−w..i+w); its action set = {flip own site, wait}. Synchronous
update: all agents sample on the same tick against the same global state,
then all flips apply at once (a stochastic cellular automaton whose local
rule is the LLM). Outputs: space–time diagrams, density/magnetization
trajectories, domain-size distributions, a synchronization order parameter,
and the same census idea applied to the LOCAL kernel: π(a | window) over
all 2^(2w+1) windows — small enough to census exactly, which turns the
whole lattice system into an analyzable probabilistic CA. Key comparisons:
window size w (how much locality changes global order), semantic vs
abstract skins, and homogeneous vs mixed framings.
