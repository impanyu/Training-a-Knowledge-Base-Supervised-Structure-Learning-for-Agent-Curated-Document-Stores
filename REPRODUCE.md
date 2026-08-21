# Reproducing the runs

No data is committed. Everything under `data/` and `runs/` is generated, and
the generators are deterministic given their seed, so the repository stays
small and there is no risk of a stale copy diverging from the code that made
it. The commands below rebuild every input from scratch.

## The universe

The 500-person KBGym universe used in the paper (5,864 atomic documents, 330
questions, splits 150/100/50/30):

```bash
PYTHONPATH=src python3 -m kb.build --seed 0 --people 500 \
  --train 150 --test-in 100 --test-out 50 --eval-in 20 --eval-out 10 \
  --distractors 0.1 --out data/v10L
```

This is byte-identical on every run; the parameters are also recorded in the
`meta` block of the file it writes, so a universe can always be traced back
to the command that produced it.

## Baseline stores

Each adapts the same universe into the store format the common reader uses,
so the arms differ only in how the store was prepared.

```bash
# B3, GraphRAG-style: entity communities + one LLM summary per community
PYTHONPATH=src python3 -m kb.baseline_graphrag \
  --universe data/v10L/universe.json --out data/v10L_graphrag

# B5, HippoRAG2-style: per-entity index nodes + passage-graph edges (no LLM)
PYTHONPATH=src python3 -m kb.baseline_hipporag \
  --universe data/v10L/universe.json --out data/v10L_hipporag

# Oracle access layer: the structure training is trying to discover, built
# directly from the universe, as a ceiling rather than a competitor
python3 scripts/build_oracle_index.py --out data/v10L_oracle
```

## Training and evaluation

```bash
bash scripts/v11_full.sh        # 2 epochs x 150 questions, then every exam
```

All calls go to the `gpt-5-mini` alias, which resolved to
`gpt-5-mini-2025-08-07` for the reported runs. The alias moves; pin the
snapshot with `--model gpt-5-mini-2025-08-07` to reproduce those numbers
rather than whatever the alias points at when you read this.

The script trains the store, runs the frozen-store exams at M = 15 and M = 8
and on the train split, re-measures all baselines under the same reader, and
writes the structural analyses. `OPENAI_API_KEY` (and `OPENAI_BASE_URL` if
you use a proxy) come from a `.env` file in the repository root.

## Analysis

```bash
python3 scripts/results_table.py                 # accuracy, steps, break-even
python3 scripts/note_character.py --kb runs/v11_main/kb_epoch_2.json
python3 scripts/index_precision.py --kb runs/v11_main/kb_epoch_2.json
python3 scripts/index_taxonomy.py  --kb runs/v11_main/kb_epoch_2.json
python3 scripts/make_figures.py                  # paper/figs/*.pdf
```

Any intermediate store state can be rebuilt from a run's trace without
spending an API call:

```bash
PYTHONPATH=src python3 -m kb.replay --universe data/v10L/universe.json \
  --trace runs/v11_main/trace.jsonl --at 150 --embed --out /tmp/kb_at_150.json
```
