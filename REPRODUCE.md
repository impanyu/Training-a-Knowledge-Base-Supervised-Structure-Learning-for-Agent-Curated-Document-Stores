# Reproducing the paper

`data/` and `runs/` are not committed: the first is regenerable, the second
is ~144 MB of LLM output. This file says exactly which numbers you can
reproduce from this repository alone, which need a training run, and which
cannot be reproduced at all.

## What regenerates exactly

**The KBGym universe.** Deterministic given its seed — verified
byte-identical, not merely equivalent:

```bash
PYTHONPATH=src python3 -m kb.build --seed 0 --people 500 \
  --train 150 --test-in 100 --test-out 50 --eval-in 20 --eval-out 10 \
  --distractors 0.1 --out data/v10L
```

**The offline baseline stores.** Both extract lexically, so both are
deterministic. B2 spends 488k LLM tokens on its community summaries; B3
spends none.

```bash
PYTHONPATH=src python3 -m kb.baseline_graphrag --universe data/v10L/universe.json --out data/v10L_graphrag
PYTHONPATH=src python3 -m kb.baseline_hipporag --universe data/v10L/universe.json --out data/v10L_hipporag
```

**The key-coverage probe** (Table V, Fig. 2). Note the dependency: the probe
groups questions by what the *training run* touched, so it needs that run's
`train_log.jsonl` and cannot be built from the repository alone.

```bash
PYTHONPATH=src python3 scripts/make_gradient_probe.py --per-group 30 \
  --arms b1=data/v10L/universe.json \
         graphrag=data/v10L_graphrag/universe.json \
         hipporag=data/v10L_hipporag/universe.json
```

## What needs a training run

```bash
PYTHONPATH=src python3 -m kb.train --universe data/v10L/universe.json \
  --epochs 2 --train-size 100 --out runs/v11_main --model gpt-5-mini-2025-08-07
```

200 iterations, 18.8M tokens, 6.8 h, about \$13. Then the exams — the four
probe arms, the six epoch-1 snapshots, and the test splits:

```bash
for a in b1 graphrag hipporag; do
  PYTHONPATH=src python3 -m kb.test --kb data/grad_$a/universe.json \
    --split exact,share2,share1,share0 --limit 30 --out runs/grad_$a --m 15
done
PYTHONPATH=src python3 -m kb.test --kb runs/v11_main/kb_epoch_2.json \
  --universe data/grad_b1/universe.json --split exact,share2,share1,share0 \
  --limit 30 --out runs/grad_trained --m 15

for N in 0 20 40 60 80 100; do
  PYTHONPATH=src python3 -m kb.replay --universe data/v10L/universe.json \
    --trace runs/v11_main/trace.jsonl --at $N --out runs/v11_main/snaps/kb_$N.json
  PYTHONPATH=src python3 -m kb.test --kb runs/v11_main/snaps/kb_$N.json \
    --universe data/grad_b1/universe.json --split exact,share2,share1,share0 \
    --limit 30 --out runs/curve_$N --m 15 --epoch $N
done
```

Pin the snapshot, not the `gpt-5-mini` alias: the alias moves, and the
reported numbers are from `gpt-5-mini-2025-08-07`.

## What will not reproduce

**The exact numbers.** The reader is an LLM at temperature 0.3, so a re-run
gives a different sample. Expect the same ordering and the same shape of the
gradient; do not expect $\rho = 0.686$ to the digit.

**The PhantomWiki arm.** Its universe came from a `phantom-wiki` 1.0.3
generation whose output directory no longer exists. `phantom-wiki` installs
into a fresh venv (the system Python is PEP-668 managed) and `swipl` is
required, but we have **not** verified that re-generating with `seed=1`
reproduces the same 3,403 statement nodes. Until that is checked, treat the
PhantomWiki results as reproducible in method but not in data.

## Analysis

```bash
python3 scripts/gradient_table.py                                   # Table V
python3 scripts/index_quality_table.py --kb runs/v11_main/kb_epoch_2.json
python3 scripts/index_precision2.py --kb runs/v11_main/kb_epoch_2.json --show 20
PYTHONPATH=src python3 scripts/make_network_layout.py               # t-SNE + annotations
PYTHONPATH=src python3 scripts/make_figures.py                      # paper/figs/*.pdf
PYTHONPATH=src python3 scripts/trajectory_diff.py --group exact     # reader traces, side by side
```

Use `index_precision2.py`, not `index_precision.py`. The older script tests
whether a linked document mentions the index's key, which is correct for
attribute indexes and wrong for relational ones — the grandchildren of a
person never name the grandparent, so a correct two-hop index scores zero.
The v2 scorer resolves the key's true member set from the universe instead.
Both are kept so the discrepancy stays inspectable.

## Rebuilding a store offline

Any intermediate state replays from the trace with no API call, including
merge verdicts, which are recorded rather than recomputed:

```bash
PYTHONPATH=src python3 -m kb.replay --universe data/v10L/universe.json \
  --trace runs/v11_main/trace.jsonl --at 100 --out /tmp/kb_100.json
```

Validated byte-exact against the epoch snapshots.

## The data itself

`data/` and `runs/` as used for the paper are archived outside git. See the
`README.md` in that archive for the directory map.
