"""Generate paper figures from run data. Okabe-Ito palette, PDF + PNG."""
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BLUE, ORANGE, VERM, GREEN, PURPLE = "#0072B2", "#E69F00", "#D55E00", "#009E73", "#9A5EA8"
RUN = Path("runs/v10L_dedup")
OUT = Path("paper/figs")
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.5,
    "figure.dpi": 150, "savefig.bbox": "tight",
})


def save(fig, name):
    fig.savefig(OUT / f"{name}.pdf")
    fig.savefig(OUT / f"{name}.png", dpi=200)
    print("wrote", OUT / f"{name}.pdf")


# ---------- Fig: learning (eval curve + rolling forward F1) ----------
rows = [json.loads(l) for l in open(RUN / "test_log.jsonl")]
by = defaultdict(list)
for r in rows:
    by[(r["epoch"], r["flavor"])].append(r["f1"])
epochs = sorted({e for e, _ in by})
ev_in = [sum(by[(e, "in")]) / len(by[(e, "in")]) for e in epochs]
ev_out = [sum(by[(e, "out")]) / len(by[(e, "out")]) for e in epochs]

tl = [json.loads(l) for l in open(RUN / "train_log.jsonl")]
f1s = [r["f1"] for r in tl]
W = 30
roll = [sum(f1s[max(0, i - W + 1):i + 1]) / len(f1s[max(0, i - W + 1):i + 1])
        for i in range(len(f1s))]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.9, 2.5))
ax1.plot(epochs, ev_in, "o-", color=BLUE, label="eval-in (trained templates)")
ax1.plot(epochs, ev_out, "s-", color=VERM, label="eval-out (reserved templates)")
ax1.set_xticks(epochs)
ax1.set_xlabel("epoch")
ax1.set_ylabel("eval F1")
ax1.set_ylim(0, 1.0)
ax1.legend(frameon=False, fontsize=7.5)
ax1.set_title("(a) held-out eval by epoch", fontsize=9)

ax2.plot(range(1, len(roll) + 1), roll, color=GREEN, lw=1.4)
ax2.axvline(150.5, color="0.6", ls="--", lw=0.8)
ax2.text(152, 0.08, "epoch 2\n(re-encountered\nquestions)", fontsize=7, color="0.35")
ax2.set_xlabel("training iteration")
ax2.set_ylabel(f"forward F1 (rolling {W})")
ax2.set_ylim(0, 1.0)
ax2.set_title("(b) train-forward accuracy", fontsize=9)
save(fig, "learning")

# ---------- Fig: store trajectory ----------
ser = [json.loads(l) for l in open(RUN / "series.jsonl")]
it = list(range(len(ser)))
links = [s["n_links"] for s in ser]
authored = [s["authored_statements"] for s in ser]
nodes_added = [s["n_nodes"] - ser[0]["n_nodes"] for s in ser]

fig, ax = plt.subplots(figsize=(3.4, 2.4))
ax.plot(it, links, color=GREEN, lw=1.4, label="links")
ax.plot(it, authored, color=VERM, lw=1.4, label="authored documents")
ax.plot(it, nodes_added, color=BLUE, lw=1.2, ls=":", label="net documents added")
ax.axvline(150.5, color="0.6", ls="--", lw=0.8)
ax.set_xlabel("training iteration")
ax.set_ylabel("count")
ax.legend(frameon=False, fontsize=7.5, loc="upper left")
save(fig, "trajectory")
