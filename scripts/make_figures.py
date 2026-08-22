"""Generate paper figures from run data. Okabe-Ito palette, PDF + PNG."""
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BLUE, ORANGE, VERM, GREEN, PURPLE = "#0072B2", "#E69F00", "#D55E00", "#009E73", "#9A5EA8"
RUN = Path("runs/v11_main")
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
roll = [sum(f1s[i - W + 1:i + 1]) / W for i in range(W - 1, len(f1s))]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.9, 2.5))
ax1.plot(epochs, ev_in, "o-", color=BLUE, label="eval-in (trained templates)")
ax1.plot(epochs, ev_out, "s-", color=VERM, label="eval-out (reserved templates)")
ax1.set_xticks(epochs)
ax1.set_xlabel("epoch")
ax1.set_ylabel("eval F1")
ax1.set_ylim(0, 1.0)
ax1.legend(frameon=False, fontsize=7.5)
ax1.set_title("(a) held-out eval by epoch", fontsize=9)

ax2.plot(range(W, W + len(roll)), roll, color=GREEN, lw=1.4)
ax2.axvline(100.5, color="0.6", ls="--", lw=0.8)
ax2.text(102, 0.08, "epoch 2\n(re-encountered\nquestions)", fontsize=7, color="0.35")
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
ax.axvline(100.5, color="0.6", ls="--", lw=0.8)
ax.set_xlabel("training iteration")
ax.set_ylabel("count")
ax.legend(frameon=False, fontsize=7.5, loc="upper left")
save(fig, "trajectory")


# ---------- Fig: network evolution (semantic layout, all nodes) ----------
def network_evolution():
    pos = json.load(open("/tmp/kbnet_pos.json"))
    CHECKS = (0, 40, 80, 120, 160, 200)
    fig, axes = plt.subplots(2, 3, figsize=(7.0, 4.6))
    for ax, N in zip(axes.flat, CHECKS):
        st = json.load(open(f"/tmp/kbstate_{N}.json"))["store"]["nodes"]
        nodes = list(st.values()) if isinstance(st, dict) else st
        nodes = [nd for nd in nodes if nd["id"] in pos]
        deg = {}
        for nd in nodes:
            for t in nd.get("links", []):
                deg[nd["id"]] = deg.get(nd["id"], 0) + 1
                deg[t] = deg.get(t, 0) + 1
        # edges under nodes
        for nd in nodes:
            x0, y0 = pos[nd["id"]]
            for t in nd.get("links", []):
                if t in pos:
                    x1, y1 = pos[t]
                    ax.plot([x0, x1], [y0, y1], color=GREEN, lw=0.35,
                            alpha=0.5, zorder=1)
        orig = [nd for nd in nodes if nd.get("flag") != "authored"]
        auth = [nd for nd in nodes if nd.get("flag") == "authored"]
        ax.scatter([pos[n["id"]][0] for n in orig], [pos[n["id"]][1] for n in orig],
                   s=0.6, color=BLUE, alpha=0.35, lw=0, zorder=2)
        if auth:
            ax.scatter([pos[n["id"]][0] for n in auth],
                       [pos[n["id"]][1] for n in auth],
                       s=[4 + 1.5 * deg.get(n["id"], 0) for n in auth],
                       color=VERM, lw=0, zorder=3)
        nlinks = sum(len(nd.get("links", [])) for nd in nodes)
        ax.set_title(f"iteration {N}\n{nlinks} links · {len(auth)} nav docs",
                     fontsize=7.5)
        ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
        for sp in ax.spines.values():
            sp.set_visible(True); sp.set_color("0.85"); sp.set_linewidth(0.6)
    fig.legend(handles=[
        plt.Line2D([], [], marker="o", ls="", color=BLUE, ms=3, label="document (position = semantic embedding, t-SNE)"),
        plt.Line2D([], [], marker="o", ls="", color=VERM, ms=4, label="authored navigation document (size = degree)"),
        plt.Line2D([], [], color=GREEN, lw=1, label="link")],
        loc="lower center", ncol=3, frameon=False, fontsize=7,
        bbox_to_anchor=(0.5, -0.02))
    fig.subplots_adjust(bottom=0.09, top=0.90, hspace=0.34, wspace=0.06)
    save(fig, "network")


network_evolution()
