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


# ---------- Fig: the key-coverage gradient over training ----------
# Each curve is one question group examined on eleven frozen snapshots of the
# same run. The reader, the budget and the questions are identical at every
# point; only the store changes, so the curves separate what training bought
# for questions it saw from what it bought for questions it did not.
CURVE = [("exact", "trained on this question", VERM, "o", "-"),
         ("share2", "both keys seen", ORANGE, "s", "-"),
         ("share1", "one key seen", BLUE, "^", "--"),
         ("share0", "neither key seen", "0.45", "v", ":")]
ITERS = [0, 20, 40, 60, 80, 100, 120, 140, 160, 180, 200]

def gradient_curve():
    import os
    fig, ax = plt.subplots(figsize=(3.5, 2.6))
    for split, label, color, marker, ls in CURVE:
        xs, ys = [], []
        for N in ITERS:
            p = f"runs/curve_{N}/test_log.jsonl"
            if not os.path.exists(p):
                continue
            rows = [json.loads(l) for l in open(p)
                    if json.loads(l).get("split") == split]
            if not rows:
                continue
            xs.append(N)
            ys.append(sum(r["steps"] for r in rows) / len(rows))
        if xs:
            ax.plot(xs, ys, marker=marker, ls=ls, color=color, lw=1.4, ms=3.4,
                    label=label)
    ax.set_xlabel("training iteration (frozen snapshot)")
    ax.set_ylabel("actions to answer")
    ax.set_xticks([0, 50, 100, 150, 200])
    ax.legend(frameon=False, fontsize=6.8, loc="lower left")
    save(fig, "learning")

gradient_curve()


# ---------- Fig: store trajectory + final out-degree ----------
ser = [json.loads(l) for l in open(RUN / "series.jsonl")]
it = list(range(len(ser)))
links = [s["n_links"] for s in ser]
authored = [s["authored_statements"] for s in ser]
nodes_added = [s["n_nodes"] - ser[0]["n_nodes"] for s in ser]

fig, (axa, axb) = plt.subplots(1, 2, figsize=(6.9, 2.9))
axa.plot(it, links, color=GREEN, lw=1.4, label="links")
axa.plot(it, authored, color=VERM, lw=1.4, label="index documents")
axa.plot(it, nodes_added, color=BLUE, lw=1.2, ls=":", label="net documents added")
axa.axvline(100.5, color="0.6", ls="--", lw=0.8)
axa.set_xlabel("training iteration")
axa.set_ylabel("count")
axa.legend(frameon=False, fontsize=8.5, loc="upper left")
axa.set_title("(a) what the store accumulates", fontsize=10)

# (b) out-degree of every authored index in the final store
final = json.load(open("/tmp/kbstate_200.json"))["store"]
degs = sorted(len(n.get("links", [])) for n in final["nodes"]
              if n.get("flag") == "authored")
import statistics
axb.hist(degs, bins=range(0, max(degs) + 3), color=VERM, alpha=0.85, lw=0)
axb.axvline(statistics.median(degs), color=BLUE, ls="--", lw=1.0)
axb.text(statistics.median(degs) + 0.7, axb.get_ylim()[1] * 0.86,
         f"median {statistics.median(degs):.0f}", fontsize=8.5, color=BLUE)
n_empty = sum(1 for d in degs if d == 0)
axb.text(0.97, 0.72, f"{len(degs)} indexes\nmean {statistics.mean(degs):.1f}, "
         f"max {max(degs)}\n{n_empty} empty ({n_empty/len(degs):.0%})",
         transform=axb.transAxes, ha="right", va="top", fontsize=7,
         color="0.25")
axb.set_xlabel("out-degree of an index document")
axb.set_ylabel("index documents")
axb.set_title("(b) final out-degree distribution", fontsize=10)
save(fig, "trajectory")


# ---------- Fig: network evolution + index anatomy ----------
def _draw(ax, nodes, pos, edge_lw=0.30, dot=0.5):
    deg = {}
    for nd in nodes:
        for t in nd.get("links", []):
            deg[nd["id"]] = deg.get(nd["id"], 0) + 1
            deg[t] = deg.get(t, 0) + 1
    for nd in nodes:
        x0, y0 = pos[nd["id"]]
        for t in nd.get("links", []):
            if t in pos:
                ax.plot([x0, pos[t][0]], [y0, pos[t][1]], color=GREEN,
                        lw=edge_lw, alpha=0.45, zorder=1)
    orig = [n for n in nodes if n.get("flag") != "authored"]
    auth = [n for n in nodes if n.get("flag") == "authored"]
    ax.scatter([pos[n["id"]][0] for n in orig], [pos[n["id"]][1] for n in orig],
               s=dot, color=BLUE, alpha=0.35, lw=0, zorder=2)
    if auth:
        ax.scatter([pos[n["id"]][0] for n in auth],
                   [pos[n["id"]][1] for n in auth],
                   s=[3 + 1.2 * deg.get(n["id"], 0) for n in auth],
                   color=VERM, lw=0, zorder=3)
    ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    for sp in ax.spines.values():
        sp.set_visible(True); sp.set_color("0.85"); sp.set_linewidth(0.6)
    return len(auth)


def _load(N, pos):
    st = json.load(open(f"/tmp/kbstate_{N}.json"))["store"]["nodes"]
    return [nd for nd in (st.values() if isinstance(st, dict) else st)
            if nd["id"] in pos]


def network_evolution():
    pos = json.load(open("/tmp/kbnet_pos.json"))
    annot = json.load(open("/tmp/kbnet_annot.json"))
    fig = plt.figure(figsize=(7.0, 5.2))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.0, 1.32],
                          hspace=0.16, wspace=0.05)

    slots = [(gs[0, 0], 0), (gs[0, 1], 50), (gs[0, 2], 100), (gs[1, 0], 150)]
    for cell, N in slots:
        ax = fig.add_subplot(cell)
        nodes = _load(N, pos)
        n_auth = _draw(ax, nodes, pos)
        nlinks = sum(len(nd.get("links", [])) for nd in nodes)
        ax.set_title(f"iter {N}: {nlinks} links, {n_auth} indexes", fontsize=6.5)

    # the final store gets two thirds of the bottom row, with four verified
    # indexes labelled and their fan-out drawn
    ax = fig.add_subplot(gs[1, 1:])
    nodes = _load(200, pos)
    n_auth = _draw(ax, nodes, pos, edge_lw=0.22, dot=0.45)
    nlinks = sum(len(nd.get("links", [])) for nd in nodes)
    corners = [(0.02, 0.97, "left", "top"), (0.98, 0.97, "right", "top"),
               (0.02, 0.04, "left", "bottom"), (0.98, 0.04, "right", "bottom")]
    for (text, a), (cx, cy, ha, va) in zip(annot.items(), corners):
        if a["id"] not in pos:
            continue
        x, y = pos[a["id"]]
        for t in a["targets"]:
            if t in pos:
                ax.plot([x, pos[t][0]], [y, pos[t][1]], color=PURPLE,
                        lw=0.85, alpha=1.0, zorder=5, solid_capstyle="round")
        ax.scatter([x], [y], s=52, facecolor="none", edgecolor=PURPLE,
                   lw=1.2, zorder=7)
        ax.annotate(f"{text}\n{a['kind']}, {a['note']}",
                    xy=(x, y), xycoords="data",
                    xytext=(cx, cy), textcoords="axes fraction",
                    fontsize=5.6, color="0.12", ha=ha, va=va, zorder=8,
                    bbox=dict(boxstyle="round,pad=0.25", fc="white",
                              ec=PURPLE, lw=0.55, alpha=0.95),
                    arrowprops=dict(arrowstyle="-", color=PURPLE, lw=0.55,
                                    alpha=0.8, shrinkA=2, shrinkB=5))
    ax.set_title(f"iter 200: {nlinks} links, {n_auth} indexes", fontsize=6.5)

    fig.legend(handles=[
        plt.Line2D([], [], marker="o", ls="", color=BLUE, ms=3,
                   label="document (position = semantic embedding, t-SNE)"),
        plt.Line2D([], [], marker="o", ls="", color=VERM, ms=4,
                   label="authored index document (size = degree)"),
        plt.Line2D([], [], color=GREEN, lw=1, label="link"),
        plt.Line2D([], [], color=PURPLE, lw=1.2,
                   label="fan-out of a labelled index")],
        loc="lower center", ncol=2, frameon=False, fontsize=6.8,
        bbox_to_anchor=(0.5, -0.035))
    fig.subplots_adjust(bottom=0.11, top=0.945)
    save(fig, "network")


network_evolution()
