"""PhantomWiki mirrors of the paper figures: learning curves and trajectory."""
import json, statistics
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BLUE, ORANGE, VERM, GREEN = "#0072B2", "#E69F00", "#D55E00", "#009E73"
RUN = Path("runs/pw1_main")
OUT = Path("paper/figs")
plt.rcParams.update({
    "font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.5,
    "figure.dpi": 150, "savefig.bbox": "tight",
})

def save(fig, name):
    fig.savefig(OUT / f"{name}.pdf")
    fig.savefig(OUT / f"{name}.png", dpi=200)
    print("wrote", OUT / f"{name}.pdf")

ITERS = [0, 30, 60, 90, 120, 150]
CURVE = [("exact", "trained on this question", VERM, "o", "-"),
         ("share1", "one key seen", BLUE, "^", "--"),
         ("share0", "neither key seen", "0.45", "v", ":")]
MIN_ROWS = 96

def _coverage(N):
    st = json.load(open(RUN / f"snaps/kb_{N}.json"))["store"]
    seeds = set(st["origins"])
    reach = {t for n in st["nodes"] if n["id"] not in seeds
             for t in (n.get("links") or []) if t in seeds}
    return 100 * len(reach) / len(seeds)

def learning():
    fig, axes = plt.subplots(2, 2, figsize=(6.9, 3.9))
    (a_s, a_f), (b_s, b_f) = axes
    for split, label, color, marker, ls in CURVE:
        xs, ys, fs = [], [], []
        for N in ITERS:
            rows = [json.loads(l) for l in open(f"runs/pwgrad_curve_{N}/test_log.jsonl")]
            if len(rows) < MIN_ROWS: continue
            rs = [r for r in rows if r.get("split") == split]
            xs.append(N)
            ys.append(sum(r["steps"] for r in rs) / len(rs))
            fs.append(sum(r["f1"] for r in rs) / len(rs))
        for ax, vals in ((a_s, ys), (a_f, fs)):
            ax.plot(xs, vals, marker=marker, ls=ls, color=color, lw=1.4,
                    ms=3.4, label=label)
    xs, ys, fs = [], [], []
    for N in ITERS:
        rows = [json.loads(l) for l in open(f"runs/pwgrad_curve_{N}/test_log.jsonl")]
        if len(rows) < MIN_ROWS: continue
        xs.append(_coverage(N))
        ys.append(sum(r["steps"] for r in rows) / len(rows))
        fs.append(sum(r["f1"] for r in rows) / len(rows))
    fin = [json.loads(l) for l in open("runs/pwgrad_trained/test_log.jsonl")]
    xs.append(12.3)
    ys.append(sum(r["steps"] for r in fin) / len(fin))
    fs.append(sum(r["f1"] for r in fin) / len(fin))
    for ax, vals in ((b_s, ys), (b_f, fs)):
        ax.plot(xs, vals, "o-", color=VERM, lw=1.6, ms=4, label="ours")
        ax.set_xlim(-0.8, 13.5)
    for ax in (a_s, a_f):
        ax.set_xlabel("training iteration (frozen snapshot)")
        ax.set_xticks(ITERS)
    for ax in (b_s, b_f):
        ax.set_xlabel("corpus reachable from an index (%)")
    a_s.set_ylabel("actions to answer"); b_s.set_ylabel("actions to answer")
    a_f.set_ylabel("F1"); b_f.set_ylabel("F1")
    a_s.set_title("(a) cost, by question group", fontsize=8)
    a_f.set_title("(b) accuracy, by question group", fontsize=8)
    b_s.set_title("(c) cost vs coverage, pooled", fontsize=8)
    b_f.set_title("(d) accuracy vs coverage, pooled", fontsize=8)
    a_s.legend(frameon=False, fontsize=6.4, loc="lower left")
    fig.subplots_adjust(hspace=0.62, wspace=0.26)
    save(fig, "pw_learning")

def trajectory():
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
    axa.set_xlabel("training iteration"); axa.set_ylabel("count")
    axa.legend(frameon=False, fontsize=8, loc="upper left")
    axa.set_title("(a) what the store accumulates", fontsize=10)
    final = json.load(open(RUN / "kb_epoch_2.json"))["store"]
    degs = sorted(len(n.get("links", [])) for n in final["nodes"]
                  if n.get("flag") == "authored")
    axb.hist(degs, bins=range(0, max(degs) + 3), color=VERM, alpha=0.85, lw=0)
    axb.axvline(statistics.median(degs), color=BLUE, ls="--", lw=1.0)
    axb.text(statistics.median(degs) + 0.7, axb.get_ylim()[1] * 0.86,
             f"median {statistics.median(degs):.0f}", fontsize=8, color=BLUE)
    n_empty = sum(1 for d in degs if d == 0)
    axb.text(0.97, 0.72, f"{len(degs)} indexes\nmean {statistics.mean(degs):.1f}, "
             f"max {max(degs)}\n{n_empty} empty ({n_empty/len(degs):.0%})",
             transform=axb.transAxes, ha="right", va="top", fontsize=7, color="0.25")
    axb.set_xlabel("out-degree of an index document")
    axb.set_ylabel("index documents")
    axb.set_title("(b) final out-degree distribution", fontsize=10)
    save(fig, "pw_trajectory")

learning()
trajectory()

def network():
    import json as _json
    pos = {k: v for k, v in _json.load(open("/tmp/pwnet_pos.json")).items()}
    ann = _json.load(open("/tmp/pwnet_annot.json"))
    PURPLE = "#9A5EA8"
    snaps = [("iteration 0", "runs/pw1_main/snaps/kb_0.json"),
             ("end of epoch 1 (iter 150)", "runs/pw1_main/snaps/kb_150.json"),
             ("final store (iter 200)", "runs/pw1_main/kb_epoch_2.json")]
    fig, axes = plt.subplots(1, 3, figsize=(9.6, 3.4))
    for ax, (title, path) in zip(axes, snaps):
        st = json.load(open(path))["store"]
        nodes = st["nodes"]
        deg = {}
        for nd in nodes:
            for t in nd.get("links", []):
                deg[nd["id"]] = deg.get(nd["id"], 0) + 1
        for nd in nodes:
            if nd["id"] not in pos: continue
            x0, y0 = pos[nd["id"]]
            for t in nd.get("links", []):
                if t in pos:
                    ax.plot([x0, pos[t][0]], [y0, pos[t][1]], color=GREEN,
                            lw=0.3, alpha=0.45, zorder=1)
        orig = [n for n in nodes if n.get("flag") != "authored" and n["id"] in pos]
        auth = [n for n in nodes if n.get("flag") == "authored" and n["id"] in pos]
        ax.scatter([pos[n["id"]][0] for n in orig], [pos[n["id"]][1] for n in orig],
                   s=0.5, color=BLUE, alpha=0.35, lw=0, zorder=2)
        if auth:
            ax.scatter([pos[n["id"]][0] for n in auth],
                       [pos[n["id"]][1] for n in auth],
                       s=[3 + 1.2 * deg.get(n["id"], 0) for n in auth],
                       color=VERM, lw=0, zorder=3)
        ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
        for sp in ax.spines.values(): sp.set_visible(True); sp.set_color("0.8")
        ax.set_title(title, fontsize=8)
    # annotate verified indexes on the final panel
    ax = axes[2]
    for nid, meta in ann.items():
        if nid not in pos: continue
        x, y = pos[nid]
        for t in meta["links"]:
            if t in pos:
                ax.scatter([pos[t][0]], [pos[t][1]], s=4, color=PURPLE, zorder=4)
        ax.annotate(meta["text"], (x, y), fontsize=5.5, color="0.1",
                    xytext=(x, y + 6), ha="center",
                    arrowprops=dict(arrowstyle="-", lw=0.5, color="0.3"))
    fig.subplots_adjust(wspace=0.05)
    save(fig, "pw_network")
