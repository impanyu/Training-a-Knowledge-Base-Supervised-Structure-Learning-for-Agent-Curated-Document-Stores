"""Report over a run directory's logs (spec §8/§11: E1/E2/E3/E5).

Emits report.json + a printed plain-text table. Headline F1 excludes QC5
(unanswerable) questions, whose accuracy is reported separately, never
blended. Train-forward F1 is valid from epoch 2 (epoch 1 trains on a store
nothing has consolidated yet). Link-vs-support alignment = share of built
links connecting docs that co-occur in some question's support set."""
import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

from kb.build import Universe


def _rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def _mean(xs) -> float:
    xs = list(xs)
    return sum(xs) / len(xs) if xs else 0.0


def _score_block(rows: list[dict]) -> dict:
    head = [r for r in rows if r["category"] != "QC5"]
    qc5 = [r for r in rows if r["category"] == "QC5"]
    block = {"n": len(rows),
             "f1": _mean(r["f1"] for r in head),
             "em": _mean(r["em"] for r in head),
             "unanswerable_acc": _mean(r["em"] for r in qc5) if qc5 else None}
    if rows and "status" in rows[0]:
        block["unanswered_rate"] = _mean(r["status"] == "unanswered" for r in rows)
        block["wrong_rate"] = _mean(r["status"] == "wrong" for r in rows)
    return block


def _final_snapshot(run_dir: Path):
    snaps = {int(re.match(r"kb_epoch_(\d+)", p.stem).group(1)): p
             for p in run_dir.glob("kb_epoch_*.json")}
    return snaps[max(snaps)] if snaps else None


def _link_alignment(run_dir: Path, universe: Universe | None) -> dict | None:
    snap = _final_snapshot(run_dir)
    if snap is None or universe is None:
        return None
    with open(snap) as f:
        docs = json.load(f)["store"]["docs"]
    links = [(d["id"], lid) for d in docs for lid in d["links"]]
    origin_home = {st["origin"]: d["id"]
                   for d in universe.docs for st in d["statements"]}
    support_sets = [frozenset(origin_home[o] for o in q.support if o in origin_home)
                    for q in universe.questions.values() if q.support]
    aligned = sum(1 for a, b in links
                  if any(a in s and b in s for s in support_sets))
    return {"links": len(links), "aligned": aligned,
            "share": aligned / len(links) if links else 0.0}


def build_report(run_dir) -> dict:
    run_dir = Path(run_dir)
    train_rows = _rows(run_dir / "train_log.jsonl")
    test_rows = _rows(run_dir / "test_log.jsonl")
    kb_stats = _rows(run_dir / "kb_stats.jsonl")
    trace = _rows(run_dir / "trace.jsonl")
    meta = {}
    if (run_dir / "meta.json").exists():
        with open(run_dir / "meta.json") as f:
            meta = json.load(f)

    universe = None
    upath = meta.get("universe")
    if upath and Path(upath).exists():
        universe = Universe.load(upath)

    # E1 learning curve: test scores by epoch x split (epoch 0 = RAG baseline)
    curve: dict = defaultdict(dict)
    for (epoch, split), rows in _group(test_rows, lambda r: (r["epoch"], r["split"])).items():
        curve[epoch][split] = _score_block(rows)

    # E3 three layers: train-forward (valid from epoch 2) + per-class slices
    forward = {epoch: _score_block(rows)
               for epoch, rows in _group(train_rows, lambda r: r["epoch"]).items()}
    by_template: dict = defaultdict(dict)
    by_category: dict = defaultdict(dict)
    for (split, t), rows in _group(test_rows, lambda r: (r["split"], r["template"])).items():
        by_template[split][t] = {e: _score_block(rs)
                                 for e, rs in _group(rows, lambda r: r["epoch"]).items()}
    for (split, c), rows in _group(test_rows, lambda r: (r["split"], r["category"])).items():
        by_category[split][c] = {e: _score_block(rs)
                                 for e, rs in _group(rows, lambda r: r["epoch"]).items()}

    # E5 structure: edit mix per epoch + kb stats trajectory + link alignment
    edit_mix: dict = defaultdict(lambda: defaultdict(int))
    for r in train_rows:
        for a, n in r["edits"].items():
            edit_mix[r["epoch"]][a] += n

    # E2 token tallies: build / train / test separated
    tok = defaultdict(lambda: {"in": 0, "out": 0})
    for r in trace:
        tok[r["kind"]]["in"] += r.get("tokens_in", 0)
        tok[r["kind"]]["out"] += r.get("tokens_out", 0)
    tokens = {"build": meta.get("build_tokens"),
              "train": dict(tok["train"]), "test": dict(tok["test"]),
              "train_summarizer": meta.get("summarizer_tokens")}

    return {"learning_curve": {e: dict(s) for e, s in sorted(curve.items())},
            "train_forward": {"valid_from_epoch": 2,
                              "by_epoch": dict(sorted(forward.items()))},
            "by_category": {s: dict(v) for s, v in by_category.items()},
            "by_template": {s: dict(v) for s, v in by_template.items()},
            "edit_mix": {e: dict(v) for e, v in sorted(edit_mix.items())},
            "link_alignment": _link_alignment(run_dir, universe),
            "kb_stats": kb_stats,
            "tokens": tokens}


def _group(rows, key):
    out = defaultdict(list)
    for r in rows:
        out[key(r)].append(r)
    return out


def render(report: dict) -> str:
    lines = ["== E1 learning curve (headline F1; QC5 reported separately) =="]
    lines.append(f"{'epoch':>5} {'split':>9} {'n':>4} {'F1':>6} {'EM':>6} "
                 f"{'unansw':>7} {'wrong':>6} {'QC5acc':>7}")
    for epoch, splits in report["learning_curve"].items():
        for split, b in sorted(splits.items()):
            qc5 = "-" if b["unanswerable_acc"] is None else f"{b['unanswerable_acc']:.2f}"
            lines.append(f"{epoch:>5} {split:>9} {b['n']:>4} {b['f1']:>6.3f} "
                         f"{b['em']:>6.3f} {b.get('unanswered_rate', 0.0):>7.2f} "
                         f"{b.get('wrong_rate', 0.0):>6.2f} {qc5:>7}")
    fwd = report["train_forward"]["by_epoch"]
    lines.append("\n== E3 train-forward F1 (valid from epoch 2) ==")
    for epoch, b in fwd.items():
        lines.append(f"  epoch {epoch}: F1 {b['f1']:.3f} (n={b['n']})")
    lines.append("\n== E5 edit mix per epoch ==")
    for epoch, mix in report["edit_mix"].items():
        used = {a: n for a, n in mix.items() if n}
        lines.append(f"  epoch {epoch}: {used or '(no edits)'}")
    la = report["link_alignment"]
    if la:
        lines.append(f"\n== E5 link-vs-support alignment == "
                     f"{la['aligned']}/{la['links']} ({la['share']:.2f})")
    lines.append("\n== KB stats trajectory ==")
    for s in report["kb_stats"]:
        lines.append(f"  epoch {s['epoch']}: docs {s['docs']} stmts "
                     f"{s['statements']} coverage {s['coverage']:.2f} dup "
                     f"{s['dup_origins']} links {s['links']} orphans "
                     f"{s['orphan_docs']} (+{s['created_docs']}/"
                     f"-{s['deleted_docs']} docs)")
    t = report["tokens"]
    lines.append(f"\n== E2 tokens == build {t['build']} | train {t['train']} "
                 f"| test {t['test']} | train summarizer {t['train_summarizer']}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    args = ap.parse_args(argv)
    report = build_report(args.run_dir)
    with open(Path(args.run_dir) / "report.json", "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(render(report))


if __name__ == "__main__":
    main()
