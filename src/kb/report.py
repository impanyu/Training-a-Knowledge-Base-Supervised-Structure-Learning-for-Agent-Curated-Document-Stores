"""Report over a run directory's logs (spec §8/§11: E1/E2/E3/E5; T39.1).

Emits report.json + a printed plain-text table. Headline F1 excludes QC5
(unanswerable) questions, whose accuracy is reported separately, never
blended. The learning curve is the EVAL split by epoch (flavors eval_in /
eval_out); test_in/test_out appear as single final points (plus whatever
baseline passes were appended manually via kb.test). Train-forward F1 is
valid from epoch 2 (epoch 1 trains on a store nothing has consolidated
yet). Cost metrics are headline (T39.1): training cost and per-question
answering cost in both tokens and wall-clock seconds, and KB size incl.
approximate statement tokens (chars // 4). Link-vs-support alignment =
share of built links connecting nodes that co-occur in some question's
support set (support ids ARE initial node ids in the v9.6 node graph)."""
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
        nodes = json.load(f)["store"]["nodes"]
    links = [(n["id"], lid) for n in nodes for lid in n["links"]]
    # support ids ARE the initial node ids (origin = own id at build time)
    support_sets = [frozenset(q.support)
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

    # E1 learning curve: EVAL split by epoch x flavor (epoch 0 = RAG baseline)
    eval_rows = [r for r in test_rows if r["split"] == "eval"]
    curve: dict = defaultdict(dict)
    for (epoch, flavor), rows in _group(eval_rows,
                                        lambda r: (r["epoch"], r["flavor"])).items():
        curve[epoch][f"eval_{flavor}"] = _score_block(rows)

    # the full test splits: single final point per split (plus any manually
    # appended baseline pass), keyed by the epoch label kb.test recorded
    final: dict = defaultdict(dict)
    for (split, epoch), rows in _group(
            [r for r in test_rows if r["split"] != "eval"],
            lambda r: (r["split"], r["epoch"])).items():
        final[split][epoch] = _score_block(rows)

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

    # E2 token tallies: train / test separated (the node-graph build makes
    # no API calls at all, so there is no build-token column anymore)
    tok = defaultdict(lambda: {"in": 0, "out": 0})
    for r in trace:
        tok[r["kind"]]["in"] += r.get("tokens_in", 0)
        tok[r["kind"]]["out"] += r.get("tokens_out", 0)
    tokens = {"train": dict(tok["train"]), "test": dict(tok["test"])}

    # T39.1 headline cost metrics: tokens AND wall-clock, train + per split
    costs = {"train": _train_cost(train_rows)}
    for split, rows in _group(test_rows, lambda r: r["split"]).items():
        costs[split] = _question_cost(rows)

    # KB size from the last kb_stats row (statement_tokens = chars // 4)
    kb_size = None
    if kb_stats:
        last = kb_stats[-1]
        kb_size = {k: last[k] for k in ("n_nodes", "n_links",
                                        "statement_tokens") if k in last}

    return {"learning_curve": {e: dict(s) for e, s in sorted(curve.items())},
            "final_test": {s: dict(sorted(v.items()))
                           for s, v in sorted(final.items())},
            "costs": costs,
            "kb_size": kb_size,
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


def _train_cost(rows: list[dict]) -> dict:
    n = len(rows)
    tin = sum(r.get("tokens_in", 0) for r in rows)
    tout = sum(r.get("tokens_out", 0) for r in rows)
    secs = sum(r.get("seconds", 0.0) for r in rows)
    return {"iterations": n, "tokens_in": tin, "tokens_out": tout,
            "seconds": secs,
            "mean_tokens_per_iteration": (tin + tout) / n if n else 0.0,
            "mean_seconds_per_iteration": secs / n if n else 0.0}


def _question_cost(rows: list[dict]) -> dict:
    return {"questions": len(rows),
            "mean_tokens": _mean(r.get("tokens_in", 0) + r.get("tokens_out", 0)
                                 for r in rows),
            "mean_seconds": _mean(r.get("seconds", 0.0) for r in rows),
            "mean_steps": _mean(r["steps"] for r in rows)}


def _score_lines(lines, blocks_by_row):
    lines.append(f"{'epoch':>5} {'split':>9} {'n':>4} {'F1':>6} {'EM':>6} "
                 f"{'unansw':>7} {'wrong':>6} {'QC5acc':>7}")
    for epoch, split, b in blocks_by_row:
        qc5 = "-" if b["unanswerable_acc"] is None else f"{b['unanswerable_acc']:.2f}"
        lines.append(f"{epoch:>5} {split:>9} {b['n']:>4} {b['f1']:>6.3f} "
                     f"{b['em']:>6.3f} {b.get('unanswered_rate', 0.0):>7.2f} "
                     f"{b.get('wrong_rate', 0.0):>6.2f} {qc5:>7}")


def render(report: dict) -> str:
    lines = ["== E1 learning curve: eval split by epoch (headline F1; QC5 "
             "reported separately) =="]
    _score_lines(lines, [(e, flavor, b)
                         for e, flavors in report["learning_curve"].items()
                         for flavor, b in sorted(flavors.items())])
    lines.append("\n== E1 final test (full splits, run once via kb.test) ==")
    if report["final_test"]:
        _score_lines(lines, [(e, split, b)
                             for split, by_epoch in report["final_test"].items()
                             for e, b in by_epoch.items()])
    else:
        lines.append("  (no test_in/test_out rows appended yet)")
    tc = report["costs"]["train"]
    lines.append("\n== headline costs ==")
    lines.append(f"  train: {tc['iterations']} iterations | tokens "
                 f"{tc['tokens_in']}+{tc['tokens_out']} | {tc['seconds']:.1f}s "
                 f"| per-iteration {tc['mean_tokens_per_iteration']:.0f} tok / "
                 f"{tc['mean_seconds_per_iteration']:.2f}s")
    for split, c in sorted(report["costs"].items()):
        if split == "train":
            continue
        lines.append(f"  {split}: {c['questions']} questions | per-question "
                     f"{c['mean_tokens']:.0f} tok / {c['mean_seconds']:.2f}s / "
                     f"{c['mean_steps']:.1f} steps")
    if report["kb_size"]:
        ks = report["kb_size"]
        lines.append(f"  kb size: {ks['n_nodes']} nodes "
                     f"(~{ks['statement_tokens']} tok), "
                     f"{ks['n_links']} links")
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
        cost = (f" | train {s['train_tokens_in'] + s['train_tokens_out']} tok "
                f"{s['train_seconds']:.1f}s" if s.get("train_iterations") else "")
        lines.append(f"  epoch {s['epoch']}: nodes {s['n_nodes']} "
                     f"(~{s.get('statement_tokens', 0)} tok) "
                     f"coverage {s['coverage']:.2f} dup {s['dup_origins']} "
                     f"links {s['n_links']} orphans {s['orphan_nodes']} "
                     f"authored {s.get('authored_statements', 0)} edited "
                     f"{s.get('edited_statements', 0)}{cost}")
    t = report["tokens"]
    lines.append(f"\n== E2 tokens == train {t['train']} | test {t['test']}")
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
