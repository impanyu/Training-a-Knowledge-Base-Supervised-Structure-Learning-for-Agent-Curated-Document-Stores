"""Crash-safe JSONL logs for a run directory (spec §8):

    trace.jsonl      every action of every iteration (kind, epoch, qid,
                     phase, step, action, input, result, tokens)
    train_log.jsonl  per train iteration
    test_log.jsonl   per test question
    kb_stats.jsonl   per epoch store stats

Token totals are tallied live per run kind (train / test) so meta.json can
report build/train/test tallies without re-scanning the trace."""
import json
from collections import defaultdict
from pathlib import Path

_FILES = ("trace", "train_log", "test_log", "kb_stats")


class RunLog:
    def __init__(self, out_dir):
        self.dir = Path(out_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self._f = {name: open(self.dir / f"{name}.jsonl", "w") for name in _FILES}
        self.tokens = defaultdict(lambda: {"in": 0, "out": 0})

    def _write(self, name: str, row: dict) -> None:
        f = self._f[name]
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
        f.flush()

    def trace(self, row: dict) -> None:
        t = self.tokens[row["kind"]]
        t["in"] += row.get("tokens_in", 0)
        t["out"] += row.get("tokens_out", 0)
        self._write("trace", row)

    def train(self, row: dict) -> None:
        self._write("train_log", row)

    def test(self, row: dict) -> None:
        self._write("test_log", row)

    def stats(self, row: dict) -> None:
        self._write("kb_stats", row)

    def close(self) -> None:
        for f in self._f.values():
            f.close()
