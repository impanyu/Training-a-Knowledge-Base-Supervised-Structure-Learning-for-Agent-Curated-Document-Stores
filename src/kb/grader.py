"""SQuAD-style answer scoring (normalization, exact match, token F1) plus
unanswerable handling. Copied from ca.grader to keep kb self-contained."""
import re
import string
from collections import Counter

# Accepted ways of saying "this universe cannot determine it" (spec §7).
UNKNOWN_ALIASES = ["unknown", "not known", "no answer", "unanswerable",
                   "cannot be determined", "i don't know", "don't know", "none"]


def normalize(s: str) -> str:
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def _f1_single(pred: str, gold: str) -> float:
    pt, gt = normalize(pred).split(), normalize(gold).split()
    if not pt or not gt:
        return float(pt == gt)
    common = Counter(pt) & Counter(gt)
    overlap = sum(common.values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(pt)
    recall = overlap / len(gt)
    return 2 * precision * recall / (precision + recall)


def exact_match(pred: str, golds: list[str]) -> float:
    return float(any(normalize(pred) == normalize(g) for g in golds))


def f1(pred: str, golds: list[str]) -> float:
    return max((_f1_single(pred, g) for g in golds), default=0.0)


def is_unknown(pred: str) -> bool:
    return normalize(pred) in {normalize(a) for a in UNKNOWN_ALIASES}


def grade(pred: str, golds: list[str], unanswerable: bool = False) -> tuple[float, float]:
    """(f1, em). Unanswerable questions are all-or-nothing: correct iff the
    normalized prediction is an unknown alias; never partial credit."""
    if unanswerable:
        ok = is_unknown(pred)
        return (1.0, 1.0) if ok else (0.0, 0.0)
    return f1(pred, golds), exact_match(pred, golds)
