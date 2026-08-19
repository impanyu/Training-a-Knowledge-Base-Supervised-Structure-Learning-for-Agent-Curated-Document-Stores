"""Characterize agent-authored notes: are they class-level or instance-level?

Proxy: how many distinct entity names a note mentions. A class index note
("People whose hobby is pottery include A, B, C, D, E") covers many; a
single verified chain covers two or three; a contentless pointer note covers
one or none. The v10L run scored mean 2.07 with 55% of notes at <=1 entity
and 1% at 10+, which is the quantitative form of its overfitting.
"""
import argparse
import json
import re
import statistics as S

NAME = re.compile(r"\b[A-Z][a-z]+ [A-Z][a-z]+\b")
BUCKETS = [("0-1 (meta/pointer)", 0, 1), ("2-3 (single instance)", 2, 3),
           ("4-9 (small class)", 4, 9), ("10+ (class index)", 10, 10**9)]


def profile(path):
    d = json.load(open(path))
    nodes = d["store"]["nodes"] if "store" in d else d["nodes"]
    auth = [n for n in nodes if n.get("flag") == "authored"]
    if not auth:
        return None
    cov = [len(set(NAME.findall(n["text"]))) for n in auth]
    return auth, cov


def show(label, path):
    p = profile(path)
    if not p:
        print(f"{label}: no authored notes")
        return
    auth, cov = p
    print(f"{label}: {len(auth)} authored notes, "
          f"mean {S.mean(cov):.2f} entities/note, median {S.median(cov):.0f}")
    for name, lo, hi in BUCKETS:
        k = sum(1 for c in cov if lo <= c <= hi)
        print(f"    {name:<24}{k:>4}  {k/len(auth):>6.0%}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--kb", required=True)
    ap.add_argument("--baseline", default=None)
    a = ap.parse_args()
    if a.baseline:
        show("baseline", a.baseline)
    show("pilot   ", a.kb)
