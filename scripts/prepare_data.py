"""Build the mixed HotpotQA+MuSiQue pool, pooled paragraph corpus, Chroma index.

Corpus strategy (standard open-retrieval practice, cf. IRCoT): pool the
per-question paragraph sets (gold + distractors) across all sampled questions
into one deduplicated corpus. Gold paragraphs are guaranteed present.
"""
import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from ca.retrieval import ChromaBackend  # noqa: E402

# Pricing rule (spec §8): R(q) ~= 1.5x the average billable token burn to solve a
# question of that tier, so solving is profitable. Calibrated from pilot round 1
# (2026-07: clean 2-hop solve ~11-15k billable tokens incl. context growth).
PRICES = {"2hop": 18000, "3hop": 30000, "4hop": 45000}


def build_pool_and_corpus(hotpot_n: int, musique_n: int, seed: int) -> tuple[list[dict], list[dict]]:
    """Download hotpot_n HotpotQA (distractor) + musique_n MuSiQue (validation)
    questions and their paragraph sets; return (pool, corpus). Requires
    network + the `datasets` extra; not exercised by offline tests -- callers
    that need to stay offline should stub this out."""
    from datasets import load_dataset  # deferred: heavy import, network on first use

    rng = random.Random(seed)
    pool, corpus, seen = [], [], set()

    def add_doc(title, text):
        key = (title, text[:80])
        if key not in seen:
            seen.add(key)
            corpus.append({"title": title, "text": text})

    # ---- HotpotQA (distractor config: paragraphs travel with the question) ----
    hp = load_dataset("hotpotqa/hotpot_qa", "distractor", split="validation")
    idxs = rng.sample(range(len(hp)), hotpot_n)
    for i in idxs:
        ex = hp[i]
        pool.append({"qid": f"q{len(pool)+1:04d}", "text": ex["question"],
                     "answers": [ex["answer"]], "difficulty": "2hop",
                     "price": PRICES["2hop"], "source": "hotpotqa"})
        for title, sents in zip(ex["context"]["title"], ex["context"]["sentences"]):
            add_doc(title, " ".join(sents))

    # ---- MuSiQue (answerable) ----
    mq = load_dataset("dgslibisey/MuSiQue", split="validation")
    idxs = rng.sample(range(len(mq)), musique_n)
    for i in idxs:
        ex = mq[i]
        hops = "4hop" if ex["id"].startswith("4hop") else (
               "3hop" if ex["id"].startswith("3hop") else "2hop")
        answers = [ex["answer"]] + list(ex.get("answer_aliases") or [])
        pool.append({"qid": f"q{len(pool)+1:04d}", "text": ex["question"],
                     "answers": answers, "difficulty": hops,
                     "price": PRICES[hops], "source": "musique"})
        for p in ex["paragraphs"]:
            add_doc(p["title"], p["paragraph_text"])

    return pool, corpus


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hotpot-n", type=int, default=150)
    ap.add_argument("--musique-n", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="data")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    pool, corpus = build_pool_and_corpus(args.hotpot_n, args.musique_n, args.seed)

    with open(out / "pool.jsonl", "w") as f:
        for r in pool:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"pool: {len(pool)} questions; corpus: {len(corpus)} paragraphs")

    ChromaBackend(corpus, persist_dir=str(out / "index"))  # builds + persists embeddings
    print(f"chroma index saved to {out}/index")


if __name__ == "__main__":
    main()
