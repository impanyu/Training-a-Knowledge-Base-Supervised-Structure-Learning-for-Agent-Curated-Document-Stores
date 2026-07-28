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

from datasets import load_dataset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from ca.retrieval import ChromaBackend  # noqa: E402

# Pricing rule (spec §8): R(q) ~= 1.5x the average billable token burn to solve a
# question of that tier, so solving is profitable. Placeholder estimates assume a
# billable turn burns ~3-4k tokens and 2/3/4-hop questions take ~2-3/4/5-6 billable
# turns. Recalibrate from pilot measurements (spec §13).
PRICES = {"2hop": 12000, "3hop": 20000, "4hop": 30000}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hotpot-n", type=int, default=150)
    ap.add_argument("--musique-n", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="data")
    args = ap.parse_args()
    rng = random.Random(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    pool, corpus, seen = [], [], set()

    def add_doc(title, text):
        key = (title, text[:80])
        if key not in seen:
            seen.add(key)
            corpus.append({"title": title, "text": text})

    # ---- HotpotQA (distractor config: paragraphs travel with the question) ----
    hp = load_dataset("hotpotqa/hotpot_qa", "distractor", split="validation")
    idxs = rng.sample(range(len(hp)), args.hotpot_n)
    for n, i in enumerate(idxs):
        ex = hp[i]
        pool.append({"qid": f"q{len(pool)+1:04d}", "text": ex["question"],
                     "answers": [ex["answer"]], "difficulty": "2hop",
                     "price": PRICES["2hop"], "source": "hotpotqa"})
        for title, sents in zip(ex["context"]["title"], ex["context"]["sentences"]):
            add_doc(title, " ".join(sents))

    # ---- MuSiQue (answerable) ----
    mq = load_dataset("dgslibisey/MuSiQue", split="validation")
    idxs = rng.sample(range(len(mq)), args.musique_n)
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

    with open(out / "pool.jsonl", "w") as f:
        for r in pool:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"pool: {len(pool)} questions; corpus: {len(corpus)} paragraphs")

    ChromaBackend(corpus, persist_dir=str(out / "index"))  # builds + persists embeddings
    print(f"chroma index saved to {out}/index")


if __name__ == "__main__":
    main()
