"""Post the flat question bank as JOBS (v4.1).

A single question is solved alone in ~3 turns, which is cheaper than hiring
anyone -- the v4 calibration found a market with zero messages, contracts and
loans. So the claimable unit becomes a JOB: 2-10 questions drawn from ONE topic
cluster, claimed and delivered as a unit. No tree, no decompose, no sentence
addressing; just a flat list of qids that does not fit in one agent's claim
window.

Jobs are allocated across topic clusters in proportion to cluster size, and
members are picked least-covered-first, so every question in an eligible
cluster sits in about the same number of jobs (~3). Job membership is therefore
the repeat mechanism, and the per-question `quota` of v4 is retired.
"""
import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path


def allocate(sizes: dict[str, int], n_jobs: int) -> dict[str, int]:
    """Jobs per topic, proportional to cluster size (largest remainder), so the
    posted demand mirrors the shape of the bank."""
    total = sum(sizes.values())
    exact = {t: n_jobs * n / total for t, n in sizes.items()}
    alloc = {t: int(v) for t, v in exact.items()}
    for t in sorted(exact, key=lambda t: (-(exact[t] - alloc[t]), t))[
            :n_jobs - sum(alloc.values())]:
        alloc[t] += 1
    return alloc


def build_jobs(questions: list[dict], n_jobs: int, size_lo: int, size_hi: int,
               seed: int) -> tuple[list[dict], list[str]]:
    """Returns (jobs, skipped_topics). A cluster smaller than `size_lo` cannot
    host a job of the required size, so it posts nothing."""
    rng = random.Random(seed)
    by_topic: dict[str, list[str]] = defaultdict(list)
    price = {}
    for q in questions:
        by_topic[q["topic"]].append(q["qid"])
        price[q["qid"]] = q["price"]
    eligible = {t: qids for t, qids in by_topic.items() if len(qids) >= size_lo}
    skipped = sorted(set(by_topic) - set(eligible))

    alloc = allocate({t: len(qids) for t, qids in eligible.items()}, n_jobs)
    jobs = []
    for topic in sorted(eligible):
        cover = dict.fromkeys(sorted(eligible[topic]), 0)
        hi = min(size_hi, len(cover))
        for _ in range(alloc[topic]):
            size = rng.randint(size_lo, hi)
            order = list(cover)
            rng.shuffle(order)
            order.sort(key=lambda qid: cover[qid])   # least-covered first
            qids = sorted(order[:size])
            for qid in qids:
                cover[qid] += 1
            jobs.append({"qids": qids, "price": sum(price[q] for q in qids)})
    rng.shuffle(jobs)                                # do not post topic by topic
    for i, job in enumerate(jobs, start=1):
        job["jid"] = f"j{i:04d}"
    return [{"jid": j["jid"], "qids": j["qids"], "price": j["price"]}
            for j in jobs], skipped


def summarize(bank: dict, skipped: list[str]) -> str:
    jobs, questions = bank["jobs"], bank["questions"]
    sizes = Counter(len(j["qids"]) for j in jobs)
    member = Counter(qid for j in jobs for qid in j["qids"])
    per_q = Counter(member[q["qid"]] for q in questions)
    lines = [
        f"{len(jobs)} jobs over {len(questions)} questions, "
        f"{bank['total_units']} (job, question) units, {bank['n_topics']} topics",
        "job sizes:      " + ", ".join(f"{s}:{n}" for s, n in sorted(sizes.items())),
        "memberships:    " + ", ".join(f"{k} job(s):{n}" for k, n in sorted(per_q.items()))
        + f"  (mean {bank['total_units'] / len(questions):.2f})",
        f"total posted value: {sum(j['price'] for j in jobs)}",
        f"job price: min {min(j['price'] for j in jobs)} "
        f"max {max(j['price'] for j in jobs)} "
        f"mean {sum(j['price'] for j in jobs) // len(jobs)}",
    ]
    if skipped:
        lines.append(f"topics too small to host a job (unposted): {', '.join(skipped)} "
                     f"-- {sum(1 for q in questions if q['topic'] in skipped)} questions")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bank", default="data/v4/bank.json")
    ap.add_argument("--out", default=None, help="defaults to --bank (in place)")
    ap.add_argument("--jobs", type=int, default=185)
    ap.add_argument("--size-lo", type=int, default=2)
    ap.add_argument("--size-hi", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    with open(args.bank) as f:
        bank = json.load(f)
    for q in bank["questions"]:
        q.pop("quota", None)            # retired: job membership is the repeat
    jobs, skipped = build_jobs(bank["questions"], args.jobs,
                               args.size_lo, args.size_hi, args.seed)
    bank["jobs"] = jobs
    bank["total_units"] = sum(len(j["qids"]) for j in jobs)
    bank["n_jobs"] = len(jobs)

    out = args.out or args.bank
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(bank, f, ensure_ascii=False)
    print(f"bank saved to {out}")
    print(summarize(bank, skipped))


if __name__ == "__main__":
    main()
