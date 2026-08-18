"""EXTERNAL-VALIDITY arm (T41): adapt an official PhantomWiki generation
(https://github.com/kilian-group/phantom-wiki, `pip install phantom-wiki`,
needs SWI-Prolog) into OUR universe.json so kb.train / kb.test run unchanged.

    python -m kb.adapt_phantomwiki --src <pw output dir> --out data/pw1

Input (phantom-wiki 1.0.3, `--article-format json --question-format json`):
`<src>/articles.json` [{title, article, facts}], `<src>/questions.json`
[{id, question, answer[list], template[tokens], type, difficulty,
is_aggregation_question, ...}].

Statements: each article is split into sentence-level statement nodes —
PhantomWiki articles are template-rendered one fact-sentence per line, so we
split on newlines, drop `#`/`##` headers and blanks, then split any residual
multi-sentence line on `. ` boundaries before a capital. Normalization for
self-containedness (their 1.0.3 templates never actually emit pronouns —
every line carries the person's full name — but the guard is kept for other
versions): a leading subject pronoun (He/She/They) is replaced by the
article's title, a leading possessive (His/Her/Their) by "<title>'s", and a
sentence that still lacks the title is prefixed "<title>: " as a last
resort. Nodes get sequential ids in article file order, origin = own id,
zero links (mirrors kb.build's initial store).

Questions: PhantomWiki's OWN generated QA. Their template `type` becomes our
template id (`pw_type<t>`, with the joined template token string recorded in
vocab); types are grouped into 4 coarse CATEGORIES by question shape
(PW_REL "who is the relation...", PW_PERSON "who is the person whose
attr...", PW_ATTR "what is the attr...", PW_AGG "how many...") so the
two-tier category/template scheme of kb.build survives; their `difficulty`
becomes our per-question `hops`. List answers -> comma-joined gold plus each
element as a variant (the kb.build QC7 convention); single numeric answers
additionally get the number-word variant. An empty answer list (never
produced in our generations) would map to gold "unknown" / unanswerable.

Splits mirror kb.build's two-tier scheme: their 8-template taxonomy is too
coarse for kb.build's one-reserved-per-category rule, so per the task spec
we reserve TWO templates for test_out — type 0 (PW_REL: "Who is the
<relation> of the person whose <attr> is <value>?") and type 6 (PW_AGG:
"How many <relation_plural> does the person whose <attr> is <value>
have?") — one relational, one aggregative, each from a category that keeps
other templates in training. train / test_in / eval-"in" draw round-robin
from the six trained templates, test_out / eval-"out" from the two reserved
ones; duplicate question TEXTS are dropped before splitting (keep first),
giving instance-level disjointness everywhere. Deterministic per --seed.

Support sets: EMPTY on every question. PhantomWiki's solution traces bind
intermediate ENTITIES, not the base facts they were derived from (e.g.
`grandfather(Y_4, Y_2)` is a Prolog-derived relation with no single article
sentence behind it); mapping traces to statement ids would mean
re-implementing their derivation rules. The train/test protocol never reads
supports — only the post-hoc repair diagnostics do, and those are simply
unavailable on this arm (noted in meta)."""
import argparse
import json
import random
import re
from pathlib import Path

from kb.build import Question, Universe, _num_variants, _take

# type -> (our template id, category); see module docstring for the shapes
CATEGORY = {0: "PW_REL", 1: "PW_REL", 2: "PW_PERSON", 3: "PW_ATTR",
            4: "PW_ATTR", 5: "PW_AGG", 6: "PW_AGG", 7: "PW_AGG"}
RESERVED_TYPES = (0, 6)

_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])")
_SUBJ_PRONOUN = re.compile(r"^(He|She|They)\b")
_POSS_PRONOUN = re.compile(r"^(His|Her|Their)\b")


def normalize(title: str, sentence: str) -> str:
    """Make one sentence self-contained: leading subject pronoun -> title,
    leading possessive -> "title's", last-resort "title: " prefix when the
    person's name still does not appear."""
    s = _SUBJ_PRONOUN.sub(title, sentence)
    s = _POSS_PRONOUN.sub(f"{title}'s", s)
    if title not in s:
        s = f"{title}: {s}"
    return s


def split_article(title: str, article: str) -> list[str]:
    """Sentence-level statements: newline-separated fact lines minus
    headers/blanks, residual multi-sentence lines split on `. Capital`
    boundaries, every sentence normalized for self-containedness."""
    out = []
    for line in article.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        for sent in _SENT_SPLIT.split(line):
            sent = sent.strip()
            if sent:
                out.append(normalize(title, sent))
    return out


def _golds(answers: list[str]) -> tuple[list[str], bool]:
    """(golds, unanswerable). Comma-joined gold + each-element variants for
    lists; number-word variant for a single numeric answer; empty list ->
    unanswerable gold "unknown" (kb.build QC5/QC7/QC4 conventions)."""
    answers = [str(a) for a in answers]
    if not answers:
        return ["unknown"], True
    if len(answers) == 1:
        a = answers[0]
        return (_num_variants(int(a)) if a.isdigit() else [a]), False
    return [", ".join(answers)] + answers, False


def adapt(src, seed: int = 0, sizes: tuple[int, int, int] = (150, 100, 50),
          eval_sizes: tuple[int, int] = (20, 10)) -> Universe:
    src = Path(src)
    with open(src / "articles.json") as f:
        articles = json.load(f)
    with open(src / "questions.json") as f:
        pw_questions = json.load(f)

    nodes, sid_n = [], 0
    for a in articles:
        for text in split_article(a["title"], a["article"]):
            sid_n += 1
            sid = f"s{sid_n:04d}"
            nodes.append({"id": sid, "text": text, "origin": sid})

    # template pools in file order, duplicate question texts dropped
    pools: dict[str, list[dict]] = {}
    templates: dict[str, dict] = {}
    seen_texts: set[str] = set()
    for q in pw_questions:
        t = q["type"]
        if t not in CATEGORY:
            raise SystemExit(f"unknown PhantomWiki question type {t}; "
                             f"extend CATEGORY/RESERVED_TYPES for this "
                             f"generation")
        if q["question"] in seen_texts:
            continue
        seen_texts.add(q["question"])
        tmpl = f"pw_type{t}"
        golds, unanswerable = _golds(q["answer"])
        pools.setdefault(tmpl, []).append(
            {"text": q["question"], "golds": golds, "hops": q["difficulty"],
             "unanswerable": unanswerable})
        templates.setdefault(tmpl, {
            "category": CATEGORY[t], "pw_type": t,
            "reserved": t in RESERVED_TYPES,
            "pw_template": " ".join(q["template"]),
            "is_aggregation": q["is_aggregation_question"]})

    rng = random.Random(seed)
    for tmpl in pools:
        rng.shuffle(pools[tmpl])
    trained = sorted(t for t in pools if not templates[t]["reserved"])
    reserved = sorted(t for t in pools if templates[t]["reserved"])

    picked = {"train": _take(pools, trained, sizes[0]),
              "test_in": _take(pools, trained, sizes[1]),
              "test_out": _take(pools, reserved, sizes[2])}
    ev = [("in", t, row) for t, row in _take(pools, trained, eval_sizes[0])]
    ev += [("out", t, row) for t, row in _take(pools, reserved, eval_sizes[1])]
    for split, want in zip(picked, sizes):
        if len(picked[split]) < want:
            raise SystemExit(f"only {len(picked[split])} questions for split "
                             f"{split} (wanted {want}); regenerate with a "
                             f"larger --num-questions-per-type")

    questions, splits, qn = {}, {}, 0

    def _add(split, tmpl, row, flavor=None):
        nonlocal qn
        qn += 1
        qid = f"q{qn:04d}"
        questions[qid] = Question(
            qid, tmpl, templates[tmpl]["category"], row["hops"], row["text"],
            row["golds"], [], row["unanswerable"], flavor)
        splits.setdefault(split, []).append(qid)

    for split, rows in picked.items():
        splits[split] = []
        for tmpl, row in rows:
            _add(split, tmpl, row)
    splits["eval"] = []
    for flavor, tmpl, row in ev:
        _add("eval", tmpl, row, flavor)

    meta = {"source": "phantom-wiki", "src": str(src), "seed": seed,
            "sizes": list(sizes), "eval_sizes": list(eval_sizes),
            "n_articles": len(articles), "n_pw_questions": len(pw_questions),
            "supports": "empty — PhantomWiki solution traces bind entities, "
                        "not base facts; repair diagnostics unavailable on "
                        "this arm"}
    vocab = {"templates": {t: {"category": d["category"],
                               "hops": None,       # varies per instance
                               "reserved": d["reserved"],
                               "pw_type": d["pw_type"],
                               "pw_template": d["pw_template"],
                               "is_aggregation": d["is_aggregation"]}
                           for t, d in sorted(templates.items())}}
    return Universe(meta, nodes, questions, splits, vocab)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True,
                    help="phantom-wiki output dir (articles.json + "
                         "questions.json, json formats)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0,
                    help="split-sampling seed (the universe itself is fixed "
                         "by the phantom-wiki generation)")
    ap.add_argument("--train", type=int, default=150)
    ap.add_argument("--test-in", type=int, default=100)
    ap.add_argument("--test-out", type=int, default=50)
    ap.add_argument("--eval-in", type=int, default=20)
    ap.add_argument("--eval-out", type=int, default=10)
    args = ap.parse_args(argv)

    u = adapt(args.src, args.seed, (args.train, args.test_in, args.test_out),
              (args.eval_in, args.eval_out))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    u.save(out / "universe.json")
    cats: dict[str, int] = {}
    for q in u.questions.values():
        cats[q.category] = cats.get(q.category, 0) + 1
    print(json.dumps({"articles": u.meta["n_articles"],
                      "nodes": len(u.nodes),
                      "questions": len(u.questions),
                      "splits": {s: len(q) for s, q in u.splits.items()},
                      "categories": dict(sorted(cats.items()))}, indent=2))


if __name__ == "__main__":
    main()
