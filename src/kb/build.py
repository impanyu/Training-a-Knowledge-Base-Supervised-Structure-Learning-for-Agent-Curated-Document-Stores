"""Universe generator (spec §6), PhantomWiki-inspired (fictional persons,
zero contamination, no Prolog — question classes are Python-derivable).

Entities: n_people fictional persons in small family trees (2-3 generations,
spouse links, <=3 children per couple) plus Erdos-Renyi friendships; job /
hobby / city attributes from fixed vocabularies. Facts render into
self-contained, pronoun-free statements; relations render one statement per
endpoint doc (two distinct origins, intentionally). Zero links either way.

Three initializations (T39.2/T39.5 — the entity init is already near-optimal
organization, leaving training no headroom):
- `entity` (default): one doc per person.
- `scattered`: the SAME statements distributed into mixed "notebook" docs of
  4-8 statements from different people (locality knob: 30% chance
  consecutive statements share a person), doc count ~ n_statements/6,
  generic summaries ("Mixed notes (7 statements).") that never enumerate
  the people inside. Arrangement uses its own derived rng, so questions,
  golds, splits and statements are IDENTICAL to the entity init at the
  same seed — only the doc arrangement differs.
- `atomized` (T39.5): every statement in its OWN single-statement doc
  (~1200 docs at --people 120), zero links. Since the doc IS the statement,
  the template summary is the statement text itself truncated to <=80 chars
  — no summarizer could say more about a one-sentence doc. Arrangement uses
  no randomness at all, so the same cross-init identity guarantee holds.

Name distractors (T39.2): a fraction `distractors` of extra people whose
FIRST names collide with real subjects but whose full names never equal a
questioned person's; they get attribute statements in the store and are
never asked about — naive retrieval precision degrades, gold validity
doesn't.

Two-level split scheme: 5 CATEGORIES (QC1-5), 15 concrete TEMPLATES.
train = i.i.d. over the 10 trained templates (all categories represented);
test_in = unseen instances of trained templates; test_out = the 5 reserved
templates (1 per category), never used in train. A fourth small `eval`
split (T39.1) drives the per-epoch learning curve so the real test splits
are touched exactly once, after training: ~20 questions drawn like test_in
(trained templates, flavor "in") + ~10 drawn like test_out (reserved
templates, flavor "out"), each question tagged with its flavor.
Instance-level disjointness everywhere. Deterministic per seed."""
import argparse
import json
import random
from dataclasses import dataclass, field
from pathlib import Path

from kb.store import TemplateSummarizer

FIRST_M = ["Arthur", "Bennett", "Caleb", "Dorian", "Edmund", "Felix", "Gideon",
           "Harlan", "Ivor", "Jasper", "Kellan", "Lowell", "Merritt", "Nolan",
           "Osric", "Pascal", "Quentin", "Rufus", "Silas", "Thaddeus"]
FIRST_F = ["Adelia", "Beatrix", "Cordelia", "Delphine", "Elowen", "Fern",
           "Ginevra", "Hazel", "Isolde", "Junia", "Katriel", "Leonora",
           "Maribel", "Nerissa", "Odette", "Petra", "Quilla", "Rosalind",
           "Sylvie", "Thea"]
LAST = ["Abernathy", "Blackwood", "Crowhurst", "Dunmore", "Everhart", "Fenwick",
        "Grimsby", "Hollowell", "Ingleside", "Jessup", "Kirkwell", "Lockridge",
        "Marchbanks", "Netherfield", "Oakhurst", "Pemberly", "Quixwood",
        "Ravenscroft", "Silverton", "Thistlewood", "Underhill", "Vexley",
        "Wetherby", "Yarrow", "Zellwood", "Ashgrove", "Briarcliff", "Coldwater",
        "Dovecote", "Elmsworth", "Foxglove", "Gorseland", "Hawthorne",
        "Ironwood", "Juniper", "Kestrel", "Larkspur", "Mosswood", "Nightingale",
        "Osprey"]
JOBS = ["arborist", "beekeeper", "cartographer", "distiller", "engraver",
        "falconer", "glassblower", "horologist", "illustrator", "jeweler",
        "locksmith", "milliner", "notary", "oboist", "potter", "quarryman",
        "roofer", "saddler", "tanner", "upholsterer", "violinist", "weaver",
        "zookeeper", "blacksmith", "cooper", "dyer", "farrier", "gilder",
        "harpist", "innkeeper", "joiner", "knifemaker", "lamplighter", "mason",
        "navigator", "organist", "printer", "quilter", "ropemaker",
        "shipwright", "tailor", "undertaker", "vintner", "wheelwright",
        "archivist", "bookbinder", "chandler", "draper", "embroiderer",
        "fletcher"]
HOBBIES = ["archery", "birdwatching", "calligraphy", "chess", "cycling",
           "embroidery", "fencing", "fishing", "gardening", "geocaching",
           "hiking", "juggling", "kayaking", "kite flying", "knitting",
           "macrame", "model trains", "mushroom foraging", "origami",
           "painting", "photography", "pottery", "puzzles", "quilting",
           "rock climbing", "rowing", "sailing", "sculpting", "singing",
           "skating", "sketching", "stargazing", "swimming", "tapestry",
           "topiary", "whittling", "woodturning", "yoga", "astronomy",
           "baking"]
CITIES = ["Ambleford", "Brackenwick", "Cindervale", "Dorringham", "Eastmere",
          "Fernholt", "Gullsworth", "Harrowgate", "Ivorton", "Jasperfield",
          "Kestrelmoor", "Lundenwick", "Marrowdale", "Northcliff", "Otterby",
          "Pinehaven", "Quillbrook", "Ravensmoor", "Stonewick", "Thornbury"]

_NUM = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight",
        "nine", "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen",
        "sixteen", "seventeen", "eighteen", "nineteen", "twenty"]


def _num_variants(n: int) -> list[str]:
    return [str(n), _NUM[n]] if n < len(_NUM) else [str(n)]


@dataclass
class Person:
    pid: int
    name: str
    gender: str
    job: str
    hobby: str
    city: str
    spouse: int | None = None
    parents: list[int] = field(default_factory=list)
    children: list[int] = field(default_factory=list)
    friends: list[int] = field(default_factory=list)


@dataclass
class Question:
    qid: str
    template: str
    category: str
    hops: int
    text: str
    golds: list[str]
    support: list[str]
    unanswerable: bool = False
    eval_flavor: str | None = None      # "in" | "out" on eval-split questions


@dataclass
class Universe:
    meta: dict
    docs: list[dict]
    questions: dict[str, Question]
    splits: dict[str, list[str]]
    vocab: dict

    def to_json(self) -> dict:
        return {"meta": self.meta, "docs": self.docs,
                "questions": [vars(q) for q in self.questions.values()],
                "splits": self.splits, "vocab": self.vocab}

    @classmethod
    def from_json(cls, data: dict) -> "Universe":
        qs = {q["qid"]: Question(**q) for q in data["questions"]}
        return cls(data["meta"], data["docs"], qs, data["splits"], data["vocab"])

    def save(self, path) -> None:
        with open(path, "w") as f:
            json.dump(self.to_json(), f, ensure_ascii=False)

    @classmethod
    def load(cls, path) -> "Universe":
        with open(path) as f:
            return cls.from_json(json.load(f))


# ---------------- people + facts ----------------

def _people(rng: random.Random, n: int) -> list[Person]:
    combos = [(f, l) for l in LAST for f in FIRST_M + FIRST_F]
    rng.shuffle(combos)
    return [Person(i, f"{f} {l}", "m" if f in FIRST_M else "f",
                   rng.choice(JOBS), rng.choice(HOBBIES), rng.choice(CITIES))
            for i, (f, l) in enumerate(combos[:n])]


def _families(rng: random.Random, people: list[Person]) -> None:
    """Deterministic family structure that guarantees every question class
    exists even in tiny universes: G0 couples, G1 = their children (some
    married to each other), G2 = children of G1 couples, leftovers single
    and parentless (QC5 fodder). Then Erdos-Renyi friendships (~degree 4)."""
    n = len(people)
    ms = [p for p in people if p.gender == "m"]
    fs = [p for p in people if p.gender == "f"]
    c0 = max(1, n // 6)
    couples0 = list(zip(ms[:c0], fs[:c0]))
    for a, b in couples0:
        a.spouse, b.spouse = b.pid, a.pid
    used = {p.pid for pair in couples0 for p in pair}
    rest = [p for p in people if p.pid not in used]
    n1 = max(2, n // 3)
    g1, g2 = rest[:n1], rest[n1:]

    def _adopt(kids, couples):
        room = [list(pair) for pair in couples]
        for i, kid in enumerate(kids):
            pair = room[i // 3] if i // 3 < len(room) else None
            if pair is None:
                continue                       # leftover: parentless single
            kid.parents = [pair[0].pid, pair[1].pid]
            pair[0].children.append(kid.pid)
            pair[1].children.append(kid.pid)

    _adopt(g1, couples0)
    # marry some G1 males to G1 females from different parents
    g1m = [p for p in g1 if p.gender == "m"]
    g1f = [p for p in g1 if p.gender == "f"]
    couples1 = []
    for m in g1m:
        for f in g1f:
            if f.spouse is None and not (set(m.parents) & set(f.parents)):
                m.spouse, f.spouse = f.pid, m.pid
                couples1.append((m, f))
                break
        if len(couples1) >= max(1, n1 // 4):
            break
    _adopt(g2, couples1)
    for i in range(n):
        for j in range(i + 1, n):
            if rng.random() < 4.0 / n:
                people[i].friends.append(j)
                people[j].friends.append(i)


def _distractors(rng: random.Random, people: list[Person], frac: float) -> list[Person]:
    """Extra people whose FIRST names collide with real subjects; their full
    names never equal a real person's, so full-name questions stay unique."""
    n = round(len(people) * frac)
    if n <= 0:
        return []
    firsts = sorted({p.name.split()[0] for p in people})
    taken = {p.name for p in people}
    extras, pid = [], len(people)
    while len(extras) < n:
        f = rng.choice(firsts)
        name = f"{f} {rng.choice(LAST)}"
        if name in taken:
            continue
        taken.add(name)
        extras.append(Person(pid, name, "m" if f in FIRST_M else "f",
                             rng.choice(JOBS), rng.choice(HOBBIES),
                             rng.choice(CITIES)))
        pid += 1
    return extras


def _statement_groups(people: list[Person],
                      extras: list[Person]) -> tuple[list, dict]:
    """Per-person statement groups (real people first, then distractors, sid
    order fixed) + the fact-key -> origin-sid index for support bookkeeping.
    Arrangement into docs is a separate, init-dependent step."""
    by = {p.pid: p for p in people}
    groups, index = [], {}
    sid_n = 0

    def _add(stmts, key, text):
        nonlocal sid_n
        sid_n += 1
        sid = f"s{sid_n:04d}"
        stmts.append({"sid": sid, "text": text, "origin": sid})
        index[key] = sid

    for p in people:
        stmts = []
        _add(stmts, ("attr", p.pid, "job"), f"{p.name}'s job is {p.job}.")
        _add(stmts, ("attr", p.pid, "hobby"), f"{p.name}'s hobby is {p.hobby}.")
        _add(stmts, ("attr", p.pid, "city"), f"{p.name} lives in the city of {p.city}.")
        if p.spouse is not None:
            _add(stmts, ("married", p.pid), f"{p.name} is married to {by[p.spouse].name}.")
        for par in sorted(p.parents):
            _add(stmts, ("childof", p.pid, par), f"{p.name} is a child of {by[par].name}.")
        for c in p.children:
            role = "father" if p.gender == "m" else "mother"
            _add(stmts, ("parentof", p.pid, c), f"{p.name} is the {role} of {by[c].name}.")
        for fr in sorted(p.friends):
            _add(stmts, ("friend", p.pid, fr), f"{p.name} is a friend of {by[fr].name}.")
        groups.append((p, stmts))
    for p in extras:
        stmts = []
        _add(stmts, ("attr", p.pid, "job"), f"{p.name}'s job is {p.job}.")
        _add(stmts, ("attr", p.pid, "hobby"), f"{p.name}'s hobby is {p.hobby}.")
        _add(stmts, ("attr", p.pid, "city"), f"{p.name} lives in the city of {p.city}.")
        groups.append((p, stmts))
    return groups, index


def _arrange_entity(groups: list, summarizer) -> list[dict]:
    """One doc per person, zero links — the near-optimal organization."""
    return [{"id": f"d{i + 1:03d}",
             "summary": summarizer.summarize([s["text"] for s in stmts]),
             "statements": stmts, "links": []}
            for i, (_, stmts) in enumerate(groups)]


SUMMARY_CHARS = 80             # atomized init: summary = statement <=80 chars


def _arrange_atomized(groups: list, summarizer=None) -> list[dict]:
    """Every statement gets its own single-statement doc, zero links (T39.5).
    Summary choice: the doc IS the statement, so the template summary is the
    truncated statement text (<=80 chars, 77 + "...") — any generated summary
    of a one-sentence doc could only paraphrase it. An explicitly passed
    (LLM) summarizer is still honored, like the other inits. No rng involved,
    so questions/golds/splits/statements stay byte-identical across inits at
    equal seed."""
    docs, n = [], 0
    for _, stmts in groups:
        for st in stmts:
            n += 1
            text = st["text"]
            summary = (summarizer.summarize([text]) if summarizer is not None
                       else text if len(text) <= SUMMARY_CHARS
                       else text[:SUMMARY_CHARS - 3] + "...")
            docs.append({"id": f"d{n:03d}", "summary": summary,
                         "statements": [st], "links": []})
    return docs


LOCALITY = 0.30                # P(next scattered statement shares its person)
DOC_SIZE = (4, 8)              # min,max statements per scattered notebook doc


def _arrange_scattered(groups: list, seed: int, summarizer=None,
                       locality: float = LOCALITY,
                       doc_size: tuple[int, int] = DOC_SIZE) -> list[dict]:
    """The SAME statements in mixed notebook docs of doc_size[0]..[1]
    statements, ~n_statements/mean docs. Uses its own derived rng so the
    main stream (and therefore the questions/golds/splits) is identical to
    the entity init. Summaries stay generic — naming the people inside
    would gift retrieval the organization training is supposed to earn —
    unless an explicit (LLM) summarizer was passed."""
    mn, mx = doc_size
    if not 1 <= mn <= mx:
        raise ValueError(f"bad doc_size {doc_size}")
    arr = random.Random(seed * 7919 + 13)
    remaining = {p.pid: list(stmts) for p, stmts in groups if stmts}
    order, cur = [], None
    while remaining:
        if not (cur in remaining and arr.random() < locality):
            cur = arr.choice(sorted(remaining))
        order.append(remaining[cur].pop(0))
        if not remaining[cur]:
            del remaining[cur]
    chunks, i = [], 0
    while i < len(order):
        size = arr.randint(mn, mx)
        chunks.append(order[i:i + size])
        i += size
    if len(chunks) > 1 and len(chunks[-1]) < mn:     # fold a tiny tail in
        tail = chunks.pop()
        chunks[-1].extend(tail)
    docs = []
    for j, chunk in enumerate(chunks):
        texts = [s["text"] for s in chunk]
        summary = (summarizer.summarize(texts) if summarizer is not None
                   else f"Mixed notes ({len(chunk)} statements).")
        docs.append({"id": f"d{j + 1:03d}", "summary": summary,
                     "statements": chunk, "links": []})
    return docs


# ---------------- question templates ----------------

def _father(by, p):
    male = [by[i] for i in p.parents if by[i].gender == "m"]
    return male[0] if male else None


def _mother(by, p):
    fem = [by[i] for i in p.parents if by[i].gender == "f"]
    return fem[0] if fem else None


def _siblings(by, p):
    out = []
    for q in by.values():
        if q.pid != p.pid and q.parents and set(q.parents) & set(p.parents):
            out.append(q)
    return out


def _instances(template: str, people: list[Person], idx: dict) -> list[dict]:
    """All eligible instances of one template, in pid order. Each instance:
    text, golds (+variants), support (origin sids), unanswerable."""
    by = {p.pid: p for p in people}
    out = []
    for p in people:
        row = _one_instance(template, p, by, idx)
        if row is not None:
            out.append(row)
    return out


def _one_instance(t: str, p, by, idx):
    def sup(*keys):
        return [idx[k] for k in keys]

    if t == "qc1_job":
        return {"text": f"What is the job of {p.name}?", "golds": [p.job],
                "support": sup(("attr", p.pid, "job"))}
    if t == "qc1_hobby":
        return {"text": f"What is the hobby of {p.name}?", "golds": [p.hobby],
                "support": sup(("attr", p.pid, "hobby"))}
    if t == "qc1_city":
        return {"text": f"In which city does {p.name} live?", "golds": [p.city],
                "support": sup(("attr", p.pid, "city"))}
    if t == "qc2_spouse_job":
        if p.spouse is None:
            return None
        s = by[p.spouse]
        return {"text": f"What is the job of the spouse of {p.name}?",
                "golds": [s.job],
                "support": sup(("married", p.pid), ("attr", s.pid, "job"))}
    if t == "qc2_father_hobby":
        f = _father(by, p)
        if f is None:
            return None
        return {"text": f"What is the hobby of the father of {p.name}?",
                "golds": [f.hobby],
                "support": sup(("parentof", f.pid, p.pid), ("attr", f.pid, "hobby"))}
    if t == "qc2_mother_city":
        m = _mother(by, p)
        if m is None:
            return None
        return {"text": f"In which city does the mother of {p.name} live?",
                "golds": [m.city],
                "support": sup(("parentof", m.pid, p.pid), ("attr", m.pid, "city"))}
    if t == "qc3_spouse_father_hobby":
        if p.spouse is None:
            return None
        s = by[p.spouse]
        f = _father(by, s)
        if f is None:
            return None
        return {"text": f"What is the hobby of the father of the spouse of {p.name}?",
                "golds": [f.hobby],
                "support": sup(("married", p.pid), ("parentof", f.pid, s.pid),
                               ("attr", f.pid, "hobby"))}
    if t == "qc3_father_spouse_job":
        f = _father(by, p)
        if f is None or f.spouse is None:
            return None
        sp = by[f.spouse]
        return {"text": f"What is the job of the spouse of the father of {p.name}?",
                "golds": [sp.job],
                "support": sup(("parentof", f.pid, p.pid), ("married", f.pid),
                               ("attr", sp.pid, "job"))}
    if t == "qc3_spouse_mother_city":
        if p.spouse is None:
            return None
        s = by[p.spouse]
        m = _mother(by, s)
        if m is None:
            return None
        return {"text": f"In which city does the mother of the spouse of {p.name} live?",
                "golds": [m.city],
                "support": sup(("married", p.pid), ("parentof", m.pid, s.pid),
                               ("attr", m.pid, "city"))}
    if t == "qc4_children":
        if not p.children:
            return None
        return {"text": f"How many children does {p.name} have?",
                "golds": _num_variants(len(p.children)),
                "support": sup(*[("parentof", p.pid, c) for c in p.children])}
    if t == "qc4_grandchildren":
        grand = [g for c in p.children for g in by[c].children]
        if not grand:
            return None
        keys = [("parentof", p.pid, c) for c in p.children]
        keys += [("parentof", c, g) for c in p.children for g in by[c].children]
        return {"text": f"How many grandchildren does {p.name} have?",
                "golds": _num_variants(len(grand)), "support": sup(*keys)}
    if t == "qc4_friends":
        if not p.friends:
            return None
        return {"text": f"How many friends does {p.name} have?",
                "golds": _num_variants(len(p.friends)),
                "support": sup(*[("friend", p.pid, fr) for fr in sorted(p.friends)])}
    if t == "qc5_brother_job":
        if any(s.gender == "m" for s in _siblings(by, p)):
            return None
        return {"text": f"What is the job of the brother of {p.name}?",
                "golds": ["unknown"], "support": [], "unanswerable": True}
    if t == "qc5_sister_hobby":
        if any(s.gender == "f" for s in _siblings(by, p)):
            return None
        return {"text": f"What is the hobby of the sister of {p.name}?",
                "golds": ["unknown"], "support": [], "unanswerable": True}
    if t == "qc5_spouse_city":
        if p.spouse is not None:
            return None
        return {"text": f"In which city does the spouse of {p.name} live?",
                "golds": ["unknown"], "support": [], "unanswerable": True}
    raise ValueError(f"unknown template {t}")


# (template, category, hops). Reserved templates (test_out) carry True.
TEMPLATES: dict[str, tuple[str, int, bool]] = {
    "qc1_job": ("QC1", 1, False),
    "qc1_hobby": ("QC1", 1, False),
    "qc1_city": ("QC1", 1, True),
    "qc2_spouse_job": ("QC2", 2, False),
    "qc2_father_hobby": ("QC2", 2, False),
    "qc2_mother_city": ("QC2", 2, True),
    "qc3_spouse_father_hobby": ("QC3", 3, False),
    "qc3_father_spouse_job": ("QC3", 3, False),
    "qc3_spouse_mother_city": ("QC3", 3, True),
    "qc4_children": ("QC4", 1, False),
    "qc4_grandchildren": ("QC4", 2, False),
    "qc4_friends": ("QC4", 1, True),
    "qc5_brother_job": ("QC5", 2, False),
    "qc5_sister_hobby": ("QC5", 2, False),
    "qc5_spouse_city": ("QC5", 2, True),
}
TRAINED = [t for t, (_, _, res) in TEMPLATES.items() if not res]
RESERVED = [t for t, (_, _, res) in TEMPLATES.items() if res]


def _take(pools: dict[str, list], templates: list[str], n: int) -> list[tuple[str, dict]]:
    """Round-robin across templates so every category is represented as long
    as instances exist; consumes the pools (instance-level disjointness)."""
    out = []
    while len(out) < n and any(pools[t] for t in templates):
        for t in templates:
            if len(out) >= n:
                break
            if pools[t]:
                out.append((t, pools[t].pop(0)))
    return out


# ---------------- entry points ----------------

def build_universe(seed: int = 0, n_people: int = 120,
                   sizes: tuple[int, int, int] = (150, 100, 50),
                   eval_sizes: tuple[int, int] = (20, 10),
                   summarizer=None, init: str = "entity",
                   distractors: float = 0.15, locality: float = LOCALITY,
                   doc_size: tuple[int, int] = DOC_SIZE) -> Universe:
    if init not in ("entity", "scattered", "atomized"):
        raise ValueError(f"unknown init {init}")
    rng = random.Random(seed)
    custom_summarizer = summarizer
    summarizer = summarizer or TemplateSummarizer()
    people = _people(rng, n_people)
    _families(rng, people)
    extras = _distractors(rng, people, distractors)
    groups, idx = _statement_groups(people, extras)
    # arrangement never touches `rng`, so questions/golds/splits below are
    # identical across inits at the same seed
    if init == "scattered":
        docs = _arrange_scattered(groups, seed, custom_summarizer,
                                  locality, doc_size)
    elif init == "atomized":
        docs = _arrange_atomized(groups, custom_summarizer)
    else:
        docs = _arrange_entity(groups, summarizer)

    pools = {t: _instances(t, people, idx) for t in TEMPLATES}
    for t in TEMPLATES:
        rng.shuffle(pools[t])
    # eval continues consuming the SAME pools, so it is instance-disjoint
    # from every other split by construction
    picked = {"train": _take(pools, TRAINED, sizes[0]),
              "test_in": _take(pools, TRAINED, sizes[1]),
              "test_out": _take(pools, RESERVED, sizes[2])}
    ev = [("in", t, row) for t, row in _take(pools, TRAINED, eval_sizes[0])]
    ev += [("out", t, row) for t, row in _take(pools, RESERVED, eval_sizes[1])]

    questions, splits, qn = {}, {}, 0
    for split, rows in picked.items():
        splits[split] = []
        for t, row in rows:
            qn += 1
            qid = f"q{qn:04d}"
            cat, hops, _ = TEMPLATES[t]
            questions[qid] = Question(qid, t, cat, hops, row["text"],
                                      row["golds"], row["support"],
                                      row.get("unanswerable", False))
            splits[split].append(qid)
    splits["eval"] = []
    for flavor, t, row in ev:
        qn += 1
        qid = f"q{qn:04d}"
        cat, hops, _ = TEMPLATES[t]
        questions[qid] = Question(qid, t, cat, hops, row["text"],
                                  row["golds"], row["support"],
                                  row.get("unanswerable", False), flavor)
        splits["eval"].append(qid)

    meta = {"seed": seed, "n_people": n_people, "sizes": list(sizes),
            "eval_sizes": list(eval_sizes), "init": init,
            "distractors": distractors, "n_distractors": len(extras),
            "locality": locality, "doc_size": list(doc_size),
            "build_tokens": {"in": summarizer.tokens_in,
                             "out": summarizer.tokens_out}}
    vocab = {"jobs": JOBS, "hobbies": HOBBIES, "cities": CITIES,
             "templates": {t: {"category": c, "hops": h, "reserved": r}
                           for t, (c, h, r) in TEMPLATES.items()}}
    return Universe(meta, docs, questions, splits, vocab)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--people", type=int, default=120)
    ap.add_argument("--train", type=int, default=150)
    ap.add_argument("--test-in", type=int, default=100)
    ap.add_argument("--test-out", type=int, default=50)
    ap.add_argument("--eval-in", type=int, default=20,
                    help="eval-split questions from trained templates")
    ap.add_argument("--eval-out", type=int, default=10,
                    help="eval-split questions from reserved templates")
    ap.add_argument("--init", choices=["entity", "scattered", "atomized"],
                    default="entity",
                    help="doc arrangement: one doc per person, mixed notebook "
                         "docs of 4-8 statements, or one doc per statement")
    ap.add_argument("--distractors", type=float, default=0.15,
                    help="fraction of extra first-name-colliding people whose "
                         "statements pad the store but are never asked about")
    ap.add_argument("--locality", type=float, default=LOCALITY,
                    help="scattered init: P(consecutive statements share a "
                         "person)")
    ap.add_argument("--doc-size", default=f"{DOC_SIZE[0]},{DOC_SIZE[1]}",
                    help="scattered init: MIN,MAX statements per notebook doc")
    ap.add_argument("--out", required=True)
    ap.add_argument("--summarize", action="store_true",
                    help="generate build-time summaries with the LLM summarizer "
                         "(default: deterministic template, no API)")
    ap.add_argument("--model", default="gpt-5-mini")
    args = ap.parse_args(argv)

    summarizer = None
    if args.summarize:
        from kb.store import LLMSummarizer
        summarizer = LLMSummarizer(args.model)
    try:
        mn, mx = (int(x) for x in args.doc_size.split(","))
    except ValueError:
        raise SystemExit(f"--doc-size must be MIN,MAX, got {args.doc_size!r}")
    u = build_universe(args.seed, args.people,
                       (args.train, args.test_in, args.test_out),
                       (args.eval_in, args.eval_out), summarizer,
                       args.init, args.distractors, args.locality, (mn, mx))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    u.save(out / "universe.json")
    print(json.dumps({"people": args.people,
                      "init": args.init,
                      "distractors": u.meta["n_distractors"],
                      "docs": len(u.docs),
                      "statements": sum(len(d["statements"]) for d in u.docs),
                      "questions": len(u.questions),
                      "splits": {s: len(q) for s, q in u.splits.items()},
                      "build_tokens": u.meta["build_tokens"]}, indent=2))


if __name__ == "__main__":
    main()
