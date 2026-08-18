"""Universe generator (spec §6), PhantomWiki-inspired (fictional persons,
zero contamination, no Prolog — question classes are Python-derivable).

Entities: n_people fictional persons in small family trees (2-3 generations,
spouse links, <=3 children per couple) plus Erdos-Renyi friendships; job /
hobby / city attributes from fixed vocabularies. Facts render into
self-contained, pronoun-free statements; relations render one statement per
endpoint person (two distinct origins, intentionally).

Initial store (v9.6, node graph — the doc level and its entity / scattered /
atomized --init machinery are gone): every statement is one NODE
{id, text, origin=id}, zero links. Deterministic per seed; support sets
reference origin ids, which ARE the initial node ids.

Name distractors (T39.2): a fraction `distractors` of extra people whose
FIRST names collide with real subjects but whose full names never equal a
questioned person's; they get attribute statements in the store and are
never asked about — naive retrieval precision degrades, gold validity
doesn't.

Question classes (T39.7 split them into EASY and HARD tiers — the easy tier
alone was solved natively by statement-level retrieval, untrained F1 0.80):
- EASY QC1-5: named-entity forward chains (attribute, 2-hop, 3-hop, simple
  aggregation, unanswerable). Every answer sits at the end of a chain that
  starts from a name in the question.
- HARD QC6-10: the answer lives in relations BETWEEN statements, golds
  computed by pure graph traversal over the fact layer:
  QC6 multi-constraint join (no name anchor; only UNIQUE-answer instances),
  QC7 set intersection (unique preferred; <=2 answers allowed with
  comma-joined gold + each-name variants), QC8 deep aggregation
  (grandchildren — moved here from QC4 — friend-job counts, spouse's friend
  count), QC9 comparison/superlative over birthdates (dates are globally
  DISTINCT by construction — generation-banded years, rejection-sampled —
  so no ties exist anywhere), QC10 reverse lookup (no forward anchor; only
  emitted when exactly one person qualifies).

Two-level split scheme: 10 categories, 26 concrete templates, exactly one
reserved template per category. train (150) = 40% easy / 60% hard, i.i.d.
over the trained templates of each tier (all categories represented);
test_in (100) and the eval "in" flavor mirror the same 40/60 mix with
unseen instances; test_out (50) = the 10 reserved templates, never used in
train. The small `eval` split (T39.1) drives the per-epoch learning curve:
~20 questions drawn like test_in (flavor "in") + ~10 like test_out
(flavor "out"). Instance-level disjointness everywhere. Deterministic per
seed (question/gold identity with pre-T39.7 builds is intentionally
broken: the rng stream now also feeds birthdates)."""
import argparse
import json
import random
from dataclasses import dataclass, field
from pathlib import Path

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

_MONTHS = ["January", "February", "March", "April", "May", "June", "July",
           "August", "September", "October", "November", "December"]


def _num_variants(n: int) -> list[str]:
    return [str(n), _NUM[n]] if n < len(_NUM) else [str(n)]


def _fmt_date(born: tuple[int, int, int]) -> str:
    y, m, d = born
    return f"{_MONTHS[m - 1]} {d}, {y}"


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
    born: tuple[int, int, int] | None = None    # (year, month, day), distinct


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
    nodes: list[dict]
    questions: dict[str, Question]
    splits: dict[str, list[str]]
    vocab: dict

    def to_json(self) -> dict:
        return {"meta": self.meta, "nodes": self.nodes,
                "questions": [vars(q) for q in self.questions.values()],
                "splits": self.splits, "vocab": self.vocab}

    @classmethod
    def from_json(cls, data: dict) -> "Universe":
        qs = {q["qid"]: Question(**q) for q in data["questions"]}
        return cls(data["meta"], data["nodes"], qs, data["splits"], data["vocab"])

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
    and parentless (QC5 fodder). Then Erdos-Renyi friendships (~degree 4).
    T39.7 knob change: the G1 marriage cap rose from n1//4 to n1//2 — more
    G1 couples means more G2 children spread across more grandparents, which
    QC8's grandchildren counts need for split-sized instance pools (14 ->
    ~30 instances at 120 people)."""
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
        if len(couples1) >= max(1, n1 // 2):    # T39.7: was n1 // 4
            break
    _adopt(g2, couples1)
    for i in range(n):
        for j in range(i + 1, n):
            if rng.random() < 4.0 / n:
                people[i].friends.append(j)
                people[j].friends.append(i)


def _birthdates(rng: random.Random, people: list[Person],
                extras: list[Person]) -> None:
    """QC9 substrate: globally DISTINCT (year, month, day) triples via
    rejection sampling — no two people share a birthdate, so comparisons and
    superlatives are unique by construction (the spec's "perturb equal dates"
    taken to its limit). Years are banded by family-tree depth (G0 1920-49,
    G1 1950-79, G2 1980-2009) so parents predate their children; distractors
    draw from the two younger bands."""
    by = {p.pid: p for p in people}

    def depth(p):
        d = 0
        while p.parents:
            p = by[p.parents[0]]
            d += 1
        return d

    used: set[tuple[int, int, int]] = set()

    def pick(lo, hi):
        while True:
            b = (rng.randint(lo, hi), rng.randint(1, 12), rng.randint(1, 28))
            if b not in used:
                used.add(b)
                return b

    bands = {0: (1920, 1949), 1: (1950, 1979), 2: (1980, 2009)}
    for p in people:
        p.born = pick(*bands[min(depth(p), 2)])
    for e in extras:
        e.born = pick(1950, 2009)


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


def _statement_nodes(people: list[Person],
                     extras: list[Person]) -> tuple[list, dict]:
    """One node per statement (real people first, then distractors, id order
    fixed; origin = own id, zero links) + the fact-key -> origin-id index
    for support bookkeeping."""
    by = {p.pid: p for p in people}
    nodes, index = [], {}
    sid_n = 0

    def _add(stmts, key, text):
        nonlocal sid_n
        sid_n += 1
        sid = f"s{sid_n:04d}"
        stmts.append({"id": sid, "text": text, "origin": sid})
        index[key] = sid

    for p in people:
        stmts = []
        _add(stmts, ("attr", p.pid, "job"), f"{p.name}'s job is {p.job}.")
        _add(stmts, ("attr", p.pid, "hobby"), f"{p.name}'s hobby is {p.hobby}.")
        _add(stmts, ("attr", p.pid, "city"), f"{p.name} lives in the city of {p.city}.")
        _add(stmts, ("born", p.pid), f"{p.name} was born on {_fmt_date(p.born)}.")
        if p.spouse is not None:
            _add(stmts, ("married", p.pid), f"{p.name} is married to {by[p.spouse].name}.")
        for par in sorted(p.parents):
            _add(stmts, ("childof", p.pid, par), f"{p.name} is a child of {by[par].name}.")
        for c in p.children:
            role = "father" if p.gender == "m" else "mother"
            _add(stmts, ("parentof", p.pid, c), f"{p.name} is the {role} of {by[c].name}.")
        for fr in sorted(p.friends):
            _add(stmts, ("friend", p.pid, fr), f"{p.name} is a friend of {by[fr].name}.")
        nodes.extend(stmts)
    for p in extras:
        stmts = []
        _add(stmts, ("attr", p.pid, "job"), f"{p.name}'s job is {p.job}.")
        _add(stmts, ("attr", p.pid, "hobby"), f"{p.name}'s hobby is {p.hobby}.")
        _add(stmts, ("attr", p.pid, "city"), f"{p.name} lives in the city of {p.city}.")
        _add(stmts, ("born", p.pid), f"{p.name} was born on {_fmt_date(p.born)}.")
        nodes.extend(stmts)
    return nodes, index


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


def _instances(template: str, people: list[Person], extras: list[Person],
               idx: dict) -> list[dict]:
    """All eligible instances of one template, in deterministic (pid) order.
    Each instance: text, golds (+variants), support (origin ids),
    unanswerable. Hard templates (QC6-10) enumerate combinations over the
    whole fact layer instead of iterating single persons."""
    by = {p.pid: p for p in people}
    if template in _ENUMERATED:
        return _ENUMERATED[template](people, extras, by, idx)
    out = []
    for p in people:
        row = _one_instance(template, p, by, idx)
        if row is not None:
            out.append(row)
    return out


# ---------------- hard-class enumerators (T39.7) ----------------
# Golds are computed by pure graph traversal over the fact layer; every
# instance carries the FULL enumeration its answer depends on in `support`
# (e.g. all of X's friend statements plus every friend's job statement for a
# friend-job count). QC6/QC10 emit ONLY instances with a unique answer,
# checked against real people AND distractors; QC9 can never tie (distinct
# birthdates by construction); QC7 prefers unique answers but also emits
# 2-answer instances with a comma-joined gold plus each-name variants.

def _qc6_join(attrs: tuple[str, ...], text_fn):
    def gen(people, extras, by, idx):
        out = []
        everyone = people + extras
        for p in people:
            key = tuple(getattr(p, a) for a in attrs)
            matches = [q for q in everyone
                       if tuple(getattr(q, a) for a in attrs) == key]
            if matches != [p]:              # unique across the WHOLE store
                continue
            out.append({"text": text_fn(p), "golds": [p.name],
                        "support": [idx[("attr", p.pid, a)] for a in attrs]})
        return out
    return gen


def _qc7_common_friend(people, extras, by, idx):
    out = []
    for x in people:
        for y in people:
            if y.pid <= x.pid:
                continue
            common = sorted(set(x.friends) & set(y.friends))
            if not 1 <= len(common) <= 2:
                continue
            names = [by[c].name for c in common]
            golds = [", ".join(names)] + (names if len(names) > 1 else [])
            support = ([idx[("friend", x.pid, f)] for f in sorted(x.friends)]
                       + [idx[("friend", y.pid, f)] for f in sorted(y.friends)])
            out.append({"text": f"Which friend of {x.name} is also a friend "
                                f"of {y.name}?",
                        "golds": golds, "support": support})
    return out


def _qc7_child_friend(people, extras, by, idx):
    out = []
    for x in people:
        if not x.children:
            continue
        for y in people:
            if y.pid == x.pid:
                continue
            qual = [c for c in x.children if c in y.friends]
            if not 1 <= len(qual) <= 2:
                continue
            names = [by[c].name for c in qual]
            golds = [", ".join(names)] + (names if len(names) > 1 else [])
            support = ([idx[("parentof", x.pid, c)] for c in x.children]
                       + [idx[("friend", y.pid, f)] for f in sorted(y.friends)])
            out.append({"text": f"Which child of {x.name} is a friend of "
                                f"{y.name}?",
                        "golds": golds, "support": support})
    return out


def _qc8_friend_job(people, extras, by, idx):
    out = []
    for x in people:
        jobs: dict[str, int] = {}
        for f in sorted(x.friends):
            jobs[by[f].job] = jobs.get(by[f].job, 0) + 1
        support = ([idx[("friend", x.pid, f)] for f in sorted(x.friends)]
                   + [idx[("attr", f, "job")] for f in sorted(x.friends)])
        for job in sorted(jobs):
            out.append({"text": f"How many friends of {x.name} have the job "
                                f"of {job}?",
                        "golds": _num_variants(jobs[job]), "support": support})
    return out


def _qc8_spouse_friends(people, extras, by, idx):
    out = []
    for x in people:
        if x.spouse is None:
            continue
        s = by[x.spouse]
        if not s.friends:
            continue
        support = ([idx[("married", x.pid)]]
                   + [idx[("friend", s.pid, f)] for f in sorted(s.friends)])
        out.append({"text": f"How many friends does the spouse of {x.name} "
                            f"have?",
                    "golds": _num_variants(len(s.friends)), "support": support})
    return out


def _qc9_older_pair(people, extras, by, idx):
    out = []
    for x in people:
        for j in sorted(x.friends):
            if j <= x.pid:
                continue
            y = by[j]
            older = x if x.born < y.born else y
            out.append({"text": f"Who is older, {x.name} or {y.name}?",
                        "golds": [older.name],
                        "support": [idx[("born", x.pid)],
                                    idx[("born", y.pid)]]})
    return out


def _qc10_spouse_job_city(people, extras, by, idx):
    """Reverse lookup: the subject is described only by attributes; the
    answer is their spouse. Unique because the (job, city) pair identifies
    exactly one person in the whole store."""
    out = []
    everyone = people + extras
    for p in people:
        if p.spouse is None:
            continue
        matches = [q for q in everyone
                   if (q.job, q.city) == (p.job, p.city)]
        if matches != [p]:
            continue
        out.append({"text": f"Whose spouse is the {p.job} who lives in the "
                            f"city of {p.city}?",
                    "golds": [by[p.spouse].name],
                    "support": [idx[("attr", p.pid, "job")],
                                idx[("attr", p.pid, "city")],
                                idx[("married", p.pid)]]})
    return out


def _qc10_spouse_job(people, extras, by, idx):
    """Whose spouse has the job of <job>? — emitted only when exactly one
    person qualifies (one married person whose spouse holds the job)."""
    holders: dict[str, list[Person]] = {}
    for x in people:
        if x.spouse is not None:
            holders.setdefault(by[x.spouse].job, []).append(x)
    out = []
    for job in sorted(holders):
        if len(holders[job]) != 1:
            continue
        x = holders[job][0]
        out.append({"text": f"Whose spouse has the job of {job}?",
                    "golds": [x.name],
                    "support": [idx[("married", x.pid)],
                                idx[("attr", x.spouse, "job")]]})
    return out


_ENUMERATED = {
    "qc6_job_city": _qc6_join(
        ("job", "city"),
        lambda p: f"Who is the {p.job} who lives in the city of {p.city}?"),
    "qc6_job_hobby": _qc6_join(
        ("job", "hobby"),
        lambda p: f"Who is the {p.job} whose hobby is {p.hobby}?"),
    "qc6_job_city_hobby": _qc6_join(
        ("job", "city", "hobby"),
        lambda p: f"Who is the {p.job} in the city of {p.city} whose hobby "
                  f"is {p.hobby}?"),
    "qc7_common_friend": _qc7_common_friend,
    "qc7_child_friend": _qc7_child_friend,
    "qc8_friend_job": _qc8_friend_job,
    "qc8_spouse_friends": _qc8_spouse_friends,
    "qc9_older_pair": _qc9_older_pair,
    "qc10_spouse_job_city": _qc10_spouse_job_city,
    "qc10_spouse_job": _qc10_spouse_job,
}


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
    if t == "qc8_grandchildren":        # moved from QC4 (T39.7): deep count
        grand = [g for c in p.children for g in by[c].children]
        if not grand:
            return None
        keys = [("parentof", p.pid, c) for c in p.children]
        keys += [("parentof", c, g) for c in p.children for g in by[c].children]
        return {"text": f"How many grandchildren does {p.name} have?",
                "golds": _num_variants(len(grand)), "support": sup(*keys)}
    if t == "qc9_oldest_child":
        if len(p.children) < 2:                     # 1 child would be trivial
            return None
        oldest = min((by[c] for c in p.children), key=lambda k: k.born)
        keys = [("parentof", p.pid, c) for c in p.children]
        keys += [("born", c) for c in p.children]
        return {"text": f"Who is the oldest child of {p.name}?",
                "golds": [oldest.name], "support": sup(*keys)}
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
    # ---- easy tier QC1-5: named-entity forward chains ----
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
    "qc4_friends": ("QC4", 1, True),
    "qc5_brother_job": ("QC5", 2, False),
    "qc5_sister_hobby": ("QC5", 2, False),
    "qc5_spouse_city": ("QC5", 2, True),
    # ---- hard tier QC6-10: relations BETWEEN statements (T39.7) ----
    "qc6_job_city": ("QC6", 2, False),
    "qc6_job_hobby": ("QC6", 2, False),
    "qc6_job_city_hobby": ("QC6", 3, True),
    "qc7_common_friend": ("QC7", 2, False),
    "qc7_child_friend": ("QC7", 2, True),
    "qc8_grandchildren": ("QC8", 2, False),
    "qc8_friend_job": ("QC8", 2, False),
    "qc8_spouse_friends": ("QC8", 2, True),
    "qc9_older_pair": ("QC9", 1, False),
    "qc9_oldest_child": ("QC9", 2, True),
    "qc10_spouse_job_city": ("QC10", 2, False),
    "qc10_spouse_job": ("QC10", 2, True),
}
HARD_CATS = {"QC6", "QC7", "QC8", "QC9", "QC10"}
HARD_SHARE = 0.60                  # hard fraction of train/test_in/eval-in
TRAINED = [t for t, (_, _, res) in TEMPLATES.items() if not res]
RESERVED = [t for t, (_, _, res) in TEMPLATES.items() if res]
TRAINED_EASY = [t for t in TRAINED if TEMPLATES[t][0] not in HARD_CATS]
TRAINED_HARD = [t for t in TRAINED if TEMPLATES[t][0] in HARD_CATS]


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
                   distractors: float = 0.15) -> Universe:
    rng = random.Random(seed)
    people = _people(rng, n_people)
    _families(rng, people)
    extras = _distractors(rng, people, distractors)
    _birthdates(rng, people, extras)
    nodes, idx = _statement_nodes(people, extras)

    pools = {t: _instances(t, people, extras, idx) for t in TEMPLATES}
    for t in TEMPLATES:
        rng.shuffle(pools[t])

    def _mixed(n):
        """HARD_SHARE hard / rest easy, each side round-robin over its
        trained templates (T39.7 rebalance)."""
        n_hard = round(n * HARD_SHARE)
        return (_take(pools, TRAINED_EASY, n - n_hard)
                + _take(pools, TRAINED_HARD, n_hard))

    # eval continues consuming the SAME pools, so it is instance-disjoint
    # from every other split by construction; test_out draws from the
    # reserved templates of BOTH tiers (one reserved template per category)
    picked = {"train": _mixed(sizes[0]),
              "test_in": _mixed(sizes[1]),
              "test_out": _take(pools, RESERVED, sizes[2])}
    ev = [("in", t, row) for t, row in _mixed(eval_sizes[0])]
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
            "eval_sizes": list(eval_sizes),
            "distractors": distractors, "n_distractors": len(extras)}
    vocab = {"jobs": JOBS, "hobbies": HOBBIES, "cities": CITIES,
             "templates": {t: {"category": c, "hops": h, "reserved": r}
                           for t, (c, h, r) in TEMPLATES.items()}}
    return Universe(meta, nodes, questions, splits, vocab)


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
    ap.add_argument("--distractors", type=float, default=0.15,
                    help="fraction of extra first-name-colliding people whose "
                         "statements pad the store but are never asked about")
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    u = build_universe(args.seed, args.people,
                       (args.train, args.test_in, args.test_out),
                       (args.eval_in, args.eval_out), args.distractors)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    u.save(out / "universe.json")
    print(json.dumps({"people": args.people,
                      "distractors": u.meta["n_distractors"],
                      "nodes": len(u.nodes),
                      "questions": len(u.questions),
                      "splits": {s: len(q) for s, q in u.splits.items()}},
                     indent=2))


if __name__ == "__main__":
    main()
