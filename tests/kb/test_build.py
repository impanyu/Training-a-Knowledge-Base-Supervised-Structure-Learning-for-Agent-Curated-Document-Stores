import json
import re

from kb.build import (HARD_CATS, RESERVED, TEMPLATES, TRAINED, TRAINED_EASY,
                      TRAINED_HARD, Universe, build_universe, main)

from .fixtures import mini_universe


def _stmt_texts(u):
    return {n["text"] for n in u.nodes}


def _id_index(u):
    return {n["id"]: n for n in u.nodes}


_MONTH_N = {m: i + 1 for i, m in enumerate(
    ["January", "February", "March", "April", "May", "June", "July",
     "August", "September", "October", "November", "December"])}


def _facts(u):
    """Independent reconstruction of the fact layer from the statement texts
    alone — the hard-class tests recompute every gold from this, so a
    generator bug cannot certify itself."""
    f = {"job": {}, "hobby": {}, "city": {}, "born": {}, "spouse": {},
         "friends": {}, "children": {}}
    for n in u.nodes:
        t = n["text"].removesuffix(".")
        if "'s job is " in t:
            a, b = t.split("'s job is ")
            f["job"][a] = b
        elif "'s hobby is " in t:
            a, b = t.split("'s hobby is ")
            f["hobby"][a] = b
        elif " lives in the city of " in t:
            a, b = t.split(" lives in the city of ")
            f["city"][a] = b
        elif " was born on " in t:
            a, b = t.split(" was born on ")
            m = re.fullmatch(r"(\w+) (\d+), (\d+)", b)
            f["born"][a] = (int(m.group(3)), _MONTH_N[m.group(1)],
                            int(m.group(2)))
        elif " is married to " in t:
            a, b = t.split(" is married to ")
            f["spouse"][a] = b
        elif " is a friend of " in t:
            a, b = t.split(" is a friend of ")
            f["friends"].setdefault(a, set()).add(b)
        elif " is the father of " in t or " is the mother of " in t:
            a, b = re.fullmatch(r"(.+) is the (?:father|mother) of (.+)", t).groups()
            f["children"].setdefault(a, []).append(b)
    return f


def _by_template(u, template):
    qs = [q for q in u.questions.values() if q.template == template]
    assert qs, f"no {template} instances landed in any split"
    return qs


def test_deterministic_per_seed():
    a = build_universe(0, 12, (8, 6, 5), (3, 2)).to_json()
    b = build_universe(0, 12, (8, 6, 5), (3, 2)).to_json()
    assert a == b                              # questions/golds/nodes byte-equal
    assert build_universe(1, 12, (8, 6, 5), (3, 2)).to_json() != a


def test_two_level_template_scheme():
    cats = {c for c, _, _ in TEMPLATES.values()}
    assert cats == {f"QC{i}" for i in range(1, 11)}
    assert len(TEMPLATES) == 26
    assert len(RESERVED) == 10
    for cat in cats:                       # 1 reserved template per category
        assert sum(1 for t in RESERVED if TEMPLATES[t][0] == cat) == 1
    assert HARD_CATS == {"QC6", "QC7", "QC8", "QC9", "QC10"}
    assert len(TRAINED_EASY) == 9 and len(TRAINED_HARD) == 7
    assert set(TRAINED) == set(TRAINED_EASY) | set(TRAINED_HARD)


def test_splits_disjoint_and_sized():
    u = build_universe(0, 120)
    splits = u.splits
    all_qids = (splits["train"] + splits["test_in"] + splits["test_out"]
                + splits["eval"])
    assert len(all_qids) == len(set(all_qids)) == 330
    assert (len(splits["train"]), len(splits["test_in"]),
            len(splits["test_out"]), len(splits["eval"])) == (150, 100, 50, 30)
    # instance-level disjointness: no (template, text) appears twice
    keys = [(u.questions[q].template, u.questions[q].text) for q in all_qids]
    assert len(keys) == len(set(keys))


def test_eval_split_flavors_and_template_provenance():
    u = build_universe(0, 120)
    ev = [u.questions[q] for q in u.splits["eval"]]
    ins = [q for q in ev if q.eval_flavor == "in"]
    outs = [q for q in ev if q.eval_flavor == "out"]
    assert (len(ins), len(outs)) == (20, 10)
    assert {q.template for q in ins} <= set(TRAINED)    # sampled like test_in
    assert {q.template for q in outs} <= set(RESERVED)  # sampled like test_out
    # only eval questions carry a flavor
    for split in ("train", "test_in", "test_out"):
        assert all(u.questions[q].eval_flavor is None for q in u.splits[split])


def test_train_covers_all_categories_test_out_only_reserved():
    u = build_universe(0, 120)
    train_templates = {u.questions[q].template for q in u.splits["train"]}
    assert {TEMPLATES[t][0] for t in train_templates} == \
        {f"QC{i}" for i in range(1, 11)}          # never exclude a category
    assert train_templates <= set(TRAINED)
    in_templates = {u.questions[q].template for q in u.splits["test_in"]}
    assert in_templates <= set(TRAINED)           # trained templates, unseen instances
    out_templates = {u.questions[q].template for q in u.splits["test_out"]}
    assert out_templates <= set(RESERVED)
    assert not (train_templates | in_templates) & out_templates


def test_initial_store_is_one_node_per_statement_zero_links():
    u = mini_universe()
    assert u.meta["n_distractors"] == 2           # round(0.15 * 12)
    ids = [n["id"] for n in u.nodes]
    assert len(ids) == len(set(ids))              # globally unique
    for n in u.nodes:
        assert n["origin"] == n["id"]             # origin = own id at build
        assert n["text"].endswith(".")
        assert not n.get("links")                 # zero links


def test_qc1_gold_matches_the_statement():
    u = build_universe(0, 30)
    q = next(q for q in u.questions.values() if q.template == "qc1_job")
    name = q.text.removeprefix("What is the job of ").removesuffix("?")
    assert f"{name}'s job is {q.golds[0]}." in _stmt_texts(u)
    n = _id_index(u)[q.support[0]]
    assert n["text"] == f"{name}'s job is {q.golds[0]}."


def test_qc2_spouse_job_chain_is_derivable_from_support():
    u = build_universe(0, 30)
    q = next(q for q in u.questions.values() if q.template == "qc2_spouse_job")
    idx = _id_index(u)
    marriage, job = (idx[s] for s in q.support)
    assert " is married to " in marriage["text"]
    spouse = marriage["text"].split(" is married to ")[1].removesuffix(".")
    assert job["text"] == f"{spouse}'s job is {q.golds[0]}."


def test_qc4_aggregation_gold_has_digit_and_word_variants():
    u = build_universe(0, 30)
    q = next(q for q in u.questions.values() if q.template == "qc4_children")
    n = int(q.golds[0])
    assert n >= 1 and len(q.support) == n         # one parent-side origin per child
    assert len(q.golds) == 2 and not q.golds[1].isdigit()
    idx = _id_index(u)
    assert all(("father of" in idx[s]["text"]) or ("mother of" in idx[s]["text"])
               for s in q.support)


def test_qc5_unanswerable_shape():
    u = build_universe(0, 30)
    qc5 = [q for q in u.questions.values() if q.category == "QC5"]
    assert qc5
    for q in qc5:
        assert q.unanswerable and q.golds == ["unknown"] and q.support == []


def test_support_ids_are_real_initial_node_ids():
    u = build_universe(0, 60)
    idx = _id_index(u)
    for q in u.questions.values():
        for s in q.support:
            assert s in idx and idx[s]["origin"] == s


def test_relations_render_one_statement_per_endpoint():
    u = mini_universe()
    texts = _stmt_texts(u)
    marriage = next(t for t in texts if " is married to " in t)
    a, b = marriage.removesuffix(".").split(" is married to ")
    assert f"{b} is married to {a}." in texts     # mirrored, distinct origin


def _subject(text: str) -> str:
    return " ".join(text.split()[:2]).removesuffix("'s")


def test_distractors_collide_on_first_name_only_and_are_never_asked():
    u = build_universe(0, 120)
    assert u.meta["n_distractors"] == 18          # round(0.15 * 120)
    # each person contributes exactly one job statement, in generation order:
    # the 120 real people first, then the distractors
    subjects = [_subject(n["text"]) for n in u.nodes if "'s job is " in n["text"]]
    assert len(subjects) == 138
    real, extras = subjects[:120], subjects[120:]
    real_firsts = {n.split()[0] for n in real}
    for name in extras:
        assert name not in real                   # full name never collides
        assert name.split()[0] in real_firsts     # first name always does
        for q in u.questions.values():
            assert name not in q.text             # never asked about
    z = build_universe(0, 120, distractors=0.0)
    assert len([n for n in z.nodes if "'s job is " in n["text"]]) == 120
    assert z.meta["n_distractors"] == 0
    # distractor nodes append after real ones: real ids unchanged
    reals = [n for n in u.nodes if _subject(n["text"]) not in extras]
    assert [n["id"] for n in reals] == [n["id"] for n in z.nodes]


# ---------------- hard classes QC6-10 (T39.7) ----------------

def test_split_mix_ratios_and_reserved_tiers():
    u = build_universe(0, 120)

    def hard_n(qids):
        return sum(1 for q in qids if u.questions[q].category in HARD_CATS)

    assert len(u.splits["train"]) == 150 and hard_n(u.splits["train"]) == 90
    assert len(u.splits["test_in"]) == 100 and hard_n(u.splits["test_in"]) == 60
    ev = [u.questions[q] for q in u.splits["eval"]]
    ev_in = [q.qid for q in ev if q.eval_flavor == "in"]
    assert len(ev_in) == 20 and hard_n(ev_in) == 12    # mirrors the 40/60 mix
    # test_out draws evenly from the reserved templates of BOTH tiers
    out_cats = {}
    for q in u.splits["test_out"]:
        out_cats[u.questions[q].category] = out_cats.get(u.questions[q].category, 0) + 1
    assert out_cats == {f"QC{i}": 5 for i in range(1, 11)}


def test_qc6_join_answers_are_unique_and_recomputable():
    u = build_universe(0, 30)
    f = _facts(u)
    pats = {
        "qc6_job_city": (r"Who is the (.+) who lives in the city of (.+)\?",
                         ("job", "city")),
        "qc6_job_hobby": (r"Who is the (.+) whose hobby is (.+)\?",
                          ("job", "hobby")),
        "qc6_job_city_hobby": (
            r"Who is the (.+) in the city of (.+) whose hobby is (.+)\?",
            ("job", "city", "hobby")),
    }
    for template, (pat, attrs) in pats.items():
        for q in _by_template(u, template):
            vals = re.fullmatch(pat, q.text).groups()
            qual = [n for n in f["job"]
                    if all(f[a].get(n) == v for a, v in zip(attrs, vals))]
            assert qual == q.golds              # exactly one, and it's the gold
            sup = {_id_index(u)[s]["text"] for s in q.support}
            assert len(sup) == len(attrs)       # the identifying statements


def test_qc7_intersection_golds_variants_and_full_support():
    u = build_universe(0, 60)
    f, idx = _facts(u), _id_index(u)
    for q in _by_template(u, "qc7_common_friend"):
        x, y = re.fullmatch(r"Which friend of (.+) is also a friend of (.+)\?",
                            q.text).groups()
        common = f["friends"][x] & f["friends"][y]
        assert 1 <= len(common) <= 2
        assert set(q.golds[0].split(", ")) == common    # comma-joined gold
        if len(common) == 2:
            assert set(q.golds[1:]) == common           # each-name variants
        else:
            assert q.golds == [next(iter(common))]
        # support = the FULL enumeration: every friend statement of X and Y
        sup = {idx[s]["text"] for s in q.support}
        assert sup == ({f"{x} is a friend of {n}." for n in f["friends"][x]}
                       | {f"{y} is a friend of {n}." for n in f["friends"][y]})
    for q in _by_template(u, "qc7_child_friend"):
        x, y = re.fullmatch(r"Which child of (.+) is a friend of (.+)\?",
                            q.text).groups()
        qual = set(f["children"][x]) & f["friends"][y]
        assert 1 <= len(qual) <= 2
        assert set(q.golds[0].split(", ")) == qual


def test_qc8_deep_aggregation_counts_and_full_support():
    u = build_universe(0, 60)
    f, idx = _facts(u), _id_index(u)
    for q in _by_template(u, "qc8_friend_job"):
        x, job = re.fullmatch(r"How many friends of (.+) have the job of (.+)\?",
                              q.text).groups()
        count = sum(1 for n in f["friends"][x] if f["job"][n] == job)
        assert count >= 1
        assert q.golds[0] == str(count)
        assert len(q.golds) == 2 and not q.golds[1].isdigit()
        # support = all of X's friend statements + EVERY friend's job statement
        sup = {idx[s]["text"] for s in q.support}
        assert sup == ({f"{x} is a friend of {n}." for n in f["friends"][x]}
                       | {f"{n}'s job is {f['job'][n]}." for n in f["friends"][x]})
    for q in _by_template(u, "qc8_grandchildren"):
        x = re.fullmatch(r"How many grandchildren does (.+) have\?",
                         q.text).group(1)
        grand = [g for c in f["children"][x] for g in f["children"].get(c, [])]
        assert q.golds[0] == str(len(grand)) and len(grand) >= 1
        # full enumeration: one parent-side statement per child + grandchild
        assert len(q.support) == len(f["children"][x]) + len(grand)
    for q in _by_template(u, "qc8_spouse_friends"):
        x = re.fullmatch(r"How many friends does the spouse of (.+) have\?",
                         q.text).group(1)
        assert q.golds[0] == str(len(f["friends"][f["spouse"][x]]))


def test_qc9_birthdates_distinct_and_comparisons_recomputable():
    u = build_universe(0, 60)
    f = _facts(u)
    born = f["born"]
    assert len(set(born.values())) == len(born)        # no ties ANYWHERE
    for q in _by_template(u, "qc9_older_pair"):
        x, y = re.fullmatch(r"Who is older, (.+) or (.+)\?", q.text).groups()
        assert q.golds == [x if born[x] < born[y] else y]
    for q in _by_template(u, "qc9_oldest_child"):
        x = re.fullmatch(r"Who is the oldest child of (.+)\?", q.text).group(1)
        kids = f["children"][x]
        assert len(kids) >= 2                          # 1 child would be trivial
        assert q.golds == [min(kids, key=lambda k: born[k])]


def test_qc9_parents_predate_their_children():
    u = build_universe(0, 60)
    f = _facts(u)
    for parent, kids in f["children"].items():
        for k in kids:
            assert f["born"][parent] < f["born"][k]    # generation-banded years


def test_qc10_reverse_lookup_has_exactly_one_qualifier():
    u = build_universe(0, 60)
    f = _facts(u)
    for q in _by_template(u, "qc10_spouse_job"):
        job = re.fullmatch(r"Whose spouse has the job of (.+)\?", q.text).group(1)
        qual = [x for x, s in f["spouse"].items() if f["job"][s] == job]
        assert qual == q.golds                         # exactly one qualifier
    for q in _by_template(u, "qc10_spouse_job_city"):
        job, city = re.fullmatch(
            r"Whose spouse is the (.+) who lives in the city of (.+)\?",
            q.text).groups()
        described = [n for n in f["job"]
                     if f["job"][n] == job and f["city"][n] == city]
        assert len(described) == 1                     # unique across the store
        assert f["spouse"][described[0]] == q.golds[0]


def test_universe_json_roundtrip(tmp_path):
    u = mini_universe()
    u.save(tmp_path / "universe.json")
    v = Universe.load(tmp_path / "universe.json")
    assert v.to_json() == u.to_json()
    assert v.questions[v.splits["train"][0]].qid == u.splits["train"][0]


def test_build_cli_writes_universe(tmp_path, capsys):
    main(["--seed", "0", "--people", "12", "--train", "8", "--test-in", "6",
          "--test-out", "5", "--eval-in", "3", "--eval-out", "2",
          "--out", str(tmp_path)])
    u = Universe.load(tmp_path / "universe.json")
    assert u.meta["seed"] == 0
    assert u.meta["eval_sizes"] == [3, 2]
    stats = json.loads(capsys.readouterr().out)
    assert stats["nodes"] == len(u.nodes)
    assert stats["distractors"] == 2
    assert "eval" in stats["splits"]
