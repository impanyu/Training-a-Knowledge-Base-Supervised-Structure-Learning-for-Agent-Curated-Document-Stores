import json

from kb.build import (RESERVED, TEMPLATES, TRAINED, Universe, build_universe,
                      main)

from .fixtures import mini_universe


def _stmt_texts(u):
    return {st["text"] for d in u.docs for st in d["statements"]}


def _sid_index(u):
    return {st["sid"]: st for d in u.docs for st in d["statements"]}


def test_deterministic_per_seed():
    a = build_universe(0, 12, (8, 6, 5), (3, 2)).to_json()
    b = build_universe(0, 12, (8, 6, 5), (3, 2)).to_json()
    assert a == b
    assert build_universe(1, 12, (8, 6, 5), (3, 2)).to_json() != a


def test_two_level_template_scheme():
    cats = {c for c, _, _ in TEMPLATES.values()}
    assert cats == {"QC1", "QC2", "QC3", "QC4", "QC5"}
    assert len(TEMPLATES) == 15
    assert len(RESERVED) == 5
    for cat in cats:                       # 1 reserved template per category
        assert sum(1 for t in RESERVED if TEMPLATES[t][0] == cat) == 1


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
        {"QC1", "QC2", "QC3", "QC4", "QC5"}       # never exclude a category
    assert train_templates <= set(TRAINED)
    in_templates = {u.questions[q].template for q in u.splits["test_in"]}
    assert in_templates <= set(TRAINED)           # trained templates, unseen instances
    out_templates = {u.questions[q].template for q in u.splits["test_out"]}
    assert out_templates <= set(RESERVED)
    assert not (train_templates | in_templates) & out_templates


def test_initial_store_shape():
    u = mini_universe()
    assert len(u.docs) == 12                      # one doc per person
    assert all(d["links"] == [] for d in u.docs)  # zero links
    assert all(d["summary"].startswith("Facts about ") for d in u.docs)
    sids = [st["sid"] for d in u.docs for st in d["statements"]]
    assert len(sids) == len(set(sids))            # globally unique
    assert all(st["origin"] == st["sid"]
               for d in u.docs for st in d["statements"])


def test_qc1_gold_matches_the_statement():
    u = build_universe(0, 30)
    q = next(q for q in u.questions.values() if q.template == "qc1_job")
    name = q.text.removeprefix("What is the job of ").removesuffix("?")
    assert f"{name}'s job is {q.golds[0]}." in _stmt_texts(u)
    st = _sid_index(u)[q.support[0]]
    assert st["text"] == f"{name}'s job is {q.golds[0]}."


def test_qc2_spouse_job_chain_is_derivable_from_support():
    u = build_universe(0, 30)
    q = next(q for q in u.questions.values() if q.template == "qc2_spouse_job")
    idx = _sid_index(u)
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
    idx = _sid_index(u)
    assert all(("father of" in idx[s]["text"]) or ("mother of" in idx[s]["text"])
               for s in q.support)


def test_qc5_unanswerable_shape():
    u = build_universe(0, 30)
    qc5 = [q for q in u.questions.values() if q.category == "QC5"]
    assert qc5
    for q in qc5:
        assert q.unanswerable and q.golds == ["unknown"] and q.support == []


def test_support_sets_reference_real_origins():
    u = build_universe(0, 60)
    idx = _sid_index(u)
    for q in u.questions.values():
        for s in q.support:
            assert s in idx and idx[s]["origin"] == s


def test_relations_render_one_statement_per_endpoint():
    u = mini_universe()
    texts = _stmt_texts(u)
    marriage = next(t for t in texts if " is married to " in t)
    a, b = marriage.removesuffix(".").split(" is married to ")
    assert f"{b} is married to {a}." in texts     # mirrored, distinct origin


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
    assert u.meta["build_tokens"] == {"in": 0, "out": 0}   # no API by default
    stats = json.loads(capsys.readouterr().out)
    assert stats["docs"] == 12
    assert "eval" in stats["splits"]
