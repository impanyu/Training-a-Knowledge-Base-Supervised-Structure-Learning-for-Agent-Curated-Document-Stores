"""Score every PhantomWiki authored index semantically (mirror of
index_precision2/index_quality_table): resolve the key's true member set
from the universe statements, then precision = links pointing at a document
that names a member, recall = members named by some linked document."""
import json, re, collections

u = json.loads(open("data/pw1/universe.json").read())
NAME = re.compile(r"\b[A-Z][a-z]+(?: [A-Z][a-z]+)+\b")

mother, father = {}, {}
children = collections.defaultdict(set)
spouse = collections.defaultdict(set)
sib_stmt = collections.defaultdict(set)
friends = collections.defaultdict(set)
gender, hobby, occ = {}, collections.defaultdict(set), collections.defaultdict(set)

for n in u["nodes"]:
    t = n["text"].rstrip(".")
    names = NAME.findall(t)
    low = t.lower()
    def rel(prefix):
        return low.startswith(prefix) and len(names) >= 2
    if rel("the mother of "):  mother[names[0]] = names[1]; children[names[1]].add(names[0])
    elif rel("the father of "): father[names[0]] = names[1]; children[names[1]].add(names[0])
    elif rel("the husband of ") or rel("the wife of "):
        spouse[names[0]].add(names[1]); spouse[names[1]].add(names[0])
    elif low.startswith(("the son of ", "the sons of ", "the daughter of ",
                         "the daughters of ")) and len(names) >= 2:
        for c in names[1:]: children[names[0]].add(c)
    elif low.startswith(("the brother of ", "the brothers of ", "the sister of ",
                         "the sisters of ")) and len(names) >= 2:
        for s in names[1:]:
            sib_stmt[names[0]].add(s); sib_stmt[s].add(names[0])
    elif low.startswith(("the friend of ", "the friends of ")) and len(names) >= 2:
        for f in names[1:]:
            friends[names[0]].add(f); friends[f].add(names[0])
    elif low.startswith("the gender of ") and names:
        gender[names[0]] = "female" if "female" in low else "male"
    elif low.startswith("the hobby of ") and names:
        hobby[t.split(" is ", 1)[1].strip()].add(names[0])
    elif low.startswith("the occupation of ") and names:
        occ[t.split(" is ", 1)[1].strip()].add(names[0])

def parents(x):
    return {p for p in (mother.get(x), father.get(x)) if p}
def sibs(x):
    s = set(sib_stmt.get(x, set()))
    for p in parents(x): s |= children[p]
    return s - {x}
def bygender(ppl, g):
    return {p for p in ppl if gender.get(p) == g}
def grandparents(x):
    return {gp for p in parents(x) for gp in parents(p)}
def grandchildren(x):
    return {gc for c in children[x] for gc in children[c]}
def cousins(x):
    return {c for p in parents(x) for ps in sibs(p) for c in children[ps]}
def auntsuncles(x):
    return {ps for p in parents(x) for ps in sibs(p)}
def niecesnephews(x):
    return {c for s in sibs(x) for c in children[s]}

def members(text):
    t = text.rstrip("."); low = t.lower()
    names = NAME.findall(t)
    who = names[0] if names else None
    for value, ppl in hobby.items():
        if f"hobby is {value.lower()}" in low: return ppl, "attribute"
    for value, ppl in occ.items():
        if f"occupation is {value.lower()}" in low: return ppl, "attribute"
    if who:
        M = {
         "great-grandmother": (lambda: bygender({p for g in grandparents(who) for p in parents(g)}, "female"), "rel2+"),
         "great-grandfather": (lambda: bygender({p for g in grandparents(who) for p in parents(g)}, "male"), "rel2+"),
         "great-grandparent": (lambda: {p for g in grandparents(who) for p in parents(g)}, "rel2+"),
         "granddaughter": (lambda: bygender(grandchildren(who), "female"), "rel2+"),
         "grandson":      (lambda: bygender(grandchildren(who), "male"), "rel2+"),
         "grandchild":    (lambda: grandchildren(who), "rel2+"),
         "grandmother":   (lambda: bygender(grandparents(who), "female"), "rel2+"),
         "grandfather":   (lambda: bygender(grandparents(who), "male"), "rel2+"),
         "grandparent":   (lambda: grandparents(who), "rel2+"),
         "female cousin": (lambda: bygender(cousins(who), "female"), "rel2+"),
         "male cousin":   (lambda: bygender(cousins(who), "male"), "rel2+"),
         "cousin":        (lambda: cousins(who), "rel2+"),
         "niece":         (lambda: bygender(niecesnephews(who), "female"), "rel2+"),
         "nephew":        (lambda: bygender(niecesnephews(who), "male"), "rel2+"),
         "aunt":          (lambda: bygender(auntsuncles(who), "female"), "rel2+"),
         "uncle":         (lambda: bygender(auntsuncles(who), "male"), "rel2+"),
         "sister":        (lambda: bygender(sibs(who), "female"), "rel1"),
         "brother":       (lambda: bygender(sibs(who), "male"), "rel1"),
         "sibling":       (lambda: sibs(who), "rel1"),
         "daughter":      (lambda: bygender(children[who], "female"), "rel1"),
         "son":           (lambda: bygender(children[who], "male"), "rel1"),
         "child":         (lambda: set(children[who]), "rel1"),
         "friend":        (lambda: set(friends[who]), "rel1"),
         "mother":        (lambda: {mother[who]} if who in mother else set(), "rel1"),
         "father":        (lambda: {father[who]} if who in father else set(), "rel1"),
         "parent":        (lambda: parents(who), "rel1"),
         "husband":       (lambda: bygender(spouse[who], "male"), "rel1"),
         "wife":          (lambda: bygender(spouse[who], "female"), "rel1"),
         "spouse":        (lambda: set(spouse[who]), "rel1"),
        }
        for kw, (fn, fam) in M.items():
            if re.search(rf"\b{kw}s?\b", low): return fn(), fam
        return {who}, "hub"
    return None, None

def main():
    d = json.loads(open("runs/pw1_main/kb_epoch_2.json").read())
    nodes = d["store"]["nodes"]
    byid = {n["id"]: n for n in nodes}
    auth = [n for n in nodes if n.get("flag") == "authored"]

    fam_stats = collections.defaultdict(lambda: [0,0,0,0,0, None])  # n, links, correct, mfound, mtotal, example
    unresolved, empty = [], 0
    for n in auth:
        if not n.get("links"): empty += 1; continue
        mem, fam = members(n["text"])
        if mem is None or not mem:
            unresolved.append(n["text"]); continue
        tgt_names = [set(NAME.findall(byid[t]["text"])) if t in byid else set()
                     for t in n["links"]]
        correct = sum(1 for s in tgt_names if s & mem)
        found = {m for m in mem if any(m in s for s in tgt_names)}
        st = fam_stats[fam]
        st[0]+=1; st[1]+=len(n["links"]); st[2]+=correct
        st[3]+=len(found); st[4]+=len(mem)
        if st[5] is None and correct == len(n["links"]) and len(n["links"])>1:
            st[5] = n["text"]

    print(f"authored {len(auth)}, empty {empty}, unresolved {len(unresolved)}")
    print(f"{'family':10s} {'n':>4s} {'deg':>5s} {'prec':>6s} {'recall':>7s}  example")
    T=[0,0,0,0,0]
    for fam in ["attribute","hub","rel1","rel2+"]:
        st = fam_stats[fam]
        if not st[0]: continue
        for i in range(5): T[i]+=st[i]
        print(f"{fam:10s} {st[0]:4d} {st[1]/st[0]:5.1f} {st[2]/st[1]:6.0%} {st[3]/st[4]:7.0%}  {st[5]}")
    print(f"{'ALL':10s} {T[0]:4d} {T[1]/T[0]:5.1f} {T[2]/T[1]:6.0%} {T[3]/T[4]:7.0%}")
    print("\nunresolved samples:", unresolved[:6])


if __name__ == "__main__":
    main()
