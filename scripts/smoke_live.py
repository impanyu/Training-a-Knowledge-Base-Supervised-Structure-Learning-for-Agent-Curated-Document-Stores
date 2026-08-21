"""Five real API calls that check the things stubs cannot.

The unit tests all substitute a stub for the model, so an entire class of
failure is invisible to them: a call that returns an empty string, a
reasoning model that spends its whole completion budget thinking, a guard
that is wired up and inert. Every such bug in this project was found hours
into a run, from the data, when one live call would have shown it.

Run this after ANY change to the action set, a model parameter, or a
completion budget - before starting anything long.

    python3 scripts/smoke_live.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from kb.actions import dispatch, tools_for           # noqa: E402
from kb.policy import make_policy                    # noqa: E402
from kb.store import LLMDuplicateJudge               # noqa: E402
from kb.test import load_kb                          # noqa: E402

UNIVERSE = "data/v10L/universe.json"
fails = []


def check(name, ok, detail=""):
    print(f"  {'PASS' if ok else 'FAIL'}  {name}{'  ' + detail if detail else ''}")
    if not ok:
        fails.append(name)


print("live smoke test\n")

# 1. the policy emits a tool call rather than burning its budget on reasoning
pol = make_policy("gpt-5-mini")
d = pol.decide("Answer using the tools.", "What is the job of Felix Oakhurst?",
               tools_for("test"))
check("policy returns a tool call", d.name != "__noop__",
      f"got {d.name}, {d.out_tokens} out tokens")

# 2. the duplicate judge actually answers, both ways
j = LLMDuplicateJudge()
same = j("Sylvie Ravenscroft's job is tailor.",
         "The job of Sylvie Ravenscroft is tailor.")
diff = j("Sylvie Ravenscroft's job is tailor.",
         "Osric Kestrel's job is glassblower.")
check("judge says yes to the same fact", same is True)
check("judge says no to different facts", diff is False,
      f"{j.tokens_out} out tokens for two calls")

# 3. paging walks a set instead of repeating page one
store, _ = load_kb(UNIVERSE, UNIVERSE)
seen, pages = set(), 0
for page in range(1, 7):
    out = dispatch(store, "search", {"query": "lives in the city of Ambleford",
                                     "page": page})
    ids = {ln.split(":")[0][2:] for ln in out.splitlines() if ln.startswith("- ")}
    if not ids:
        break
    pages += 1
    check(f"page {page} adds new documents", bool(ids - seen),
          f"{len(ids - seen)} new")
    seen |= ids
check("paging reached a real set", len(seen) >= 25, f"{len(seen)} documents over {pages} pages")

print()
if fails:
    raise SystemExit(f"{len(fails)} live check(s) failed: {', '.join(fails)}")
print("all live checks passed")
