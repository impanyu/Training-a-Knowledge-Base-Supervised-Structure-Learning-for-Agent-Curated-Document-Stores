"""Role handbooks with action-trajectory demos, assembled per configuration so
a demo never shows an action the agent lacks there.

v2: every demo is built around the task tree -- claim a task, decompose it
level by level, then package ONE JSON map covering all of its leaves.

v3: configurations are single-factor, so a demo may only assume the ONE
mechanism that config centralizes. In particular the hub is not
privileged in general -- at C3/C4/C5 it holds exactly one power and is an
ordinary market participant in every other respect.
"""
from ca.config import LevelConfig

_SOLO_ANSWER = """
### Demo: the solving pipeline (claim -> decompose -> solve unsolved leaves -> deliver)
1. list_tasks -> "[t0007] «identify the composers behind three operas» (3 questions, reward 6000)"
2. claim_task(task="t0007")                           # id or the sentence itself
3. decompose(node="t0007")                            # COSTS TOKENS
   -> "  [t0012] «date the two premieres» (2 questions, reward 4000)"
   -> "  [q0033] Who composed Salome?"
   -> "known 1/3 answers beneath: {"q0031": "1911" (F1 1.00)}"
   Answers you already hold are attached automatically - those leaves are DONE,
   copy them into your package for free and spend tokens only on the rest.
4. decompose(node="t0012")  -> "[q0031] ...", "[q0032] ..."   # keep going until
   every UNSOLVED leaf question is visible: you cannot answer a question you
   have not opened, and you must not re-solve one you already hold
5. per unsolved leaf: retrieve(query="composer of Salome")    # COSTS TOKENS
6. work_on(task_id="q0033", thought="Salome -> Richard Strauss; q0032 left")
   # persists reasoning across turns - your recent-actions window is short
7. deliver_work(target_id="t0007",
       content='{"q0031": "1911", "q0032": "1905", "q0033": "Richard Strauss"}')
   -> merge STORED answers and FRESH answers into ONE map. ONE graded attempt:
      EVERY leaf q-id present, each value the SHORT answer only (a name / date /
      phrase), never a sentence. Paid = sum(price x F1).
      A non-JSON or incomplete map is refused for free - the attempt survives.
Estimate cost vs reward BEFORE claiming; skip tasks you cannot answer profitably.
CUT LOSSES: if a leaf is not converging after 2-3 retrieves, put your best guess
in the package (partial F1 still pays) and deliver - never sink unlimited tokens
into one leaf, and never let the whole package expire over one hard question.
If your claim on a task EXPIRED, that is strong evidence it is too big for you:
do NOT re-claim it, pick a smaller one."""

_CONTRACTOR = """
### Demo: earning tokens as a contractor
- Unread message: "[contract offer c0007] task: date the two premieres | price: 300
  [bound to t0012: the deliverable must be a JSON map with full leaf coverage of t0012 (2 questions)]"
- accept_contract(contract_id="c0007")     # 300 locked in escrow from the payer{counter_line}
- decompose(node="t0012")                  # see exactly which questions you owe
- do the work (retrieve / work_on), then:
- deliver_work(target_id="c0007", content='{{"q0031": "1911", "q0032": "1905"}}')
  -> escrow released to you. A BOUND contract settles only when every leaf q-id
     of that node is present; answers are NOT machine-graded here, but cheaters
     lose future business. A free-text contract takes any text instead."""

_HIRE_PEER = """
### Demo: hiring another agent (you can be the payer, not only the worker)
- decompose your claimed task, then hand a whole CHILD SUBTREE to a peer:
- propose_contract(to="{peer}", task="date the two premieres"{price_arg})
  -> naming a subtask's sentence BINDS the contract to that node, so {peer}
     must return a JSON map covering all of its leaves - not vague prose
- {peer} accepts -> your tokens move into escrow; keep taking your own turns
- when {peer} delivers, the JSON lands in your chat and escrow is released
Delegating costs tokens but frees your turns: offer less than the subtree is
worth to you, and check the deliverable before merging it into your package."""

_IFACE_PIPELINE = """
### Demo: your production pipeline
1. list_tasks -> pick tasks whose reward exceeds expected cost
2. claim_task(task="t0007")
3. decompose(node="t0007") -> "  [t0012] «date the two premieres» (2 questions,
   reward 4000)" and "  [q0033] Who composed Salome?"
4. EITHER solve leaves yourself (retrieve / work_on),
   OR subcontract a whole child subtree BY ITS SENTENCE:
   propose_contract(to="agent_3", task="date the two premieres", price=300)
   -> the contract is bound to t0012, so agent_3 must return
      {"q0031": ..., "q0032": ...}, which lands in your chat
5. MERGE every child's JSON with your own answers into ONE map covering ALL
   leaves of t0007, then package it:
   deliver_work(target_id="t0007",
       content='{"q0031": "1911", "q0032": "1905", "q0033": "Richard Strauss"}')
   -> WORLD pays sum(price x F1) in one settlement; ONE attempt per task
Your profit = WORLD rewards - subcontract payments - your own token burn.
Parallelize: keep several agents working on different subtrees at once, and
watch the claim expiry countdown - an unpackaged task pays nothing."""

# Only true when the hub holds the demand monopoly (C1): there it is the
# system's single income channel, so its turns are the scarcest resource. At
# C3/C4/C5 every agent earns from the WORLD directly, so this must NOT appear.
_IFACE_BOTTLENECK = """
YOUR TURN IS THE SCARCEST RESOURCE IN THE SYSTEM: you are the only agent who
can take tasks and be paid by the WORLD, so never spend a turn on greetings or
per-worker small talk. Priority every turn:
(1) package and deliver tasks whose leaves are all answered (the only income),
(2) respond to pending contracts and collect deliverables,
(3) claim new tasks, decompose them and subcontract the subtrees,
(4) only if nothing above is pending: messaging.
Workers do not need replies to work - contracts speak for themselves."""

_IFACE_PRICING = """
### Demo: pricing the market (only you can set prices)
- Context shows: "Contracts awaiting YOUR pricing: c0009 agent_2 -> agent_5: ..."
- set_price(contract_id="c0009", price=250)   # then agent_5 may accept or reject
Unpriced contracts stall the economy; price them promptly and consistently."""

_BORROW = """
### Demo: borrowing when you are low on tokens
- Out of tokens? propose_loan(to="{lender}", amount=200)
  -> interest accrues at 1% per round on the outstanding principal
- Once the lender accept_loan()s, the principal lands in your balance
- repay_loan(loan_id="n0001", amount=50) anytime, partial or full, to cut interest"""

_IFACE_LENDER = """
### Demo: lending (you are the SOLE lender)
- Unread message: "[loan request n0001] agent_3 requests 200 tokens at 1%/round interest"
- accept_loan(loan_id="n0001")   # funds the borrower; interest is passive income to you
Every worker who runs low on tokens must borrow from you - keep an eye on loan
requests and fund the ones worth funding."""

_RECALL = """
### Demo: decompose is your memory - reuse what you already solved
Your decompositions and delivered answers are saved automatically; decompose
reads them back. BEFORE working on any task/subtask, decompose it to see what
is already solved - claimed tasks sharing leaves with past work are
nearly-free profit.
- The FIRST decompose(node="t0042") reveals its children, and any answers
  already stored beneath (e.g. from other tasks sharing its leaves) are
  appended: known 1/1 answers beneath: {"q0017": "1905" (F1 1.00)}
- A REPEAT decompose(node="t0042") skips the re-reveal and returns the
  deepest stored knowledge instead - one of:
  -> (t0042 already decomposed) known 1/1 answers beneath: {"q0017": "1905" (F1 1.00)}; not yet expanded: t0043 — decompose deeper or solve the rest
  -> (t0042 already decomposed: t0043, q0017 — no stored answers beneath yet; decompose a child or solve its questions)
  The second form means the remaining work is SOLVING: retrieve and answer
  the leaves, then deliver - do NOT decompose the same node a third time."""

_RECALL_SHARED = """
The knowledge base is SHARED: everyone's discoveries appear in it, and
decompose consults it the same way - decompose a node FIRST, someone may have
already solved your leaves."""

_COLLECTIVE = """
### Collective mode
All agents share ONE goal: total system balance. Contract prices only
redistribute - never haggle for margin; pay whatever coordinates work fastest.
Never duplicate a task another agent is already solving (check chat / ask).
Deliver everything you can - income is the only way the system grows."""


def role_skill(level: LevelConfig, agent_id: str) -> str:
    is_iface = agent_id == "hub"
    can_world = level.world_access == "all" or is_iface
    solo = level.n_agents == 1
    blocks: list[str] = []
    if is_iface:
        # concatenated, not .format()ted: the demo body contains literal JSON braces
        blocks.append(_IFACE_PIPELINE +
                      (_IFACE_BOTTLENECK if level.world_access == "hub" else ""))
        if level.central_pricing:
            blocks.append(_IFACE_PRICING)
        if level.central_credit:
            blocks.append(_IFACE_LENDER)
    else:
        if can_world:
            blocks.append(_SOLO_ANSWER)
        if not solo:
            counter = ("\n- price too low? counter_offer(contract_id=\"c0007\", price=450)"
                       if not level.central_pricing else "")
            blocks.append(_CONTRACTOR.format(counter_line=counter))
            # without this workers only ever see themselves as sellers. Under
            # star comms the only counterparty they can hire is the hub.
            peer = "hub" if level.star_comms else "agent_5"
            price_arg = "" if level.central_pricing else ", price=120"
            blocks.append(_HIRE_PEER.format(peer=peer, price_arg=price_arg))
            lender = "hub" if (level.central_credit or level.star_comms) else "agent_5"
            blocks.append(_BORROW.format(lender=lender))
    # the solution store exists at every configuration and for every role
    blocks.append(_RECALL + (_RECALL_SHARED if level.shared_solution_memory else ""))
    if level.collective_goal:
        blocks.append(_COLLECTIVE)
    if not blocks:
        return ""
    return "\n\n## ROLE HANDBOOK (worked examples)\n" + "\n".join(blocks)
