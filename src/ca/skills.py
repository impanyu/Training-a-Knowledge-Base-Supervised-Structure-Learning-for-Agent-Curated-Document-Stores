"""Role handbooks with action-trajectory demos, assembled per level so a
demo never shows an action the agent lacks at that level.

v2: every demo is built around the task tree -- claim a task, decompose it
level by level, then package ONE JSON map covering all of its leaves.
"""
from ca.config import LevelConfig

_SOLO_ANSWER = """
### Demo: taking a task from the WORLD and answering it yourself
1. list_tasks -> "[t0007] «identify the composers behind three operas» (3 questions, reward 6000)"
2. claim_task(task="t0007")                           # id or the sentence itself
3. decompose(node="t0007")                            # COSTS TOKENS
   -> "  [t0012] «date the two premieres» (2 questions, reward 4000)"
   -> "  [q0033] Who composed Salome?"
4. decompose(node="t0012")  -> "[q0031] ...", "[q0032] ..."   # keep going until
   every leaf question is visible: you cannot deliver a task you have not opened
5. retrieve(query="composer of Salome")               # COSTS TOKENS
6. work_on(task_id="q0033", thought="Salome -> Richard Strauss; 2 leaves left")
7. deliver_work(target_id="t0007",
       content='{"q0031": "1911", "q0032": "1905", "q0033": "Richard Strauss"}')
   -> ONE graded attempt: EVERY leaf q-id present, each value the SHORT answer
      only (a name / date / phrase), never a sentence. Paid = sum(price x F1).
      A non-JSON or incomplete map is refused for free - the attempt survives.
Estimate cost vs reward BEFORE claiming; skip tasks you cannot answer profitably.
CUT LOSSES: if a leaf is not converging after 2-3 retrieves, put your best guess
in the package (partial F1 still pays) and deliver - never sink unlimited tokens
into one leaf, and never let the whole package expire over one hard question.
If your claim on a task EXPIRED, that is strong evidence it is too big for you:
do NOT re-claim it, pick a smaller one."""

_SOLO_ANSWER_NO_RETRIEVE = """
### Demo: taking a task from the WORLD and answering it yourself
1. list_tasks -> "[t0007] «identify the composers behind three operas» (3 questions, reward 6000)"
2. claim_task(task="t0007")
3. decompose(node="t0007") ... decompose each child until every leaf is visible
4. work_on(task_id="q0033", thought="recall what I know about Salome ...")
5. deliver_work(target_id="t0007",
       content='{"q0031": "1911", "q0032": "1905", "q0033": "Richard Strauss"}')
   -> one attempt, every leaf q-id required, paid = sum(price x F1)"""

_CONTRACTOR = """
### Demo: earning tokens as a contractor
- Unread message: "[contract offer c0007] task: date the two premieres | price: 300
  [bound to t0012: the deliverable must be a JSON map with full leaf coverage of t0012 (2 questions)]"
- accept_contract(contract_id="c0007")     # 300 locked in escrow from the payer{counter_line}
- decompose(node="t0012")                  # see exactly which questions you owe
- do the work{how}, then:
- deliver_work(target_id="c0007", content='{{"q0031": "1911", "q0032": "1905"}}')
  -> escrow released to you. A BOUND contract settles only when every leaf q-id
     of that node is present; answers are NOT machine-graded here, but cheaters
     lose future business. A free-text contract takes any text instead."""

_BUY_INFO = """
### Demo: buying external information (only the interface can search the corpus)
- propose_contract(to="interface", task="look up: who composed Salome"{price_arg})
- interface accepts+delivers -> the passages arrive in your chat."""

_HIRE_PEER = """
### Demo: hiring another agent (you can be the payer, not only the worker)
- decompose your claimed task, then hand a whole CHILD SUBTREE to a peer:
- propose_contract(to="agent_5", task="date the two premieres"{price_arg})
  -> naming a subtask's sentence BINDS the contract to that node, so agent_5
     must return a JSON map covering all of its leaves - not vague prose
- agent_5 accepts -> your tokens move into escrow; keep taking your own turns
- when agent_5 delivers, the JSON lands in your chat and escrow is released
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
watch the claim expiry countdown - an unpackaged task pays nothing.
YOUR TURN IS THE SCARCEST RESOURCE IN THE SYSTEM: never spend it on greetings
or per-worker small talk. Priority every turn:
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


def role_skill(level: LevelConfig, agent_id: str) -> str:
    is_iface = agent_id == "interface"
    can_retrieve = level.retrieve_access == "all" or is_iface
    can_world = level.world_access == "all" or is_iface
    solo = level.n_agents == 1
    blocks: list[str] = []
    if is_iface:
        blocks.append(_IFACE_PIPELINE)
        if level.central_pricing:
            blocks.append(_IFACE_PRICING)
        if level.central_credit:
            blocks.append(_IFACE_LENDER)
    else:
        if can_world:
            blocks.append(_SOLO_ANSWER if can_retrieve else _SOLO_ANSWER_NO_RETRIEVE)
        if not solo:
            counter = ("\n- price too low? counter_offer(contract_id=\"c0007\", price=450)"
                       if not level.central_pricing else "")
            how = (" (retrieve / work_on)" if can_retrieve
                   else " (work_on with what you know, or buy info)")
            blocks.append(_CONTRACTOR.format(counter_line=counter, how=how))
            if not can_retrieve:
                price_arg = "" if level.central_pricing else ", price=80"
                blocks.append(_BUY_INFO.format(price_arg=price_arg))
            elif not level.star_comms:
                # workers who CAN retrieve never see _BUY_INFO, so without this
                # they are only ever shown how to be hired, never how to hire
                price_arg = "" if level.central_pricing else ", price=120"
                blocks.append(_HIRE_PEER.format(price_arg=price_arg))
            lender = "interface" if (level.central_credit or level.star_comms) else "agent_5"
            blocks.append(_BORROW.format(lender=lender))
    if not blocks:
        return ""
    return "\n\n## ROLE HANDBOOK (worked examples)\n" + "\n".join(blocks)
