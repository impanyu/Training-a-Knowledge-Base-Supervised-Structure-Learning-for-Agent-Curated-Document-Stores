"""Role handbooks with action-trajectory demos, assembled per configuration so
a demo never shows an action the agent lacks there.

v4: a task is ONE question with a quota. Every demo is built around the flat
pipeline -- list, claim (which hands back any stored answer), solve, deliver one
short answer string.

Configurations are single-factor, so a demo may only assume the ONE mechanism
that config centralizes. In particular the hub is not privileged in general --
at C3/C4/C5 it holds exactly one power and is an ordinary market participant in
every other respect.
"""
from ca.config import LevelConfig

_SOLO_ANSWER = """
### Demo: the solving pipeline (list -> claim -> solve -> deliver one answer)
1. list_questions
   -> "[q0042] Who composed Salome? (2hop, reward 63000, 3 left)"
   "3 left" is the QUOTA still on offer: three more agents (or you, twice) can
   still be paid for this question. Pick by reward vs what it will cost you.
2. claim_question(qid="q0042")
   -> "claimed [q0042] Who composed Salome? (2hop, reward 63000, 2 left)"
   If an answer for it is already in your memory it is attached right here:
   -> memory: stored answer [q0042] Who composed Salome? -> "Richard Strauss"
      (F1 1.00) - GOOD (F1 1.00 of 1.00): you can deliver it as-is
   A GOOD stored answer is nearly free money: deliver it and move on.
   A LOW QUALITY one (F1 below 0.50) is a warning: re-solve before delivering.
3. retrieve(query="composer of Salome")               # COSTS TOKENS
4. work_on(question_id="q0042", thought="Salome -> Richard Strauss")
   # persists reasoning across turns - your recent-actions window is short
5. deliver_work(target_id="q0042", content="Richard Strauss")
   -> ONE graded attempt per claim. `content` is the SHORT ANSWER ONLY (a name /
      date / phrase) - never a sentence, never a structured object. Paid
      round(price x F1).
CUT LOSSES: if a question is not converging after 2-3 retrieves, deliver your
best guess (partial F1 still pays) rather than sinking unlimited tokens into it.
You get at most TWO claims on any one question, and an EXPIRED claim burns one
of them, so never sit on a claim you are not going to deliver."""

_CONTRACTOR = """
### Demo: earning tokens as a contractor
- Unread message: "[contract offer c0007] task: q0031 | price: 300
  [bound to q0031: the deliverable is the short answer to "In what year did
  Der Rosenkavalier premiere?"]"
- accept_contract(contract_id="c0007")     # 300 locked in escrow from the payer{counter_line}
- do the work (retrieve / work_on), then:
- deliver_work(target_id="c0007", content="1911")
  -> escrow released to you, and the answer lands in the payer's chat AND in
     both of your memories. A bound deliverable is NOT machine-graded, but
     cheaters lose future business. A free-text contract takes any text."""

_HIRE_PEER = """
### Demo: hiring another agent (you can be the payer, not only the worker)
- Claim a question, then hand the actual solving to a peer:
- propose_contract(to="{peer}", task="q0031"{price_arg})
  -> naming a QUESTION ID as the task BINDS the contract to it, so {peer} owes
     you that question's short answer - not vague prose
- {peer} accepts -> your tokens move into escrow; keep taking your own turns
- when {peer} delivers, the answer lands in your chat and in your memory, and
  you deliver it to the WORLD under your own claim
Delegating costs tokens but frees your turns: offer less than the question's
reward, and check the deliverable before you submit it."""

_IFACE_PIPELINE = """
### Demo: your production pipeline
1. list_questions -> pick questions whose reward exceeds expected cost
2. claim_question(qid="q0042")   # any stored answer is attached to the result
3. EITHER solve it yourself (retrieve / work_on),
   OR subcontract it BY ITS QUESTION ID:
   propose_contract(to="agent_3", task="q0042", price=300)
   -> the contract is bound to q0042, so agent_3 owes you that short answer,
      which lands in your chat
4. deliver_work(target_id="q0042", content="Richard Strauss")
   -> WORLD pays round(price x F1); ONE graded attempt per claim
Your profit = WORLD rewards - subcontract payments - your own token burn.
Parallelize: keep several agents working on different questions at once, and
watch the claim expiry countdown - an undelivered claim pays nothing."""

# Only true when the hub holds the demand monopoly (C1): there it is the
# system's single income channel, so its turns are the scarcest resource. At
# C3/C4/C5 every agent earns from the WORLD directly, so this must NOT appear.
_IFACE_BOTTLENECK = """
YOUR TURN IS THE SCARCEST RESOURCE IN THE SYSTEM: you are the only agent who
can claim questions and be paid by the WORLD, so never spend a turn on greetings
or per-worker small talk. Priority every turn:
(1) deliver answers you already hold (the only income),
(2) respond to pending contracts and collect deliverables,
(3) claim new questions and subcontract them,
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

_MEMORY = """
### Demo: your memory works for you, unprompted
Every answer you deliver is written to long-term memory automatically, and
claiming that question again hands the answer straight back to you. So a
question with quota left that you have ALREADY answered well is the cheapest
money on the board: claim it, read the stored answer, deliver it.
- memory_write(content="4-hop opera questions are rarely worth the retrieves")
  saves a free-text note; memory_search(query="opera questions") finds notes
  AND past answers by meaning, which is how you spot a question you can answer
  from what you already know before you spend a single retrieve."""

_MEMORY_SHARED = """
Memory is SHARED at this configuration: every agent writes into the same store
and reads the same store. Answers your peers deliver come back to you on
claim_question exactly as your own do, and your notes are visible to them.
Claim first, look at what is already known, and only then spend tokens."""

_COLLECTIVE = """
### Collective mode
All agents share ONE goal: total system balance. Contract prices only
redistribute - never haggle for margin; pay whatever coordinates work fastest.
Never duplicate a question another agent is already solving (check chat / ask).
Deliver everything you can - income is the only way the system grows."""


def role_skill(level: LevelConfig, agent_id: str) -> str:
    is_iface = agent_id == "hub"
    can_world = level.world_access == "all" or is_iface
    solo = level.n_agents == 1
    blocks: list[str] = []
    if is_iface:
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
    # memory exists at every configuration and for every role
    blocks.append(_MEMORY + (_MEMORY_SHARED if level.shared_memory else ""))
    if level.collective_goal:
        blocks.append(_COLLECTIVE)
    if not blocks:
        return ""
    return "\n\n## ROLE HANDBOOK (worked examples)\n" + "\n".join(blocks)
