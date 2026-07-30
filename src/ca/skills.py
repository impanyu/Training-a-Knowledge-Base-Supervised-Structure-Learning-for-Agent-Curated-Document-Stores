"""Role handbooks with action-trajectory demos, assembled per level so a
demo never shows an action the agent lacks at that level."""
from ca.config import LevelConfig

_SOLO_ANSWER = """
### Demo: answering a question yourself
1. list_questions -> "q0012 [3hop, reward 2000]: In which city was the author of X born?"
2. claim_question(qid="q0012")
3. retrieve(query="author of X")                      # COSTS TOKENS
4. work_on(task_id="q0012", thought="author is Y; now need Y's birthplace")
5. retrieve(query="Y birthplace")                     # COSTS TOKENS
6. deliver_work(target_id="q0012", content="Paris")   # graded, paid = 2000 x F1
Estimate cost vs reward BEFORE claiming; skip questions you cannot answer profitably.
CUT LOSSES: if a question is not converging after 2-3 retrieves, deliver your best
guess (partial F1 still pays something) and move to another question - never sink
unlimited tokens into one question. If your claim on a question EXPIRED, that is
strong evidence it is too hard for you: do NOT re-claim it, pick a different one."""

_SOLO_ANSWER_NO_RETRIEVE = """
### Demo: answering a question yourself
1. claim_question(qid="q0012")
2. work_on(task_id="q0012", thought="recall what I know about X ...")
3. deliver_work(target_id="q0012", content="Paris")   # graded, paid = price x F1"""

_CONTRACTOR = """
### Demo: earning tokens as a contractor
- Unread message: "[contract offer c0007] task: find the birthplace of Y | price: 300"
- accept_contract(contract_id="c0007")     # 300 locked in escrow from the payer{counter_line}
- do the work{how}, then:
- deliver_work(target_id="c0007", content="Y was born in Paris (source: ...)")
  -> escrow released to you. Deliver USEFUL content: cheaters lose future business."""

_BUY_INFO = """
### Demo: buying external information (only the interface can search the corpus)
- propose_contract(to="interface", task="look up: birthplace of Y"{price_arg})
- interface accepts+delivers -> the passages arrive in your chat."""

_HIRE_PEER = """
### Demo: hiring another agent (you can be the payer, not only the worker)
- A task is worth more to you than what a peer would charge to do it:
- propose_contract(to="agent_5", task="find the birthplace of Y"{price_arg})
- agent_5 accepts -> your tokens move into escrow; keep taking your own turns
- when agent_5 delivers, the content lands in your chat and escrow is released
Delegating costs tokens but frees your turns: offer less than the work is
worth to you, and check the deliverable before relying on it."""

_IFACE_PIPELINE = """
### Demo: your production pipeline
1. list_questions -> pick questions whose reward exceeds expected cost
2. claim_question(qid="q0012")
3. EITHER answer it yourself (retrieve / work_on / deliver_work),
   OR subcontract: propose_contract(to="agent_3", task="find the birthplace of Y", price=300)
   -> when agent_3 delivers, the answer arrives in your chat
4. deliver_work(target_id="q0012", content="Paris")   # WORLD pays you 2000 x F1
Your profit = WORLD rewards - subcontract payments - your own token burn.
Parallelize: keep several agents working on different questions at once.
YOUR TURN IS THE SCARCEST RESOURCE IN THE SYSTEM: never spend it on greetings
or per-worker small talk. Priority every turn:
(1) deliver finished answers to the WORLD (this is the only income),
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
