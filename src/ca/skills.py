"""Role handbooks with action-trajectory demos, assembled per configuration so
a demo never shows an action the agent lacks there.

v4.1: what the WORLD posts is a JOB -- 6-10 flat questions claimed and
delivered as a unit. Every demo is built around that pipeline: list, claim
(which marks the members you already know), solve or subcontract the rest,
assemble ONE map, deliver. The arithmetic of "this does not fit in my claim
window" is spelled out, because delegation is the point.

Configurations are single-factor, so a demo may only assume the ONE mechanism
that config centralizes. In particular the hub is not privileged in general --
at C3/C4/C5 it holds exactly one power and is an ordinary market participant in
every other respect.
"""
from ca.config import LevelConfig

_SOLO_ANSWER = """
### Demo: the job pipeline (list -> claim -> solve -> deliver ONE map)
1. list_jobs
   -> "[j0007] 8 questions, topic k07, reward 512000"
   A job is a BUNDLE of 8 related questions. Its reward is the sum of their
   prices; you are paid the sum of round(price x F1) over all of them.
2. claim_job(jid="j0007")
   -> "claimed [j0007] 8 questions, topic k07, reward 512000"
      "memory: stored answer for 2 of these 8 questions (marked ✓ below)"
      "  ✓ q0042 Who composed Salome? -> "Richard Strauss" (F1 1.00) GOOD: deliver as-is"
      "  ✓ q0055 Where is Juanda? -> "Sidoarjo" (F1 0.30) LOW QUALITY: re-solve it"
      "  — q0107 In what year did Der Rosenkavalier premiere? (unanswered)"
      ... every question of the job, every time
   A GOOD ✓ row is free money: that answer is already yours. A LOW QUALITY one
   is a warning. Spend tokens on those and on the "—" rows.
3. retrieve(query="Der Rosenkavalier premiere year")     # COSTS TOKENS
4. work_on(question_id="q0107", thought="Der Rosenkavalier -> 1911")
   # persists reasoning across turns - your recent-actions window is short
5. deliver_work(target_id="j0007",
                content='{"q0042": "Richard Strauss", "q0107": "1911", ...}')
   -> ALL-OR-NOTHING. The map must name EVERY question of the job and nothing
      else. An incomplete or malformed map is rejected and YOU KEEP YOUR CLAIM,
      so a typo costs only the turn. A complete map is your ONE graded attempt.
      Each value is the SHORT ANSWER ONLY (a name / date / phrase).
DO THE ARITHMETIC BEFORE YOU CLAIM: one answer takes about 3 turns (a retrieve,
a work_on, a check), so a job of 8 is ~24 turns of your own - and a claim
expires after 20 rounds. Alone, you will not finish it.
CUT LOSSES on any single question: if it is not converging after 2-3 retrieves,
put your best guess in the map (partial F1 still pays) rather than sinking
unlimited tokens into it - and NEVER let the claim expire over one question."""

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
     cheaters lose future business. A free-text contract takes any text.
Selling single questions is steady income and it costs you nothing to look:
job holders NEED buyers, because a job of 8 does not fit in one agent's turns."""

_HIRE_PEER = """
### Demo: splitting a claimed job across peers (this is how jobs get finished)
You hold [j0007]: 8 questions, 20 rounds before the claim expires. Solving all
8 yourself is ~24 turns - it does not fit. Delegating does:
- propose_contract(to="{peer}", task="q0107"{price_arg})
  propose_contract(to="agent_2", task="q0111"{price_arg})   # one contract PER QUESTION
  -> naming a QUESTION ID as the task BINDS the contract, so the contractor owes
     you that question's short answer - not vague prose
- they accept -> your tokens move into escrow; you keep taking your own turns
  and solve 2-3 of the questions yourself
- each deliverable lands in your chat AND in your memory, so your active-claims
  block flips that row to "✓ answered"
- when every row is ✓, assemble ONE map and deliver_work(target_id="j0007", ...)
The arithmetic: 5 contracts at ~a tenth of the job reward each still leaves you
most of it, and it is the difference between delivering the job and losing the
claim. Offer less than a question's own price, and check what you are sent."""

_IFACE_PIPELINE = """
### Demo: your production pipeline
1. list_jobs -> pick jobs whose reward exceeds expected cost
2. claim_job(jid="j0007")   # reveals all 8 questions; ✓ = already in memory
3. For each unanswered question EITHER solve it yourself (retrieve / work_on),
   OR subcontract it BY ITS QUESTION ID:
   propose_contract(to="agent_3", task="q0107", price=300)
   -> bound to q0107, so agent_3 owes you that short answer, which lands in
      your chat and in your memory
4. deliver_work(target_id="j0007",
                content='{"q0042": "Richard Strauss", "q0107": "1911", ...}')
   -> the map must cover EVERY question of the job; the WORLD pays the sum of
      round(price x F1). ONE graded attempt per claim.
Your profit = WORLD rewards - subcontract payments - your own token burn.
A job of 8 is ~24 turns of solving but the claim lasts 20 rounds: parallelize.
Keep several agents working on different questions at once, and watch the
expiry countdown - an undelivered claim pays nothing at all."""

# Only true when the hub holds the demand monopoly (C1): there it is the
# system's single income channel, so its turns are the scarcest resource. At
# C3/C4/C5 every agent earns from the WORLD directly, so this must NOT appear.
_IFACE_BOTTLENECK = """
YOUR TURN IS THE SCARCEST RESOURCE IN THE SYSTEM: you are the only agent who
can claim jobs and be paid by the WORLD, so never spend a turn on greetings or
per-worker small talk. Priority every turn:
(1) deliver jobs whose questions are all answered (the only income),
(2) respond to pending contracts and collect deliverables,
(3) subcontract the questions still open on your claimed jobs,
(4) claim a new job only when the current ones are nearly done,
(5) only if nothing above is pending: messaging.
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
claiming a job hands those answers straight back to you, question by question.
Because a question sits in about THREE different jobs, an answer you paid for
once keeps paying: a job whose rows are mostly ✓ is the cheapest money on the
board.
ALSO WRITE DOWN WHAT YOU LEARN ON THE WAY, not just final answers:
- memory_write(content="Lion Air's hub airport = Juanda International, Surabaya")
- memory_write(content="Der Rosenkavalier premiered 1911, Dresden Hofoper")
These intermediate findings are the only decomposition in this world - nobody
hands you sub-questions. memory_search(query="Surabaya airport") retrieves them
BY MEANING, so a fact you noted while working one question answers the next one
for free. Jobs are drawn from one topic, so the questions around you keep
rewarding the same notes."""

_MEMORY_SHARED = """
Memory is SHARED at this configuration: every agent writes into the same store
and reads the same store. Answers your peers deliver come back to you on
claim_job exactly as your own do, and the intermediate findings you write down
become a COLLECTIVE asset - your note about Juanda International is on your
peers' next claim too. Claim first, look at what is already known, and only
then spend tokens."""

_COLLECTIVE = """
### Collective mode
All agents share ONE goal: total system balance. Contract prices only
redistribute - never haggle for margin; pay whatever coordinates work fastest.
Never duplicate a question another agent is already solving (check chat / ask).
Split every claimed job you cannot finish alone, and write your intermediate
findings down - income is the only way the system grows."""


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
