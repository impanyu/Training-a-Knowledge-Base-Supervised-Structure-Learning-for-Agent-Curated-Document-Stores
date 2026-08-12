"""Role handbooks with action-trajectory demos, assembled per configuration so
a demo never shows an action the agent lacks there.

v5: a task is ONE question, and the corpus is memory. Every demo is built
around the same pipeline: list, claim (which hands back the stored answer if
one exists), memory_search the born-in corpus, deliver ONE short answer.

Configurations are single-factor, so a demo may only assume the ONE mechanism
that config centralizes. In particular the hub is not privileged in general --
at C3/C4/C5 it holds exactly one power and is an ordinary market participant in
every other respect.
"""
from ca.config import LevelConfig

_SOLO_ANSWER = """
### Demo: the question pipeline (list -> claim -> search memory -> deliver)
1. list_questions
   -> "[q0107] In what year did Der Rosenkavalier premiere? (2hop, reward 18000)"
      ... one line per open question
2. claim_question(qid="q0107")
   -> "claimed [q0107] In what year did Der Rosenkavalier premiere? (2hop, reward 18000)"
      "deliver ONE short answer: deliver_work(target_id="q0107", content="<answer>")"
   If your memory already held an answer you would see it here with its F1:
      "memory: stored answer: [q0107] ... -> "1911" (F1 1.00) GOOD: deliver it as-is"
   A GOOD stored answer is free money. A LOW QUALITY one is a warning: re-solve.
3. memory_search(query="Der Rosenkavalier premiere")     # COSTS TOKENS
   -> "- [Der Rosenkavalier] Der Rosenkavalier ... premiered in 1911 at the ..."
4. memory_write(content="q0107: Der Rosenkavalier premiered 1911, Dresden")
   # persists reasoning across turns - your recent-actions window is short
5. deliver_work(target_id="q0107", content="1911")
   -> `content` is the SHORT ANSWER ONLY (a name / date / phrase), a bare
      string - never a sentence. This is your ONE graded attempt on the
      claim; it pays round(price x F1).
DO THE ARITHMETIC BEFORE YOU CLAIM: one answer takes about 3 turns (a search,
a note, a check). Claims never expire, but every turn burns tokens, so claim
only what you can answer profitably.
CUT LOSSES: if a question is not converging after 2-3 searches, deliver your
best guess (partial F1 still pays) rather than sinking unlimited tokens in."""

_CONTRACTOR = """
### Demo: earning tokens as a contractor
- Unread message: "[contract offer c0007] task: q0031 | price: 300
  [bound to q0031: the deliverable is the short answer to "In what year did
  Der Rosenkavalier premiere?"]"
- accept_contract(contract_id="c0007")     # 300 locked in escrow from the payer{counter_line}
- do the work (memory_search, reason), then:
- deliver_work(target_id="c0007", content="1911")
  -> escrow released to you, and the answer lands in the payer's chat AND in
     both of your memories. A bound deliverable is NOT machine-graded, but
     cheaters lose future business. A free-text contract takes any text.
Selling answers is steady income and it costs you nothing to look:
agents holding claims they cannot finish are steady buyers."""

_HIRE_PEER = """
### Demo: buying an answer instead of solving it (per-question subcontracts)
You hold [q0107] but your turns are worth more elsewhere. Any answer you buy
for less than the question pays is profit:
- propose_contract(to="{peer}", task="q0107"{price_arg})
  -> naming a QUESTION ID as the task BINDS the contract, so the contractor
     owes you that question's short answer - not vague prose
- they accept -> your tokens move into escrow; you keep taking your own turns
- the deliverable lands in your chat AND in your memory, so claiming (or
  holding) q0107 now hands you the answer to deliver
- deliver_work(target_id="q0107", content="1911") and pocket the difference
The arithmetic: pay a peer a fraction of the reward and keep the rest without
spending your own solving turns. Offer less than the question's own price,
and check what you are sent."""

_IFACE_PIPELINE = """
### Demo: your production pipeline
1. list_questions -> pick questions whose reward exceeds expected cost
2. claim_question(qid="q0107")   # hands back your stored answer, if any
3. EITHER solve it yourself (memory_search, then reason),
   OR subcontract it BY ITS QUESTION ID:
   propose_contract(to="agent_3", task="q0107", price=300)
   -> bound to q0107, so agent_3 owes you that short answer, which lands in
      your chat and in your memory
4. deliver_work(target_id="q0107", content="1911")
   -> ONE short answer, ONE graded attempt; the WORLD pays round(price x F1).
Your profit = WORLD rewards - subcontract payments - your own token burn.
Hold several claims and keep several agents answering different questions at
once; buy any answer for less than its question is worth to you."""

# Only true when the hub holds the demand monopoly (C1): there it is the
# system's single income channel, so its turns are the scarcest resource. At
# C3/C4/C5 every agent earns from the WORLD directly, so this must NOT appear.
_IFACE_BOTTLENECK = """
YOUR TURN IS THE SCARCEST RESOURCE IN THE SYSTEM: you are the only agent who
can claim questions and be paid by the WORLD, so never spend a turn on
greetings or per-worker small talk. Priority every turn:
(1) deliver answers you already hold (the only income),
(2) respond to pending contracts and collect deliverables,
(3) subcontract the questions still open on your claims,
(4) claim new questions only when the current ones are nearly done,
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
### Demo: your memory was born knowing the corpus
Your long-term memory starts pre-loaded with the WORLD's whole knowledge
corpus (~12k encyclopedia paragraphs). memory_search is the ONE way to look
anything up, and it searches everything at once - corpus paragraphs, your own
notes, and every answer you have delivered:
- memory_search(query="Lion Air hub airport")
  -> "- [Juanda International Airport] Juanda International Airport ... serves
      Surabaya ..."          (a corpus paragraph)
     "- Lion Air's hub airport = Juanda International, Surabaya"  (your note)
Every answer you deliver is written back automatically, and claiming a
question hands its stored answer straight back with its F1, so an answer you
paid for once keeps paying.
ALSO WRITE DOWN WHAT YOU LEARN ON THE WAY, not just final answers:
- memory_write(content="Lion Air's hub airport = Juanda International, Surabaya")
- memory_write(content="Der Rosenkavalier premiered 1911, Dresden Hofoper")
Multi-hop questions need TWO facts chained: search for the first, note the
bridge, search for the second. These intermediate findings come back by
MEANING on your next search, so a fact you noted while answering one question
answers the next one for free."""

_MEMORY_SHARED = """
Memory is SHARED at this configuration: every agent writes into the same store
and reads the same store. Answers your peers deliver come back to you on
claim_question exactly as your own do, and the intermediate findings you write
down become a COLLECTIVE asset - your note about Juanda International is in
your peers' next search too. Claim first, look at what is already known, and
only then spend tokens."""

_COLLECTIVE = """
### Collective mode
All agents share ONE goal: total system balance. Contract prices only
redistribute - never haggle for margin; pay whatever coordinates work fastest.
Never duplicate a question another agent is already solving (check chat / ask).
Buy answers you cannot produce cheaply, and write your intermediate findings
down - income is the only way the system grows."""


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
