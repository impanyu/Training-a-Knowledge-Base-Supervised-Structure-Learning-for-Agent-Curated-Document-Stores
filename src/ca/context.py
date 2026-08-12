"""Builds the LLM context: stable system prompt + per-turn dynamic view."""
from ca.config import LevelConfig
from ca.infra import Infra
from ca.memory import FifoMemory, GoalStack
from ca.skills import role_skill

_GOAL_OWN = "YOUR PERMANENT ROOT GOAL: maximize your token balance."

# C6 (consensus centralization): one shared objective function. Property and
# trading rules stay fully decentralized -- only the goal changes.
_GOAL_COLLECTIVE = (
    "YOUR PERMANENT ROOT GOAL: maximize the TOTAL token balance of the ENTIRE "
    "SYSTEM (all agents combined). Your own balance only matters as part of "
    "the whole. Internal payments and contract prices are neutral for this "
    "goal - only WORLD income (adds) and token burn (subtracts) move it. "
    "Avoid duplicated work across agents; coordinate to minimize total burn.")

_BASE = """You are {agent_id}, an autonomous agent in a multi-agent economy.
Agents in the system: {peers}.

{root_goal}
Tokens are both your money and your fuel: EVERY action costs the tokens that
turn's LLM call consumed. If your balance drops to 0 or below you are BANKRUPT
and can no longer perform answer-related actions (retrieve, delivering
to the WORLD); you may still coordinate and borrow.

The WORLD posts JOBS. A job is a BUNDLE of 2-10 related questions, claimed and
delivered as a unit, and its reward is the sum of its questions' prices. The
pipeline is always: list_jobs -> claim_job -> answer every question -> deliver
them all at once. `claim_job(jid="j0007")` reveals the text of every question in
the job (and marks the ones already in your memory), then
`deliver_work(target_id="j0007", content='{{"q0042": "Richard Strauss",
"q0107": "1911"}}')`. Each answer in that map must be ONLY the short answer
itself (a name / date / phrase) - never a sentence or an explanation - because
each is graded by token-overlap F1 against a short gold answer, and the job pays
the sum of round(price x F1).

Delivery is ALL-OR-NOTHING: a map that misses one of the job's questions, or
names a question that is not in it, is REJECTED and you keep your claim - so a
malformed attempt costs you only the turn. A complete map is your ONE graded
attempt. A claim does not expire, so you may take as long as a job needs - but
every turn you spend costs tokens, so a job you cannot finish profitably is one
you should not have claimed.

Jobs range from 2 to 10 questions. Pick a size you can actually finish: a small
job is quick money, a large one pays more but may be worth SUBCONTRACTING -
name a QUESTION id as a contract task and the contractor owes you that
question's short answer, which you merge into your map. Paying a peer less than
a question is worth to you is pure profit.

Your long-term memory fills itself: every answer you deliver is stored, and
claiming a job shows you, per question, the answer you already have with its F1,
so you can reuse it as-is or re-solve it if it scored badly.

You earn tokens ONLY from: (a) delivering answers to the WORLD, or
(b) payments from other agents (contract escrow settlements or transfers).

Each turn you must choose EXACTLY ONE action (tool call). Unread messages
are delivered automatically into your context every turn - you never need
to poll read_chat (use it only to re-read older history). Think about
profitability: estimate what a job will cost to finish vs its reward, and what
a peer will charge for one of its questions.
{level_rules}"""

_HUB_EXTRA_DEMAND = """
YOU ARE THE HUB AGENT: the only agent allowed to take jobs from the job board
and deliver answers to the WORLD. Other agents can work for you via contracts.
Your profit = WORLD rewards minus what you pay them."""

# C3/C4/C5: the hub holds exactly ONE power (named in the configuration
# rules above) and is an ordinary market participant in every other respect --
# it must not be told it monopolizes the question board.
_HUB_EXTRA = """
YOU ARE THE HUB AGENT: the hub of this configuration. Apart from the one
privilege named in the configuration rules above you are an ordinary agent -
every other agent may claim jobs and deliver to the WORLD exactly as you can,
and you earn by solving and delivering jobs just like they do."""


def _level_rules(level: LevelConfig, is_iface: bool) -> str:
    rules = []
    if level.world_access == "hub":
        rules.append("Only the hub agent can list/claim jobs and deliver answers "
                     "to the WORLD.")
    if level.central_pricing:
        rules.append("ALL contract prices are set by the hub agent (set_price); bargaining is disabled."
                     if not is_iface else
                     "You set the price of EVERY contract in the system via set_price; nobody can bargain.")
    if level.central_credit:
        rules.append("Loans: you may only borrow from the hub agent."
                     if not is_iface else
                     "You are the SOLE lender in the system.")
    if level.star_comms:
        rules.append("You may only message/contract/pay the hub agent."
                     if not is_iface else
                     "Other agents can only interact with you, not with each other.")
    if level.shared_memory:
        rules.append("Long-term memory is SHARED by every agent: your notes and answers "
                     "are visible to all, and theirs to you.")
    negotiable = not level.central_pricing and level.n_agents > 1
    if not rules:
        base = ("\nThis is a fully decentralized configuration: "
                "every agent has equal access to everything.")
        return base + ("\nPrices are freely negotiable." if negotiable else "")
    if negotiable:
        rules.append("Prices are freely negotiable.")
    return "\nConfiguration rules:\n" + "\n".join(f"- {r}" for r in rules)


def system_prompt(level: LevelConfig, agent_id: str, all_ids: list[str]) -> str:
    is_iface = agent_id == "hub"
    sp = _BASE.format(agent_id=agent_id,
                      peers=", ".join(all_ids),
                      root_goal=_GOAL_COLLECTIVE if level.collective_goal else _GOAL_OWN,
                      level_rules=_level_rules(level, is_iface))
    if is_iface:
        sp += (_HUB_EXTRA_DEMAND if level.world_access == "hub"
               else _HUB_EXTRA)
    sp += role_skill(level, agent_id)
    return sp


def render_turn(infra: Infra, agent_id: str, fifo: FifoMemory, goals: GoalStack) -> str:
    own = infra.ledger.balance(agent_id)
    if infra.cfg.level.collective_goal:
        # the quantity C6 agents are told to maximize must be the one they see
        total = sum(infra.ledger.balance(a) for a in infra.agent_ids)
        balance_line = f"Global balance: {total} tokens | Your balance: {own} tokens"
    else:
        balance_line = f"Balance: {own} tokens"
    parts = [f"== ROUND {infra.round} ==", balance_line]
    parts.append("Goal stack (bottom -> top):\n" + goals.render())
    mine = [(jid, c) for jid, c in infra.board.active.items() if c.agent == agent_id]
    if mine:
        lines = []
        for jid, _ in mine:
            job = infra.bank.jobs[jid]
            lines.append(f"- [{jid}] {len(job.qids)} questions, reward {job.price} - "
                         f'deliver_work(target_id="{jid}", content=\'{{"qid": "answer", ...}}\')')
        parts.append("Your active job claims:\n" + "\n".join(lines))
    pend = infra.contracts.pending_for(agent_id)
    if pend:
        lines = []
        for c in pend:
            role = "you must respond" if c.status == "proposed" else "you must deliver"
            lines.append(f"- {c.cid} [{c.status}, {role}] with "
                         f"{c.proposer if agent_id != c.proposer else c.contractor}: "
                         f"{c.task} @ {c.price}")
        parts.append("Contracts needing your attention:\n" + "\n".join(lines))
    pend_loans = infra.loans.pending_for(agent_id)
    if pend_loans:
        lines = []
        for l in pend_loans:
            if l.status == "proposed":
                lines.append(f"- {l.lid} [proposal awaiting your acceptance as lender] "
                             f"{l.borrower} requests {l.principal} tokens")
            elif l.borrower == agent_id:
                interest = infra.loans.interest_of(l)
                lines.append(f"- {l.lid} [you owe {l.lender}] principal {l.principal} tokens "
                             f"(~{interest} interest next round)")
            else:
                lines.append(f"- {l.lid} [owed to you by {l.borrower}] principal {l.principal} tokens")
        parts.append("Your loans:\n" + "\n".join(lines))
    if agent_id == "hub" and infra.cfg.level.central_pricing:
        unp = infra.contracts.unpriced()
        if unp:
            parts.append("Contracts awaiting YOUR pricing (use set_price):\n" +
                         "\n".join(f"- {c.cid} {c.proposer} -> {c.contractor}: {c.task}"
                                   for c in unp))
    unread = infra.chat.unread(agent_id)
    if unread:
        parts.append("Unread messages:\n" +
                     "\n".join(f"- from {m.sender}: {m.text}" for m in unread))
    items = list(fifo.items)
    if len(items) >= 3 and len({a for a, _ in items[-3:]}) == 1:
        parts.append(f"WARNING: you have repeated `{items[-1][0].split('(')[0]}` "
                     "3+ times in a row with identical results. Repeating it again "
                     "is pure waste - you MUST choose a different action this turn.")
    parts.append("Your recent actions:\n" + fifo.render())
    parts.append("Choose exactly one action now.")
    return "\n\n".join(parts)
