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
and can no longer perform answer-related actions (retrieve, work_on, decompose,
delivering to the WORLD); you may still coordinate and borrow.

The WORLD posts TASKS, not single questions. A task is a tree: a one-sentence
summary (its name), child subtasks, and question leaves at the bottom. Its
reward is the sum of its leaf prices. Claim a task, use `decompose` to reveal
one level of children at a time (leaf questions only become visible when you
decompose down to them), then deliver the WHOLE task in one package:
`deliver_work(target_id=<task>, content='{{"q0031": "answer", ...}}')` with one
entry for EVERY leaf question - no missing, no extra. Each answer is graded by
F1 and you are paid the sum of price x F1 in a single settlement. You get ONE
graded attempt per task; a malformed or incomplete JSON map is rejected without
spending it. Any task or subtask can be named either by its short id (t0012)
or by its one-sentence summary.

You earn tokens ONLY from: (a) delivering task packages to the WORLD, or
(b) payments from other agents (contract escrow settlements or transfers).

Each turn you must choose EXACTLY ONE action (tool call). Unread messages
are delivered automatically into your context every turn - you never need
to poll read_chat (use it only to re-read older history). Think about
profitability: estimate what a task will cost to solve vs its reward.
You may subcontract a whole subtask subtree to another agent via contracts:
naming the subtask in the contract binds it, and the contractor must return a
JSON map covering that subtask's leaves.
{level_rules}"""

_HUB_EXTRA_DEMAND = """
YOU ARE THE HUB AGENT: the only agent allowed to take tasks from the
task board and deliver packages to the WORLD. Other agents can work for you
via contracts. Your profit = WORLD rewards minus what you pay them."""

# C3/C4/C5: the hub holds exactly ONE power (named in the configuration
# rules above) and is an ordinary market participant in every other respect --
# it must not be told it monopolizes the task board.
_HUB_EXTRA = """
YOU ARE THE HUB AGENT: the hub of this configuration. Apart from the one
privilege named in the configuration rules above you are an ordinary agent -
every other agent may claim tasks and deliver to the WORLD exactly as you can,
and you earn by solving and packaging tasks just like they do."""


def _level_rules(level: LevelConfig, is_iface: bool) -> str:
    rules = []
    if level.world_access == "hub":
        rules.append("Only the hub agent can list/claim tasks and deliver packages to the WORLD.")
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
    pad = infra.scratchpads.get(agent_id) or {}
    pad_lines = []
    for task_id, thoughts in pad.items():
        if not thoughts:
            continue
        pad_lines.append(f"[{task_id}]")
        pad_lines += [f"  - {t}" for t in thoughts[-5:]]
    if pad_lines:
        parts.append("Your scratchpad (latest thoughts per task):\n" + "\n".join(pad_lines))
    # the solution store fills itself silently; agents only reuse what they
    # know is there, so advertise it (once it holds anything at all)
    st = infra.solutions.stats(agent_id)
    if st["answers"] or st["decompositions"]:
        line = f"Solution memory: {st['answers']} answers stored"
        decomposed = infra.solutions.decomposed_ids(agent_id)
        if decomposed:
            shown = ", ".join(decomposed[:12])
            if len(decomposed) > 12:
                shown += f" … +{len(decomposed) - 12} more"
            line += f"; decomposed: {shown}"
        parts.append(line + " (decompose a node to reuse what is known)")
    # tasks this agent already delivered: once closed they vanish from every
    # other view, and agents were observed re-claiming their own finished work
    done = [t for t in infra.board.tasks.values()
            if t.status == "closed" and t.claimed_by == agent_id]
    if done:
        shown = ", ".join(f"{t.nid} (paid {t.payout})" for t in done[-8:])
        if len(done) > 8:
            shown += f" … +{len(done) - 8} more"
        parts.append(f"Tasks you already completed - do NOT claim or deliver "
                     f"these again: {shown}")
    mine = [t for t in infra.board.tasks.values()
            if t.status == "claimed" and t.claimed_by == agent_id]
    if mine:
        ttl = infra.cfg.claim_ttl
        lines = []
        for t in mine:
            lib = infra.library
            leaves = lib.leaves(t.nid)
            left = t.claimed_round + ttl - infra.round
            lines.append(f"- [{t.nid}] «{lib.sentence(t.nid)}» "
                         f"({len(leaves)} questions, reward {lib.price(t.nid)})  "
                         f"[claim EXPIRES in {max(left,0)} round(s) - deliver before then!]")
            # progress hint: leaves this agent has already worked on are named
            # -- undiscovered q-ids stay behind `decompose`. Notes filed under
            # the task's own nid (e.g. before decomposing down to leaves)
            # also count as progress, just not attributable to one leaf.
            noted = [q for q in leaves if pad.get(q)]
            task_notes = pad.get(t.nid) or []
            if noted:
                lines.append(f"    progress: notes on {len(noted)}/{len(leaves)} "
                             f"questions ({', '.join(noted)}); deliver ALL "
                             f"{len(leaves)} in one JSON package")
            elif task_notes:
                lines.append(f"    progress: {len(task_notes)} note(s) filed under "
                             f"{t.nid} itself (not yet broken out per question) - "
                             f"use decompose to reveal its {len(leaves)} question(s)")
            else:
                lines.append(f"    progress: notes on 0/{len(leaves)} questions - "
                             "use decompose to reveal them")
        parts.append("Your ACTIVE CLAIMS (deliver these first):\n" + "\n".join(lines))
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
