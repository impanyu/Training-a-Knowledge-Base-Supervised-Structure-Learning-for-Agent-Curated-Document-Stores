"""Builds the LLM context: stable system prompt + per-turn dynamic view."""
from ca.config import LevelConfig
from ca.infra import Infra
from ca.memory import FifoMemory, GoalStack
from ca.skills import role_skill

_BASE = """You are {agent_id}, an autonomous agent in a multi-agent economy.
Agents in the system: {peers}.

YOUR PERMANENT ROOT GOAL: maximize your token balance.
Tokens are both your money and your fuel: answer-related actions (retrieve,
work_on, and delivering answers to the WORLD) consume tokens equal to the LLM
cost of that turn. Coordination actions (chat, contracts, payments, memory,
goals) are free. If your balance drops to 0 or below you are BANKRUPT and can
no longer perform answer-related actions.

You earn tokens ONLY from: (a) delivering correct answers to the WORLD's
questions (paid = price x answer quality F1, one attempt per question), or
(b) payments from other agents (contract escrow settlements or transfers).

Each turn you must choose EXACTLY ONE action (tool call). Unread messages
are delivered automatically into your context every turn - you never need
to poll read_chat (use it only to re-read older history). Think about
profitability: estimate what a question will cost to answer vs its reward.
You may subcontract work to other agents via contracts.
{level_rules}"""

_INTERFACE_EXTRA = """
YOU ARE THE INTERFACE AGENT: the only agent allowed to take questions from
the task board and deliver answers to the WORLD. Other agents can work for
you via contracts. Your profit = WORLD rewards minus what you pay them."""


def _level_rules(level: LevelConfig, is_iface: bool) -> str:
    rules = []
    if level.world_access == "interface":
        rules.append("Only the interface agent can list/claim questions and deliver answers to the WORLD.")
    if level.retrieve_access == "interface":
        rules.append("Only the interface agent can retrieve external information; others must ask it via chat/contracts.")
    if level.central_pricing:
        rules.append("ALL contract prices are set by the interface agent (set_price); bargaining is disabled."
                     if not is_iface else
                     "You set the price of EVERY contract in the system via set_price; nobody can bargain.")
    if level.star_comms:
        rules.append("You may only message/contract/pay the interface agent."
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
    is_iface = agent_id == "interface"
    sp = _BASE.format(agent_id=agent_id,
                      peers=", ".join(all_ids),
                      level_rules=_level_rules(level, is_iface))
    if is_iface:
        sp += _INTERFACE_EXTRA
    sp += role_skill(level, agent_id)
    return sp


def render_turn(infra: Infra, agent_id: str, fifo: FifoMemory, goals: GoalStack) -> str:
    parts = [f"== ROUND {infra.round} ==",
             f"Balance: {infra.ledger.balance(agent_id)} tokens"]
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
    mine = [q for q in infra.board.questions.values()
            if q.status == "claimed" and q.claimed_by == agent_id]
    if mine:
        ttl = infra.cfg.claim_ttl
        lines = []
        for q in mine:
            left = q.claimed_round + ttl - infra.round
            lines.append(f"- {q.qid} (reward {q.price}): {q.text}  "
                         f"[claim EXPIRES in {max(left,0)} round(s) - deliver before then!]")
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
    if agent_id == "interface" and infra.cfg.level.central_pricing:
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
