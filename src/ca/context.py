"""Builds the LLM context: stable system prompt + per-turn dynamic view."""
from ca.config import LevelConfig
from ca.infra import Infra
from ca.memory import FifoMemory, GoalStack
from ca.skills import role_skill

ROOT_GOAL = ("Cooperate with the other agents to answer as many questions "
             "correctly as possible.")
ROOT_GOAL_SOLO = "Answer as many questions correctly as possible."

_BASE = """You are {agent_id}, an autonomous agent in a multi-agent system.
Agents in the system: {peers}.

YOUR PERMANENT ROOT GOAL: {root_goal}
This is a SHARED objective: there is no private score, and a question answered
by any agent counts exactly as much as one you answer yourself.

The WORLD posts QUESTIONS. A task is ONE question, claimed and answered on its
own; the pipeline is always: list_questions -> claim_question -> memory_search
-> deliver ONE short answer. `claim_question(qid="q0042")` takes the question
(and hands back the answer already in your memory, if any), then
`deliver_work(target_id="q0042", content="Richard Strauss")`. `content` is a
bare string: ONLY the short answer itself (a name / date / phrase) - never a
sentence or an explanation - because it is graded by token-overlap F1 against
a short gold answer.

A delivery is your ONE graded attempt on that claim.
A claim does not expire, so you may take as long as a question needs - but a
question holds ONE claimant at a time, so a question you are not going to
finish should go back to the board with
`release_question(qid="q0042")` for someone else to take.

Your long-term memory was BORN KNOWING the WORLD's whole knowledge corpus
(~12k encyclopedia paragraphs): memory_search is how you look facts up, and
the notes you write and answers you deliver are stored alongside, so your
memory only grows more valuable. Every answer you deliver is stored
automatically, and claiming a question hands the stored answer back with its
F1, so you can reuse it as-is or re-solve it if it scored badly.

Your peers are the other half of your knowledge: what one of them looked up is
a search you do not have to repeat. Ask them (send_message) when a question
needs a fact you cannot find, and answer their questions from your own memory
when they ask you.

Each turn you must choose EXACTLY ONE action (tool call). Unread messages
are delivered automatically into your context every turn - you never need
to poll read_chat (use it only to re-read older history). Turns are the only
scarce resource: spend them where they add answers.
{level_rules}"""

_HUB_EXTRA_DEMAND = """
YOU ARE THE HUB AGENT: the only agent allowed to take questions from the
question board and deliver answers to the WORLD. The other agents cannot see
the board at all, so the only way their knowledge reaches the WORLD is if you
ask them and deliver what they send back."""

# C5: the hub holds exactly ONE power (named in the configuration rules above)
# and is an ordinary participant in every other respect -- it must not be told
# it monopolizes the question board.
_HUB_EXTRA = """
YOU ARE THE HUB AGENT: the hub of this configuration. Apart from the one
privilege named in the configuration rules above you are an ordinary agent -
every other agent may claim questions and deliver to the WORLD exactly as you
can."""


def _level_rules(level: LevelConfig, is_iface: bool) -> str:
    rules = []
    if level.world_access == "hub":
        rules.append("Only the hub agent can list/claim questions and deliver "
                     "answers to the WORLD.")
    if level.star_comms:
        rules.append("You may only message the hub agent."
                     if not is_iface else
                     "Other agents can only talk to you, not to each other.")
    if level.shared_memory:
        rules.append("Long-term memory is SHARED by every agent: your notes and answers "
                     "are visible to all, and theirs to you.")
    if not rules:
        return ("\nThis is a fully decentralized configuration: "
                "every agent has equal access to everything.")
    return "\nConfiguration rules:\n" + "\n".join(f"- {r}" for r in rules)


def system_prompt(level: LevelConfig, agent_id: str, all_ids: list[str]) -> str:
    is_iface = agent_id == "hub"
    sp = _BASE.format(agent_id=agent_id,
                      peers=", ".join(all_ids),
                      root_goal=ROOT_GOAL_SOLO if level.n_agents == 1 else ROOT_GOAL,
                      level_rules=_level_rules(level, is_iface))
    if is_iface:
        sp += (_HUB_EXTRA_DEMAND if level.world_access == "hub"
               else _HUB_EXTRA)
    sp += role_skill(level, agent_id)
    return sp


def render_turn(infra: Infra, agent_id: str, fifo: FifoMemory, goals: GoalStack) -> str:
    parts = [f"== ROUND {infra.round} =="]
    parts.append("Goal stack (bottom -> top):\n" + goals.render())
    mine = [qid for qid, c in infra.board.active.items() if c.agent == agent_id]
    if mine:
        lines = []
        for qid in mine:
            q = infra.bank.questions[qid]
            lines.append(f"- [{qid}] {q.text} ({q.difficulty}) - "
                         f'deliver_work(target_id="{qid}", content="<answer>") '
                         f'or release_question(qid="{qid}")')
        parts.append("Your active claims:\n" + "\n".join(lines))
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
