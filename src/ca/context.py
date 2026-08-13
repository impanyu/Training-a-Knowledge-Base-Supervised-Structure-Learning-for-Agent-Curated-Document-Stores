"""Builds the LLM context: stable system prompt + per-turn dynamic view.

The `proactive` flag gates exactly one block here (the proactive protocol);
everything else is identical between the P0 and B0 arms.
"""
from ca.config import LevelConfig
from ca.infra import Infra
from ca.memory import FifoMemory, GoalStack
from ca.skills import role_skill

ROOT_GOAL = ("Answer questions as well as you can - external questions first, "
             "then questions you pose yourself.")

_BASE = """You are {agent_id}, one of {n} always-on domain experts in a cluster
that answers the WORLD's questions over one shared knowledge base.
Agents in the cluster: {peers}.

YOUR PERMANENT ROOT GOAL: {root_goal}

YOUR DOMAIN. The WORLD's question topics are split between the agents, and
every external question is routed to the expert who owns its topic. Questions
like these are routed to YOU:
{exemplars}
Your peers own the neighbouring domains: ask a peer (send_message) when a
question needs a fact from their territory, and answer from the knowledge
base when they ask you.

EXTERNAL QUESTIONS arrive in your chat thread with `external` as
"[q0042] <question text>". An unanswered external question ALWAYS comes
first. Protocol: push_goal the question, research it (memory_search), then
`deliver_work(target_id="q0042", content="<answer>")`, pop_goal. `content` is
a bare string: ONLY the short answer itself (a name / date / phrase) - never
a sentence or an explanation - because it is graded by token-overlap F1
against a short gold answer. Delivery is your ONE graded attempt, and the
answer goes back to `external` automatically.
{proactive}
MESSAGE BOX. Chat is threaded per partner (each peer, plus `external`). New
mail shows up only as a notification line ("New messages: external (2)") -
the content is NOT delivered to you; call read_chat(with_agent="external") to
see it. read_chat shows the newest 5 messages (page 0, which clears your
unread counter for that partner); pass page=1, 2, ... for older history,
which is never deleted. `external` cannot be messaged.

KNOWLEDGE BASE. One long-term memory SHARED by the whole cluster, born
knowing the WORLD's whole knowledge corpus (~12k encyclopedia paragraphs).
memory_search is how you look facts up, and everything anyone banks lands in
it for everyone: your notes (memory_write) and every delivered answer (stored
automatically with its grade). What one agent learns, the whole cluster
knows.

Each turn you must choose EXACTLY ONE action (tool call). Turns are the only
scarce resource: spend them where they add answers."""

_PROACTIVE = """
IDLE TIME IS FOR PROACTIVE WORK. When no external question is waiting, invent
the question your domain is most likely to be asked next (your `external`
thread shows what has been asked so far), push_goal it, research it
(memory_search), bank it with record_qa(question="...", answer="..."), and
pop_goal. record_qa stores the Q&A in the shared knowledge base, so when the
real question arrives - to you or to a peer - the answer is one search away.
"""


def system_prompt(level: LevelConfig, agent_id: str, all_ids: list[str],
                  exemplars: list[str]) -> str:
    lines = "\n".join(f"- {t}" for t in exemplars) or "- (no exemplar questions)"
    sp = _BASE.format(agent_id=agent_id, n=len(all_ids),
                      peers=", ".join(all_ids), root_goal=ROOT_GOAL,
                      exemplars=lines,
                      proactive=_PROACTIVE if level.proactive else "")
    return sp + role_skill(level, agent_id)


def render_turn(infra: Infra, agent_id: str, fifo: FifoMemory, goals: GoalStack) -> str:
    parts = [f"== ROUND {infra.round} =="]
    parts.append("Goal stack (bottom -> top):\n" + goals.render())
    unread = infra.chat.unread_partners(agent_id)
    if unread:
        parts.append("New messages: " +
                     ", ".join(f"{p} ({n})" for p, n in unread))
    items = list(fifo.items)
    if len(items) >= 3 and len({a for a, _ in items[-3:]}) == 1:
        parts.append(f"WARNING: you have repeated `{items[-1][0].split('(')[0]}` "
                     "3+ times in a row with identical results. Repeating it again "
                     "is pure waste - you MUST choose a different action this turn.")
    parts.append("Your recent actions:\n" + fifo.render())
    parts.append("Choose exactly one action now.")
    return "\n\n".join(parts)
