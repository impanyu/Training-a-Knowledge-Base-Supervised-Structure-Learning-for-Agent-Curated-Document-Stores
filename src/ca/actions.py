"""Action registry: specs (tool schemas), the proactive gate, dispatch.

Nine actions. Solving: memory_search, deliver_work, record_qa. Admin:
memory_write, send_message, read_chat, push_goal, pop_goal, list_agents.
The ONLY catalog difference between the arms is record_qa, present at P0 and
absent at B0 -- there is no other permission machinery left.
"""

from ca.bank import BankError
from ca.chat import EXTERNAL
from ca.config import LevelConfig
from ca.infra import Infra
from ca.stream import StreamError


def _schema(props: dict, required: list[str]) -> dict:
    return {"type": "object", "properties": props, "required": required}


_S = {"type": "string"}
_I = {"type": "integer"}

ACTION_SPECS: dict[str, dict] = {
    # -------- solving (answer-related) --------
    "memory_search": {
        "description": ("Search the cluster's shared knowledge base by meaning. It was "
                        "born knowing the whole knowledge corpus (~12k encyclopedia "
                        "paragraphs), and every note, recorded Q&A and delivered answer "
                        "-- yours or a peer's -- is stored alongside; one query searches "
                        "them all. This is how you find the facts a question needs."),
        "input_schema": _schema({"query": _S}, ["query"]),
    },
    "deliver_work": {
        "description": ("Deliver your answer to an external question from your "
                        "`external` thread ('q0042'). `content` is the short answer "
                        "itself (a name / date / phrase), never a sentence or an "
                        "explanation - it is graded by token-overlap F1 against a short "
                        "gold answer. Only the assignee may deliver, delivery is your "
                        "ONE graded attempt, and the answer is sent back to `external` "
                        "automatically."),
        "input_schema": _schema({"target_id": _S, "content": _S}, ["target_id", "content"]),
    },
    "record_qa": {
        "description": ("Bank a question you posed yourself: stores 'Q: ... / A: ...' "
                        "in the shared knowledge base, visible to every agent's "
                        "memory_search. This is how idle-time work pays off when the "
                        "real question arrives."),
        "input_schema": _schema({"question": _S, "answer": _S}, ["question", "answer"]),
    },
    # -------- admin (coordination) --------
    "send_message": {
        "description": ("Send a chat message to another agent - ask what they know, or "
                        "tell them what you found. `external` cannot be messaged: "
                        "answers reach the outside only through deliver_work."),
        "input_schema": _schema({"to": _S, "text": _S}, ["to", "text"]),
    },
    "read_chat": {
        "description": ("Read your chat thread with one partner (`external` or a "
                        "peer). Page 0 is the newest 5 messages and clears your unread "
                        "counter for that partner; page 1, 2, ... go further back. "
                        "History is never deleted."),
        "input_schema": _schema({"with_agent": _S, "page": _I}, ["with_agent"]),
    },
    "push_goal": {
        "description": "Push a sub-goal note onto your goal stack.",
        "input_schema": _schema({"note": _S}, ["note"]),
    },
    "pop_goal": {
        "description": "Pop the top goal off your goal stack (root goal cannot be popped).",
        "input_schema": _schema({}, []),
    },
    "memory_write": {
        "description": ("Save a free-form note to the shared knowledge base (delivered "
                        "answers are saved there automatically). Notes are found again "
                        "by memory_search - by you and by every peer."),
        "input_schema": _schema({"content": _S}, ["content"]),
    },
    "list_agents": {
        "description": "List all agents in the cluster.",
        "input_schema": _schema({}, []),
    },
}

_SOLVING = {"memory_search", "deliver_work", "record_qa"}


def classify(name: str, inp: dict) -> str:
    """"solving" (answer-related) vs "admin" (coordination). Nothing is charged
    for either -- this only labels *what kind* of work a turn's tokens went to,
    for recorder tallies / coordination_overhead."""
    return "solving" if name in _SOLVING else "admin"


def visible_tools(level: LevelConfig, agent_id: str) -> list[dict]:
    return [{"name": name, **spec} for name, spec in ACTION_SPECS.items()
            if name != "record_qa" or level.proactive]


def dispatch(infra: Infra, agent_id: str, name: str, inp: dict) -> str:
    try:
        return _HANDLERS[name](infra, agent_id, inp)
    except (StreamError, BankError, KeyError, ValueError, IndexError) as e:
        return f"ERROR: {e}"


# ---------------- rendering helpers ----------------

def answer_entry(q, answer: str, f1: float) -> str:
    """The text a delivered answer is STORED under: self-describing, so it is
    findable by meaning alongside the question it answers."""
    return f'[{q.qid}] {q.text} -> "{answer}" (F1 {f1:.2f})'


# ---------------- handlers ----------------

def _h_deliver_work(infra, a, inp):
    qid, content = str(inp["target_id"]).strip(), str(inp["content"])
    r = infra.stream.deliver(a, qid, content, infra.round)
    q = infra.bank.questions[r.qid]
    # the WORLD graded this, so the F1 is known and worth storing for everyone
    infra.memory.write(a, answer_entry(q, r.submitted, r.f1), kind="answer",
                       qid=r.qid, f1=r.f1)
    infra.chat.send(a, EXTERNAL, f"[{r.qid}] {r.submitted}", infra.round)
    return f"delivered {r.qid}: F1 {r.f1:.2f}"


def _h_record_qa(infra, a, inp):
    question, answer = str(inp["question"]).strip(), str(inp["answer"]).strip()
    infra.memory.write(a, f"Q: {question}\nA: {answer}", kind="selfqa")
    return "recorded to the shared knowledge base"


def _h_send_message(infra, a, inp):
    to = str(inp["to"]).strip()
    if to == EXTERNAL:
        return ("ERROR: 'external' is a reserved partner - answers reach the "
                "outside only through deliver_work")
    if to == a:
        return "ERROR: you cannot message yourself"
    if to not in infra.agent_ids:
        return f"ERROR: unknown agent {to}; valid agents: {', '.join(infra.agent_ids)}"
    infra.chat.send(a, to, inp["text"], infra.round)
    return f"sent to {to}"


def _h_read_chat(infra, a, inp):
    partner = str(inp["with_agent"]).strip()
    if partner != EXTERNAL and partner not in infra.agent_ids:
        return (f"ERROR: unknown partner {partner}; valid partners: "
                f"{', '.join([EXTERNAL] + [x for x in infra.agent_ids if x != a])}")
    page = max(0, int(inp.get("page") or 0))
    msgs, older = infra.chat.read(a, partner, page)
    if not msgs:
        return (f"(no messages with {partner})" if page == 0
                else f"(no messages with {partner} at page {page})")
    lines = [f"[r{m.round_no}] {m.sender}: {m.text}" for m in msgs]
    if older:
        lines.append(f"({older} older: read_chat(with_agent=\"{partner}\", "
                     f"page={page + 1}))")
    return "\n".join(lines)


def _h_push_goal(infra, a, inp):
    return "__PUSH_GOAL__"  # handled by Agent (owns the stack); see agent.py


def _h_pop_goal(infra, a, inp):
    return "__POP_GOAL__"


def _h_memory_write(infra, a, inp):
    infra.memory.write(a, inp["content"])
    return "saved to the shared knowledge base"


def _h_memory_search(infra, a, inp):
    hits = infra.memory.search(inp["query"], k=infra.cfg.memory_k)
    lines = []
    for h in hits:
        if h["kind"] == "corpus":
            lines.append(f"- [{h['title']}] {h['text']}")
        else:
            lines.append(f"- {h['text']}")
    return "\n".join(lines) or "(no matching memories)"


def _h_list_agents(infra, a, inp):
    return ", ".join(infra.agent_ids)


_HANDLERS = {
    "memory_search": _h_memory_search, "deliver_work": _h_deliver_work,
    "record_qa": _h_record_qa,
    "send_message": _h_send_message, "read_chat": _h_read_chat,
    "push_goal": _h_push_goal, "pop_goal": _h_pop_goal,
    "memory_write": _h_memory_write, "list_agents": _h_list_agents,
}
