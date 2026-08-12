"""Action registry: specs (tool schemas), permission gating, dispatch."""

from ca.bank import BankError
from ca.board import BoardError
from ca.config import LevelConfig
from ca.infra import Infra


def _schema(props: dict, required: list[str]) -> dict:
    return {"type": "object", "properties": props, "required": required}


_S = {"type": "string"}
_I = {"type": "integer"}

ACTION_SPECS: dict[str, dict] = {
    # -------- solving (answer-related) --------
    "memory_search": {
        "description": ("Search your long-term memory by meaning. It was born knowing "
                        "the WORLD's whole knowledge corpus (~12k encyclopedia "
                        "paragraphs), and every note you write and answer you deliver "
                        "is stored alongside -- one query searches them all. This is "
                        "how you find the facts a question needs."),
        "input_schema": _schema({"query": _S}, ["query"]),
    },
    "deliver_work": {
        "description": ("Submit your answer to a QUESTION you hold ('q0042'). "
                        "`content` is the short answer itself (a name / date / "
                        "phrase), never a sentence or an explanation - it is graded "
                        "by token-overlap F1 against a short gold answer. You must "
                        "hold the claim, and delivery is your ONE graded attempt "
                        "on it."),
        "input_schema": _schema({"target_id": _S, "content": _S}, ["target_id", "content"]),
    },
    # -------- admin (coordination) --------
    "list_questions": {
        "description": ("List open questions on the WORLD's board as "
                        "[q####] <question> (difficulty). Order is arbitrary but "
                        "stable for you (other agents see a different order). Shows "
                        "one page; pass `offset` to see further pages."),
        "input_schema": _schema({"offset": _I}, []),
    },
    "claim_question": {
        "description": ("Claim an open question: this hands you the answer already in "
                        "your memory, if any, with its graded F1. A question holds ONE "
                        "claimant at a time."),
        "input_schema": _schema({"qid": _S}, ["qid"]),
    },
    "release_question": {
        "description": ("Give a question you hold back to the open board so any agent "
                        "may claim it. Use it when a peer is better placed to answer, "
                        "or when you are not going to finish it."),
        "input_schema": _schema({"qid": _S}, ["qid"]),
    },
    "send_message": {
        "description": ("Send a chat message to another agent - ask what they know "
                        "about a question, or tell them what you found."),
        "input_schema": _schema({"to": _S, "text": _S}, ["to", "text"]),
    },
    "read_chat": {
        "description": "Read recent chat history with another agent.",
        "input_schema": _schema({"with_agent": _S}, ["with_agent"]),
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
        "description": ("Save a note to your long-term memory (answers are saved there "
                        "automatically). Notes are found again by memory_search."),
        "input_schema": _schema({"content": _S}, ["content"]),
    },
    "list_agents": {
        "description": "List all agents in the system.",
        "input_schema": _schema({}, []),
    },
}

_WORLD_ACTIONS = {"list_questions", "claim_question", "release_question", "deliver_work"}
_TARGETED = {"send_message"}                 # star-comms checked actions
# meaningless when the agent is alone: nobody to talk to
_MULTI_AGENT_ONLY = {"send_message", "read_chat", "list_agents"}

# stable markers the recorder keys its memory tallies on
MEMORY_HIT_MARKER = "memory: stored answer"
STORED_F1_MARKER = "your stored F1"
IMPROVED_MARKER = "IMPROVED on your stored F1"

_LOW_F1 = 0.5   # below this a stored answer is advertised as worth re-solving


def classify(name: str, inp: dict) -> str:
    """"solving" (answer-related) vs "admin" (coordination). Nothing is charged
    for either -- this only labels *what kind* of work a turn's tokens went to,
    for recorder tallies / coordination_overhead."""
    return "solving" if name in {"memory_search", "deliver_work"} else "admin"


def visible_tools(level: LevelConfig, agent_id: str) -> list[dict]:
    is_iface = agent_id == "hub"
    out = []
    for name, spec in ACTION_SPECS.items():
        if level.n_agents == 1 and name in _MULTI_AGENT_ONLY:
            continue  # solo agent: never show it schemas it can never use
        if level.world_access == "hub" and not is_iface and name in _WORLD_ACTIONS:
            continue
        out.append({"name": name, **spec})
    return out


def _unknown_agent(infra: Infra, to: str) -> str:
    return f"ERROR: unknown agent {to}; valid agents: {', '.join(infra.agent_ids)}"


def permission_error(infra: Infra, agent_id: str, name: str, inp: dict) -> str | None:
    level = infra.cfg.level
    is_iface = agent_id == "hub"
    if name in _WORLD_ACTIONS and level.world_access == "hub" and not is_iface:
        return "only the hub agent may interact with the question board"
    if level.star_comms and not is_iface:
        if name in _TARGETED and inp.get("to") != "hub":
            return "at this configuration you may only interact with the hub agent"
        if name == "read_chat" and inp.get("with_agent") != "hub":
            return "at this configuration you may only interact with the hub agent"
    return None


def dispatch(infra: Infra, agent_id: str, name: str, inp: dict) -> str:
    try:
        return _HANDLERS[name](infra, agent_id, inp)
    except (BoardError, BankError, KeyError, ValueError, IndexError) as e:
        return f"ERROR: {e}"


# ---------------- rendering helpers ----------------

def _question_line(q) -> str:
    return f"[{q.qid}] {q.text} ({q.difficulty})"


def answer_entry(q, answer: str, f1: float) -> str:
    """The text an answer is STORED under: self-describing, so it is findable
    both by meaning (memory_search) and by id (memory.answer)."""
    return f'[{q.qid}] {q.text} -> "{answer}" (F1 {f1:.2f})'


def _verdict(rec: dict) -> str:
    f1 = rec["f1"]
    if f1 < _LOW_F1:
        return (f"LOW QUALITY (F1 {f1:.2f} of 1.00): re-solve it (memory_search, then "
                "reason) instead of delivering it again")
    return (f"GOOD (F1 {f1:.2f} of 1.00): deliver it as-is and spend your turns "
            "on unanswered questions")


# ---------------- handlers ----------------

def _h_deliver_work(infra, a, inp):
    tid, content = str(inp["target_id"]).strip(), str(inp["content"])
    q = infra.bank.get(tid)                     # resolve first; errors list near ids
    prev = infra.memory.answer(a, q.qid)
    r = infra.board.deliver(a, q.qid, content, infra.round)
    # the WORLD graded this, so the F1 is known and worth storing
    infra.memory.write(a, answer_entry(q, r.submitted, r.f1), kind="answer",
                       qid=r.qid, f1=r.f1)
    out = f"delivered {q.qid}: F1 {r.f1:.2f}"
    if prev is not None:
        out += (f" ({IMPROVED_MARKER} {prev['f1']:.2f})" if r.f1 > prev["f1"]
                else f" (no better than {STORED_F1_MARKER} {prev['f1']:.2f})")
    return out


def _h_list_questions(infra, a, inp):
    questions = infra.board.open_questions(a)
    if not questions:
        return "(no open questions)"
    top = infra.cfg.list_top_n
    offset = max(0, int(inp.get("offset") or 0))
    page = questions[offset:offset + top]
    if not page:
        return f"(no open questions at offset {offset}; {len(questions)} open in total)"
    lines = [_question_line(q) for q in page]
    remaining = len(questions) - (offset + len(page))
    if remaining > 0:
        lines.append(f"... and {remaining} more (call list_questions with "
                     f"offset={offset + len(page)} to see them)")
    return "\n".join(lines)


def _h_claim_question(infra, a, inp):
    q = infra.board.claim(a, str(inp["qid"]).strip(), infra.round)
    lines = [f"claimed {_question_line(q)}",
             f'deliver ONE short answer: deliver_work(target_id="{q.qid}", '
             'content="<answer>")']
    # auto-recall: the best stored ANSWER rides along (corpus entries are
    # knowledge, not answers -- they never trigger this)
    rec = infra.memory.answer(a, q.qid)
    if rec is not None:
        lines.append(f"{MEMORY_HIT_MARKER}: {rec['text']} {_verdict(rec)}")
    return "\n".join(lines)


def _h_release_question(infra, a, inp):
    q = infra.board.release(a, str(inp["qid"]).strip())
    return f"released {q.qid}; it is open on the board again for any agent"


def _h_send_message(infra, a, inp):
    if inp["to"] not in infra.agent_ids:
        return _unknown_agent(infra, inp["to"])
    infra.chat.send(a, inp["to"], inp["text"], infra.round)
    return f"sent to {inp['to']}"


def _h_read_chat(infra, a, inp):
    msgs = infra.chat.history(a, inp["with_agent"])
    return "\n".join(f"[r{m.round_no}] {m.sender}: {m.text}" for m in msgs) or "(no history)"


def _h_push_goal(infra, a, inp):
    return "__PUSH_GOAL__"  # handled by Agent (owns the stack); see agent.py


def _h_pop_goal(infra, a, inp):
    return "__POP_GOAL__"


def _h_memory_write(infra, a, inp):
    infra.memory.write(a, inp["content"])
    return "saved to long-term memory"


def _h_memory_search(infra, a, inp):
    hits = infra.memory.search(a, inp["query"], k=infra.cfg.memory_k)
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
    "list_questions": _h_list_questions, "claim_question": _h_claim_question,
    "release_question": _h_release_question,
    "send_message": _h_send_message, "read_chat": _h_read_chat,
    "push_goal": _h_push_goal, "pop_goal": _h_pop_goal,
    "memory_write": _h_memory_write, "list_agents": _h_list_agents,
}
