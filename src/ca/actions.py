"""Action registry: specs (tool schemas), permission gating, dispatch."""
from ca.config import LevelConfig
from ca.contracts import ContractError
from ca.infra import Infra
from ca.taskboard import BoardError


def _schema(props: dict, required: list[str]) -> dict:
    return {"type": "object", "properties": props, "required": required}


_S = {"type": "string"}
_I = {"type": "integer"}

ACTION_SPECS: dict[str, dict] = {
    # -------- billable (answer-related) --------
    "retrieve": {
        "description": "Search the external knowledge corpus. COSTS TOKENS.",
        "input_schema": _schema({"query": _S}, ["query"]),
    },
    "work_on": {
        "description": "Record one reasoning step about a task in your private scratchpad. COSTS TOKENS.",
        "input_schema": _schema({"task_id": _S, "thought": _S}, ["task_id", "thought"]),
    },
    "deliver_work": {
        "description": ("Deliver work. target_id starting with 'q' = submit final answer to the WORLD "
                        "(graded, paid by quality, one shot, COSTS TOKENS). target_id starting with 'c' "
                        "= deliver an accepted contract (escrow released to you, free)."),
        "input_schema": _schema({"target_id": _S, "content": _S}, ["target_id", "content"]),
    },
    # -------- free (coordination) --------
    "list_questions": {
        "description": "List open questions on the task board with prices.",
        "input_schema": _schema({}, []),
    },
    "claim_question": {
        "description": "Exclusively claim an open question (others can no longer see it).",
        "input_schema": _schema({"qid": _S}, ["qid"]),
    },
    "send_message": {
        "description": "Send a chat message to another agent.",
        "input_schema": _schema({"to": _S, "text": _S}, ["to", "text"]),
    },
    "read_chat": {
        "description": "Read recent chat history with another agent.",
        "input_schema": _schema({"with_agent": _S}, ["with_agent"]),
    },
    "propose_contract": {
        "description": ("Offer to PAY another agent to do `task` for you. Include `price` "
                        "when bargaining is allowed; under central pricing the interface "
                        "agent sets the price after you propose."),
        "input_schema": _schema({"to": _S, "task": _S, "price": _I}, ["to", "task"]),
    },
    "set_price": {
        "description": "INTERFACE ONLY (central pricing): set the final price of an unpriced contract.",
        "input_schema": _schema({"contract_id": _S, "price": _I}, ["contract_id", "price"]),
    },
    "accept_contract": {
        "description": "Accept a contract offer (price is locked in escrow from the payer).",
        "input_schema": _schema({"contract_id": _S}, ["contract_id"]),
    },
    "reject_contract": {
        "description": "Reject a contract offer.",
        "input_schema": _schema({"contract_id": _S}, ["contract_id"]),
    },
    "counter_offer": {
        "description": "Counter a pending contract with a new price.",
        "input_schema": _schema({"contract_id": _S, "price": _I}, ["contract_id", "price"]),
    },
    "cancel_contract": {
        "description": "Cancel a contract you are party to (escrow refunded to payer).",
        "input_schema": _schema({"contract_id": _S}, ["contract_id"]),
    },
    "pay": {
        "description": "Freely transfer tokens to another agent (tips, deposits, aid).",
        "input_schema": _schema({"to": _S, "amount": _I}, ["to", "amount"]),
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
        "description": "Save a note to your private long-term memory.",
        "input_schema": _schema({"content": _S}, ["content"]),
    },
    "memory_search": {
        "description": "Search your private long-term memory.",
        "input_schema": _schema({"query": _S}, ["query"]),
    },
    "check_balance": {
        "description": "Check your current token balance.",
        "input_schema": _schema({}, []),
    },
    "list_agents": {
        "description": "List all agents in the system.",
        "input_schema": _schema({}, []),
    },
}

_WORLD_ACTIONS = {"list_questions", "claim_question"}
_TARGETED = {"send_message", "propose_contract", "pay"}  # star-comms checked actions


def is_billable(name: str, inp: dict) -> bool:
    if name in ("retrieve", "work_on"):
        return True
    if name == "deliver_work":
        return str(inp.get("target_id", "")).startswith("q")
    return False


def visible_tools(level: LevelConfig, agent_id: str) -> list[dict]:
    is_iface = agent_id == "interface"
    out = []
    for name, spec in ACTION_SPECS.items():
        if level.world_access == "interface" and not is_iface and name in _WORLD_ACTIONS:
            continue
        if level.retrieve_access == "interface" and not is_iface and name == "retrieve":
            continue
        if name == "counter_offer" and level.central_pricing:
            continue  # bargaining disabled for everyone
        if name == "set_price" and not (level.central_pricing and is_iface):
            continue
        out.append({"name": name, **spec})
    return out


def permission_error(infra: Infra, agent_id: str, name: str, inp: dict) -> str | None:
    level = infra.cfg.level
    is_iface = agent_id == "interface"
    # world access (incl. deliver to WORLD)
    world_call = name in _WORLD_ACTIONS or (
        name == "deliver_work" and str(inp.get("target_id", "")).startswith("q"))
    if world_call and level.world_access == "interface" and not is_iface:
        return "only the interface agent may interact with the task board"
    if name == "retrieve" and level.retrieve_access == "interface" and not is_iface:
        return "only the interface agent may retrieve external information"
    # star comms
    if level.star_comms and not is_iface:
        if name in _TARGETED and inp.get("to") != "interface":
            return "at this configuration you may only interact with the interface agent"
        if name == "read_chat" and inp.get("with_agent") != "interface":
            return "at this configuration you may only interact with the interface agent"
    # pricing centralization: interface monopolizes ALL contract pricing
    if level.central_pricing:
        if name == "counter_offer":
            return "all contract prices are set by the interface agent; bargaining is disabled"
        if name == "set_price" and not is_iface:
            return "only the interface agent may set contract prices"
    elif name == "set_price":
        return "set_price does not exist in this configuration (prices are negotiated)"
    # bankruptcy freezes billable actions
    if is_billable(name, inp) and infra.ledger.is_bankrupt(agent_id):
        return "you are bankrupt (balance <= 0): answer-related actions are frozen"
    return None


def dispatch(infra: Infra, agent_id: str, name: str, inp: dict) -> str:
    try:
        return _HANDLERS[name](infra, agent_id, inp)
    except (BoardError, ContractError, KeyError, ValueError, IndexError) as e:
        return f"ERROR: {e}"


# ---------------- handlers ----------------

def _h_retrieve(infra, a, inp):
    hits = infra.retriever.search(inp["query"], k=5)
    return "\n\n".join(f"[{d['title']}] {d['text']}" for d in hits) or "(no results)"


def _h_work_on(infra, a, inp):
    infra.scratchpads[a][inp["task_id"]].append(inp["thought"])
    return f"noted on scratchpad for {inp['task_id']} ({len(infra.scratchpads[a][inp['task_id']])} entries)"


def _h_deliver_work(infra, a, inp):
    tid, content = inp["target_id"], inp["content"]
    if tid.startswith("q"):
        score, payout = infra.board.deliver(a, tid, content)
        return f"answer to {tid} graded: F1={score:.2f}, paid {payout} tokens"
    c = infra.contracts.deliver(a, tid, content)
    infra.chat.send(a, c.proposer, f"[deliverable for {c.cid}] {content}", infra.round)
    return f"delivered {c.cid}; escrow of {c.price} tokens released to you"


def _h_list_questions(infra, a, inp):
    qs = infra.board.list_open()
    if not qs:
        return "(no open questions)"
    return "\n".join(f"{q.qid} [{q.difficulty}, reward {q.price}]: {q.text}" for q in qs)


def _h_claim_question(infra, a, inp):
    q = infra.board.claim(a, inp["qid"])
    return f"claimed {q.qid}: {q.text} (reward up to {q.price})"


def _h_send_message(infra, a, inp):
    if inp["to"] not in infra.agent_ids:
        return f"ERROR: unknown agent {inp['to']}"
    infra.chat.send(a, inp["to"], inp["text"], infra.round)
    return f"sent to {inp['to']}"


def _h_read_chat(infra, a, inp):
    msgs = infra.chat.history(a, inp["with_agent"])
    return "\n".join(f"[r{m.round_no}] {m.sender}: {m.text}" for m in msgs) or "(no history)"


def _h_propose_contract(infra, a, inp):
    if inp["to"] not in infra.agent_ids:
        return f"ERROR: unknown agent {inp['to']}"
    central = infra.cfg.level.central_pricing
    if central and a != "interface":
        # price (if any) is ignored: the interface will set it
        c = infra.contracts.propose(a, inp["to"], inp["task"])
        infra.chat.send(a, "interface",
                        f"[contract {c.cid} awaits your pricing] {a} -> {inp['to']}: {c.task}",
                        infra.round)
        infra.chat.send(a, inp["to"],
                        f"[contract offer {c.cid}, price pending interface] task: {c.task}",
                        infra.round)
        return f"proposed {c.cid} to {inp['to']}; awaiting interface pricing"
    if inp.get("price") is None:
        return "ERROR: price is required (bargaining configuration)"
    c = infra.contracts.propose(a, inp["to"], inp["task"], int(inp["price"]))
    infra.chat.send(a, inp["to"],
                    f"[contract offer {c.cid}] task: {c.task} | price: {c.price}", infra.round)
    return f"proposed {c.cid} to {inp['to']} at {c.price}"


def _h_set_price(infra, a, inp):
    c = infra.contracts.set_price(inp["contract_id"], int(inp["price"]))
    infra.chat.send(a, c.proposer, f"[{c.cid} priced] {c.price} tokens", infra.round)
    infra.chat.send(a, c.contractor,
                    f"[{c.cid} priced] {c.price} tokens; accept or reject", infra.round)
    return f"priced {c.cid} at {c.price}; awaiting {c.contractor}"


def _h_accept_contract(infra, a, inp):
    c = infra.contracts.accept(a, inp["contract_id"])
    other = c.proposer if a == c.contractor else c.contractor
    infra.chat.send(a, other, f"[{c.cid} accepted] price {c.price} in escrow", infra.round)
    return f"accepted {c.cid}; {c.price} locked in escrow from {c.proposer}"


def _h_reject_contract(infra, a, inp):
    c = infra.contracts.reject(a, inp["contract_id"])
    other = c.proposer if a != c.proposer else c.contractor
    infra.chat.send(a, other, f"[{c.cid} rejected]", infra.round)
    return f"rejected {c.cid}"


def _h_counter_offer(infra, a, inp):
    c = infra.contracts.counter(a, inp["contract_id"], int(inp["price"]))
    infra.chat.send(a, c.awaiting, f"[{c.cid} counter-offer] new price: {c.price}", infra.round)
    return f"countered {c.cid} at {c.price}; awaiting {c.awaiting}"


def _h_cancel_contract(infra, a, inp):
    c = infra.contracts.cancel(a, inp["contract_id"])
    other = c.proposer if a != c.proposer else c.contractor
    infra.chat.send(a, other, f"[{c.cid} cancelled]", infra.round)
    return f"cancelled {c.cid}"


def _h_pay(infra, a, inp):
    infra.ledger.transfer(a, inp["to"], int(inp["amount"]))
    infra.chat.send(a, inp["to"], f"[payment] {inp['amount']} tokens", infra.round)
    return f"paid {inp['amount']} to {inp['to']}"


def _h_push_goal(infra, a, inp):
    return "__PUSH_GOAL__"  # handled by Agent (owns the stack); see agent.py


def _h_pop_goal(infra, a, inp):
    return "__POP_GOAL__"


def _h_memory_write(infra, a, inp):
    infra.ltm.write(a, inp["content"])
    return "saved to long-term memory"


def _h_memory_search(infra, a, inp):
    hits = infra.ltm.search(a, inp["query"])
    return "\n".join(f"- {h}" for h in hits) or "(no matching memories)"


def _h_check_balance(infra, a, inp):
    return f"balance: {infra.ledger.balance(a)} tokens"


def _h_list_agents(infra, a, inp):
    return ", ".join(infra.agent_ids)


_HANDLERS = {
    "retrieve": _h_retrieve, "work_on": _h_work_on, "deliver_work": _h_deliver_work,
    "list_questions": _h_list_questions, "claim_question": _h_claim_question,
    "send_message": _h_send_message, "read_chat": _h_read_chat,
    "propose_contract": _h_propose_contract, "accept_contract": _h_accept_contract,
    "reject_contract": _h_reject_contract, "counter_offer": _h_counter_offer,
    "cancel_contract": _h_cancel_contract, "set_price": _h_set_price, "pay": _h_pay,
    "push_goal": _h_push_goal, "pop_goal": _h_pop_goal,
    "memory_write": _h_memory_write, "memory_search": _h_memory_search,
    "check_balance": _h_check_balance, "list_agents": _h_list_agents,
}
