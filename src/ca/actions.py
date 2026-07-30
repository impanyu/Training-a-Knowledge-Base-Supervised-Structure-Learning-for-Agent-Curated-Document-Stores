"""Action registry: specs (tool schemas), permission gating, dispatch."""
import json
import re

from ca.config import LevelConfig
from ca.contracts import ContractError
from ca.economy import InsufficientFunds
from ca.infra import Infra
from ca.loans import LoanError
from ca.solutions import is_question
from ca.taskboard import BoardError
from ca.tasktree import TreeError


def _schema(props: dict, required: list[str]) -> dict:
    return {"type": "object", "properties": props, "required": required}


_S = {"type": "string"}
_I = {"type": "integer"}

ACTION_SPECS: dict[str, dict] = {
    # -------- solving (answer-related) --------
    "retrieve": {
        "description": "Search the external knowledge corpus. COSTS TOKENS.",
        "input_schema": _schema({"query": _S}, ["query"]),
    },
    "work_on": {
        "description": ("Record one reasoning step about a task, subtask or question in your "
                        "private scratchpad. COSTS TOKENS."),
        "input_schema": _schema({"task_id": _S, "thought": _S}, ["task_id", "thought"]),
    },
    "decompose": {
        "description": ("Reveal the direct children of a task or subtask: child subtasks are "
                        "shown as [t####] «one-sentence summary» (n questions, reward r), child "
                        "leaves as [q####] the full question text. Questions only become visible "
                        "once you decompose down to them. `node` accepts a short id (t0012) or "
                        "the one-sentence summary; passing a q-id just re-prints that question. "
                        "COSTS TOKENS."),
        "input_schema": _schema({"node": _S}, ["node"]),
    },
    "deliver_work": {
        "description": ("Deliver work. target_id = a TASK (short id 't0012' or its one-sentence "
                        "summary) submits the whole task to the WORLD: `content` must be a JSON "
                        "object {\"q0031\": \"answer\", ...} with one entry for EVERY leaf "
                        "question of that task - no missing and no extra q-ids. Each answer is "
                        "graded by token-overlap F1 against a short gold answer, so every JSON "
                        "VALUE must be ONLY the short answer itself (a name / date / phrase, "
                        "e.g. 'Richard Strauss'), never a sentence or explanation. You get ONE "
                        "graded attempt per task and are paid the sum of price x F1 over its "
                        "leaves; a malformed or incomplete map is rejected without using up that "
                        "attempt. COSTS TOKENS. target_id starting with 'c' = deliver an accepted "
                        "contract (escrow released to you) — only the CONTRACTOR (the agent "
                        "hired to do the work) delivers a contract; if you are the payer, wait for "
                        "the deliverable to arrive in your chat instead. If the contract was bound "
                        "to a subtask node, its content must likewise be a JSON map covering that "
                        "node's leaves."),
        "input_schema": _schema({"target_id": _S, "content": _S}, ["target_id", "content"]),
    },
    "recall_solutions": {
        "description": ("Look up what you have ALREADY solved under a task, subtask or "
                        "question, reusing your automatically-saved decompositions and "
                        "answers. `name` accepts a short id (t0012), a one-sentence "
                        "summary, or a question id (q0031). Expands the subtree "
                        "recursively and reports every known answer plus what is still "
                        "missing - a subtask you solved before comes back whole, for the "
                        "price of one action instead of re-solving it. COSTS TOKENS."),
        "input_schema": _schema({"name": _S}, ["name"]),
    },
    # -------- admin (coordination) --------
    "list_tasks": {
        "description": ("List open tasks on the WORLD's task board as "
                        "[t####] «one-sentence summary» (n questions, reward R), sorted by "
                        "reward (highest first). Shows one page; pass `offset` to see further "
                        "pages. Use decompose to see what a task actually contains."),
        "input_schema": _schema({"offset": _I}, []),
    },
    "claim_task": {
        "description": ("Exclusively claim an open task tree (others can no longer see or take "
                        "it). `task` accepts a short id (t0012) or its one-sentence summary."),
        "input_schema": _schema({"task": _S}, ["task"]),
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
                        "agent sets the price after you propose. If `task` names a subtask "
                        "node (short id or its one-sentence summary) the contract is BOUND to "
                        "that node: the contractor must then deliver a JSON map covering all "
                        "of its leaf questions. Any other text is a free-text contract."),
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
    "propose_loan": {
        "description": ("Ask another agent (`to`, the lender) to lend you `amount` tokens. "
                        "Interest accrues at cfg.loan_rate per round while the loan is active. "
                        "The lender may accept or ignore."),
        "input_schema": _schema({"to": _S, "amount": _I}, ["to", "amount"]),
    },
    "accept_loan": {
        "description": "As lender, accept a loan proposal awaiting you: transfers the principal to the borrower.",
        "input_schema": _schema({"loan_id": _S}, ["loan_id"]),
    },
    "repay_loan": {
        "description": "As borrower, repay (partially or fully) an active loan.",
        "input_schema": _schema({"loan_id": _S, "amount": _I}, ["loan_id", "amount"]),
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

_WORLD_ACTIONS = {"list_tasks", "claim_task"}
_TARGETED = {"send_message", "propose_contract", "pay", "propose_loan"}  # star-comms checked actions
# meaningless when the agent is alone in the economy: nobody to talk to, hire or pay
_MULTI_AGENT_ONLY = {"send_message", "read_chat", "propose_contract", "accept_contract",
                     "reject_contract", "counter_offer", "cancel_contract", "set_price",
                     "pay", "list_agents", "propose_loan", "accept_loan", "repay_loan"}


_CONTRACT_ID_RE = re.compile(r"c\d{4}")


def _is_contract_target(target: str) -> bool:
    """Contract ids are the ONE reserved namespace for deliver_work targets;
    everything else (short task id or its sentence) addresses the WORLD.
    Matched by id SHAPE (c#### ), not by a leading letter -- a task sentence
    that happens to start with "c" (e.g. "compare...", "count...") must still
    route to the WORLD."""
    return bool(_CONTRACT_ID_RE.fullmatch(str(target).strip()))


def classify(name: str, inp: dict) -> str:
    """"solving" (answer-related) vs "admin" (coordination). EVERY action bills
    its turn's tokens (see agent.take_turn) -- this only labels *what kind* of
    work the tokens paid for, for recorder tallies / coordination_overhead."""
    if name in ("retrieve", "work_on", "decompose", "recall_solutions"):
        return "solving"
    if name == "deliver_work" and not _is_contract_target(inp.get("target_id", "")):
        return "solving"
    return "admin"


def visible_tools(level: LevelConfig, agent_id: str) -> list[dict]:
    is_iface = agent_id == "interface"
    out = []
    for name, spec in ACTION_SPECS.items():
        if level.n_agents == 1 and name in _MULTI_AGENT_ONLY:
            continue  # solo agent: never bill it for schemas it can never use
        if level.world_access == "interface" and not is_iface and name in _WORLD_ACTIONS:
            continue
        if name == "counter_offer" and level.central_pricing:
            continue  # bargaining disabled for everyone
        if name == "set_price" and not (level.central_pricing and is_iface):
            continue
        out.append({"name": name, **spec})
    return out


def _unknown_agent(infra: Infra, to: str) -> str:
    return f"ERROR: unknown agent {to}; valid agents: {', '.join(infra.agent_ids)}"


def permission_error(infra: Infra, agent_id: str, name: str, inp: dict) -> str | None:
    level = infra.cfg.level
    is_iface = agent_id == "interface"
    # world access (incl. deliver to WORLD)
    world_call = name in _WORLD_ACTIONS or (
        name == "deliver_work" and not _is_contract_target(inp.get("target_id", "")))
    if world_call and level.world_access == "interface" and not is_iface:
        return "only the interface agent may interact with the task board"
    # retrieval is infrastructure: every agent may query the corpus at every
    # configuration (info centralization was deleted in v3).
    # credit centralization: the interface is the sole lender
    if level.central_credit and name == "propose_loan":
        if is_iface:
            return "the interface agent is the sole lender; it cannot borrow"
        if inp.get("to") != "interface":
            return "at this configuration you may only borrow from the interface agent"
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
    # bankruptcy freezes SOLVING actions only; admin actions (incl. borrowing
    # and coordination) remain available -- and still bill, so debt deepens.
    if classify(name, inp) == "solving" and infra.ledger.is_bankrupt(agent_id):
        return "bankrupt: answer-related actions are frozen; you may still coordinate or borrow"
    return None


def dispatch(infra: Infra, agent_id: str, name: str, inp: dict) -> str:
    try:
        return _HANDLERS[name](infra, agent_id, inp)
    except (BoardError, TreeError, ContractError, LoanError, InsufficientFunds,
            KeyError, ValueError, IndexError) as e:
        return f"ERROR: {e}"


# ---------------- handlers ----------------

def _h_retrieve(infra, a, inp):
    hits = infra.retriever.search(inp["query"], k=infra.cfg.retrieve_k)
    return "\n\n".join(f"[{d['title']}] {d['text']}" for d in hits) or "(no results)"


def _h_work_on(infra, a, inp):
    infra.scratchpads[a][inp["task_id"]].append(inp["thought"])
    return f"noted on scratchpad for {inp['task_id']} ({len(infra.scratchpads[a][inp['task_id']])} entries)"


_JSON_HINT = ('content must be JSON {qid: answer}, e.g. '
              '{"q0031": "Richard Strauss", "q0032": "1911"}')


def _parse_answer_map(content: str) -> dict[str, str] | None:
    """A well-formed {qid: short answer} map, or None. Rejecting here (rather
    than inside the board) is what keeps a malformed submission from consuming
    the task's single delivery attempt."""
    try:
        data = json.loads(content)
    except (json.JSONDecodeError, TypeError):
        return None
    if not isinstance(data, dict) or not data:
        return None
    return {str(k): str(v) for k, v in data.items()}


def _task_line(infra, nid: str) -> str:
    lib = infra.library
    return (f"[{nid}] «{lib.sentence(nid)}» "
            f"({len(lib.leaves(nid))} questions, reward {lib.price(nid)})")


def _deliver_contract(infra, a, cid, content):
    c = infra.contracts.get(cid)
    answers = None
    # coverage is checked BEFORE settlement so a short deliverable cannot
    # release escrow; auth/status errors still come from contracts.deliver
    if c.node_id and c.status == "accepted" and a == c.contractor:
        wanted = infra.library.leaves(c.node_id)
        answers = _parse_answer_map(content)
        if answers is None:
            return (f"ERROR: {c.cid} is bound to {c.node_id}; {_JSON_HINT} - "
                    f"one entry for each of {', '.join(wanted)}")
        missing = [q for q in wanted if q not in answers]
        extra = sorted(set(answers) - set(wanted))
        if missing or extra:
            detail = []
            if missing:
                detail.append("missing " + ", ".join(missing))
            if extra:
                detail.append("unknown " + ", ".join(extra))
            return (f"ERROR: {c.cid} is bound to {c.node_id} and needs exactly "
                    f"one answer per leaf ({', '.join(wanted)}): {'; '.join(detail)}")
    c = infra.contracts.deliver(a, cid, content)
    # solution memory trigger 2b/3: a node-bound deliverable is a verified-shape
    # answer map, so BOTH sides learn it -- the contractor produced it and the
    # payer receives it. Ungraded: no F1 exists for internal trade.
    if c.node_id and answers:
        for qid, ans in answers.items():
            infra.solutions.record_answer(a, qid, ans)
            infra.solutions.record_answer(c.proposer, qid, ans)
    infra.chat.send(a, c.proposer, f"[deliverable for {c.cid}] {content}", infra.round)
    return f"delivered {c.cid}; escrow of {c.price} tokens released to you"


def _h_deliver_work(infra, a, inp):
    tid, content = str(inp["target_id"]).strip(), inp["content"]
    if _is_contract_target(tid):
        return _deliver_contract(infra, a, tid, content)
    # WORLD: packaged delivery of a whole task tree
    answers = _parse_answer_map(content)
    if answers is None:
        return (f"ERROR: {_JSON_HINT} - one entry for every leaf question of "
                "the task (use decompose to see them). Your delivery attempt "
                "was NOT used.")
    task = infra.board.get(tid)          # resolve once; errors list candidates
    per_leaf, total = infra.board.deliver(a, task.nid, answers)
    # solution memory trigger 2a: the WORLD graded these, so the F1 is known
    for qid, score, _pay in per_leaf:
        infra.solutions.record_answer(a, qid, answers[qid], f1=score)
    detail = "; ".join(f"{qid} F1={score:.2f} -> {pay}" for qid, score, pay in per_leaf)
    return (f"delivered {task.nid} ({len(per_leaf)} questions graded): {detail}. "
            f"Total paid: {total} tokens")


def _h_list_tasks(infra, a, inp):
    tasks = infra.board.list_open()
    if not tasks:
        return "(no open tasks)"
    top = infra.cfg.list_top_n
    offset = max(0, int(inp.get("offset") or 0))
    page = tasks[offset:offset + top]
    if not page:
        return f"(no open tasks at offset {offset}; {len(tasks)} open in total)"
    lines = [_task_line(infra, t.nid) for t in page]
    remaining = len(tasks) - (offset + len(page))
    if remaining > 0:
        lines.append(f"... and {remaining} more (call list_tasks with "
                     f"offset={offset + len(page)} to see them)")
    return "\n".join(lines)


def _h_claim_task(infra, a, inp):
    t = infra.board.claim(a, inp["task"], infra.round)
    return (f"claimed {_task_line(infra, t.nid)}. Call decompose to reveal its "
            "structure; deliver all of its questions in one JSON package.")


def _h_decompose(infra, a, inp):
    ref = str(inp["node"]).strip()
    if ref in infra.library.questions:          # a leaf: nothing left to reveal
        q = infra.library.questions[ref]
        return f"[{q.qid}] {q.text}"
    node = infra.library.resolve(ref)
    rows = infra.library.children_view(node.nid)
    # solution memory trigger 1: structure learned is structure stored
    infra.solutions.record_decomposition(
        a, node.nid, [r.get("nid") or r["qid"] for r in rows])
    lines = [f"[{node.nid}] «{node.sentence}» breaks down into {len(rows)} part(s):"]
    for r in rows:
        if r["kind"] == "subtask":
            lines.append(f"  [{r['nid']}] «{r['sentence']}» "
                         f"({r['leaf_count']} questions, reward {r['price']})")
        else:
            lines.append(f"  [{r['qid']}] {r['text']}")
    return "\n".join(lines)


def _solution_key(infra, ref: str) -> str:
    """What the agent typed -> a solution-store key. The store holds ids only,
    so sentences are resolved here (fuzzily, as everywhere an agent names a
    node it can see). A bare q-id passes straight through even if the library
    does not know it: recall must be able to say 'never solved' about it."""
    r = str(ref).strip()
    if r in infra.library.questions or is_question(r):
        return r
    return infra.library.resolve(r).nid


def _h_recall_solutions(infra, a, inp):
    key = _solution_key(infra, inp["name"])
    res = infra.solutions.recall(a, key)
    known, missing, unexpanded = res["known"], res["missing"], res["unexpanded"]
    if not known:
        return f"(no stored solutions under {key})"
    items = []
    for qid, rec in known.items():
        tag = f" (F1 {rec['f1']:.2f})" if "f1" in rec else ""
        items.append(f'"{qid}": "{rec["answer"]}"{tag}')
    parts = [f"known {len(known)}/{len(known) + len(missing)} answers under "
             f"{key}: {{{', '.join(items)}}}"]
    if missing:
        parts.append("missing: " + ", ".join(missing))
    if unexpanded:
        parts.append("not yet decomposed (may hide more leaves): "
                     + ", ".join(unexpanded))
    return "; ".join(parts)


def _h_send_message(infra, a, inp):
    if inp["to"] not in infra.agent_ids:
        return _unknown_agent(infra, inp["to"])
    infra.chat.send(a, inp["to"], inp["text"], infra.round)
    return f"sent to {inp['to']}"


def _h_read_chat(infra, a, inp):
    msgs = infra.chat.history(a, inp["with_agent"])
    return "\n".join(f"[r{m.round_no}] {m.sender}: {m.text}" for m in msgs) or "(no history)"


def _bind_node(infra, c) -> str:
    """A contract whose task text names a library node is bound to it: the
    deliverable then has to cover that node's leaves. Free text stays free.
    Binding uses ONLY exact resolution (exact node id or exact normalized
    sentence) -- fuzzy resolution stays reserved for claim/decompose/deliver,
    where the user is explicitly naming an existing node and an approximate
    match is what they intend. Here `task` is free text chosen by the
    PROPOSER, and merely resembling a node sentence is not consent to bind
    the contract's coverage requirements to it."""
    node = infra.library.resolve_exact(c.task)
    if node is None:
        return ""
    c.node_id = node.nid
    return (f" [bound to {c.node_id}: the deliverable must be a JSON map with "
            f"full leaf coverage of {c.node_id} "
            f"({len(infra.library.leaves(c.node_id))} questions)]")


def _h_propose_contract(infra, a, inp):
    if inp["to"] not in infra.agent_ids:
        return _unknown_agent(infra, inp["to"])
    central = infra.cfg.level.central_pricing
    if central and a != "interface":
        # price (if any) is ignored: the interface will set it
        c = infra.contracts.propose(a, inp["to"], inp["task"])
        bound = _bind_node(infra, c)
        infra.chat.send(a, "interface",
                        f"[contract {c.cid} awaits your pricing] {a} -> {inp['to']}: "
                        f"{c.task}{bound}", infra.round)
        if inp["to"] != "interface":
            infra.chat.send(a, inp["to"],
                            f"[contract offer {c.cid}, price pending interface] "
                            f"task: {c.task}{bound}", infra.round)
        return (f"proposed {c.cid} to {inp['to']}; awaiting interface pricing"
                + (f" (bound to {c.node_id})" if c.node_id else ""))
    if inp.get("price") is None:
        return "ERROR: price is required (bargaining configuration)"
    c = infra.contracts.propose(a, inp["to"], inp["task"], int(inp["price"]))
    bound = _bind_node(infra, c)
    infra.chat.send(a, inp["to"],
                    f"[contract offer {c.cid}] task: {c.task} | price: {c.price}{bound}",
                    infra.round)
    return (f"proposed {c.cid} to {inp['to']} at {c.price}"
            + (f" (bound to {c.node_id})" if c.node_id else ""))


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
    if inp["to"] not in infra.agent_ids:
        return _unknown_agent(infra, inp["to"])
    try:
        infra.ledger.transfer(a, inp["to"], int(inp["amount"]))
    except InsufficientFunds as e:
        return f"ERROR: {e}"
    infra.chat.send(a, inp["to"], f"[payment] {inp['amount']} tokens", infra.round)
    return f"paid {inp['amount']} to {inp['to']}"


def _h_propose_loan(infra, a, inp):
    if inp["to"] not in infra.agent_ids:
        return _unknown_agent(infra, inp["to"])
    loan = infra.loans.propose(a, inp["to"], int(inp["amount"]))
    infra.chat.send(a, inp["to"],
                    f"[loan request {loan.lid}] {a} requests {loan.principal} tokens "
                    f"at {infra.loans.rate:.0%}/round interest", infra.round)
    return (f"proposed loan {loan.lid} to {inp['to']} for {loan.principal} tokens; "
            "lender may accept or ignore")


def _h_accept_loan(infra, a, inp):
    loan = infra.loans.accept(a, inp["loan_id"])
    infra.chat.send(a, loan.borrower,
                    f"[loan {loan.lid} accepted] {loan.principal} tokens transferred to you",
                    infra.round)
    return f"accepted {loan.lid}; transferred {loan.principal} tokens to {loan.borrower}"


def _h_repay_loan(infra, a, inp):
    loan, paid = infra.loans.repay(a, inp["loan_id"], int(inp["amount"]))
    return f"repaid {paid} tokens on {loan.lid}; principal now {loan.principal} ({loan.status})"


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
    "list_tasks": _h_list_tasks, "claim_task": _h_claim_task, "decompose": _h_decompose,
    "recall_solutions": _h_recall_solutions,
    "send_message": _h_send_message, "read_chat": _h_read_chat,
    "propose_contract": _h_propose_contract, "accept_contract": _h_accept_contract,
    "reject_contract": _h_reject_contract, "counter_offer": _h_counter_offer,
    "cancel_contract": _h_cancel_contract, "set_price": _h_set_price, "pay": _h_pay,
    "propose_loan": _h_propose_loan, "accept_loan": _h_accept_loan, "repay_loan": _h_repay_loan,
    "push_goal": _h_push_goal, "pop_goal": _h_pop_goal,
    "memory_write": _h_memory_write, "memory_search": _h_memory_search,
    "check_balance": _h_check_balance, "list_agents": _h_list_agents,
}
