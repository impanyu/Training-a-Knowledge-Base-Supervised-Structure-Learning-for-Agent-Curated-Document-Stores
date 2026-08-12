"""Action registry: specs (tool schemas), permission gating, dispatch."""
import re

from ca.bank import BankError
from ca.board import BoardError
from ca.config import LevelConfig
from ca.contracts import ContractError
from ca.economy import InsufficientFunds
from ca.infra import Infra
from ca.loans import LoanError


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
                        "how you find the facts a question needs. COSTS TOKENS."),
        "input_schema": _schema({"query": _S}, ["query"]),
    },
    "deliver_work": {
        "description": ("Deliver work. target_id = a QUESTION id ('q0042') submits "
                        "your answer to the WORLD: `content` is the short answer "
                        "itself (a name / date / phrase), never a sentence or an "
                        "explanation - it is graded by token-overlap F1 against a "
                        "short gold answer and pays round(price x F1). You must hold "
                        "the claim, and delivery is your ONE graded attempt on it. "
                        "COSTS TOKENS. target_id starting with 'c' = deliver an "
                        "accepted contract (escrow released to you) - only the "
                        "CONTRACTOR (the agent hired to do the work) delivers a "
                        "contract; if you are the payer, wait for the deliverable to "
                        "arrive in your chat instead."),
        "input_schema": _schema({"target_id": _S, "content": _S}, ["target_id", "content"]),
    },
    # -------- admin (coordination) --------
    "list_questions": {
        "description": ("List open questions on the WORLD's board as "
                        "[q####] <question> (difficulty, reward R). Order is arbitrary "
                        "but stable for you (other agents see a different order, so "
                        "the reward listed next to each question is what matters, not "
                        "its position). Shows one page; pass `offset` to see further "
                        "pages."),
        "input_schema": _schema({"offset": _I}, []),
    },
    "claim_question": {
        "description": ("Claim an open question: this hands you the answer already in "
                        "your memory, if any, with its graded F1. A question holds ONE "
                        "claimant at a time."),
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
                        "when bargaining is allowed; under central pricing the hub "
                        "agent sets the price after you propose. If `task` is exactly a "
                        "question id (q0042) the contract is BOUND to that question: the "
                        "deliverable is that question's short answer. That is how you "
                        "buy an answer instead of solving it yourself. Any other text is "
                        "a free-text contract."),
        "input_schema": _schema({"to": _S, "task": _S, "price": _I}, ["to", "task"]),
    },
    "set_price": {
        "description": "HUB ONLY (central pricing): set the final price of an unpriced contract.",
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
        "description": ("Save a note to your long-term memory (answers are saved there "
                        "automatically). Notes are found again by memory_search."),
        "input_schema": _schema({"content": _S}, ["content"]),
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
_TARGETED = {"send_message", "propose_contract", "pay", "propose_loan"}  # star-comms checked actions
# meaningless when the agent is alone in the economy: nobody to talk to, hire or pay
_MULTI_AGENT_ONLY = {"send_message", "read_chat", "propose_contract", "accept_contract",
                     "reject_contract", "counter_offer", "cancel_contract", "set_price",
                     "pay", "list_agents", "propose_loan", "accept_loan", "repay_loan"}

# stable markers the recorder keys its memory tallies on
MEMORY_HIT_MARKER = "memory: stored answer"
STORED_F1_MARKER = "your stored F1"
IMPROVED_MARKER = "IMPROVED on your stored F1"

_LOW_F1 = 0.5   # below this a stored answer is advertised as worth re-solving

_CONTRACT_ID_RE = re.compile(r"c\d{4}")


def _is_contract_target(target: str) -> bool:
    """Contract ids are the ONE reserved namespace for deliver_work targets;
    everything else addresses the WORLD. Matched by id SHAPE (c####)."""
    return bool(_CONTRACT_ID_RE.fullmatch(str(target).strip()))


def classify(name: str, inp: dict) -> str:
    """"solving" (answer-related) vs "admin" (coordination). EVERY action bills
    its turn's tokens (see agent.take_turn) -- this only labels *what kind* of
    work the tokens paid for, for recorder tallies / coordination_overhead."""
    if name == "memory_search":
        return "solving"
    if name == "deliver_work" and not _is_contract_target(inp.get("target_id", "")):
        return "solving"
    return "admin"


def visible_tools(level: LevelConfig, agent_id: str) -> list[dict]:
    is_iface = agent_id == "hub"
    out = []
    for name, spec in ACTION_SPECS.items():
        if level.n_agents == 1 and name in _MULTI_AGENT_ONLY:
            continue  # solo agent: never bill it for schemas it can never use
        if level.world_access == "hub" and not is_iface and name in _WORLD_ACTIONS:
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
    is_iface = agent_id == "hub"
    # world access (incl. deliver to WORLD)
    world_call = name in _WORLD_ACTIONS or (
        name == "deliver_work" and not _is_contract_target(inp.get("target_id", "")))
    if world_call and level.world_access == "hub" and not is_iface:
        return "only the hub agent may interact with the question board"
    # credit centralization: the hub is the sole lender
    if level.central_credit and name == "propose_loan":
        if is_iface:
            return "the hub agent is the sole lender; it cannot borrow"
        if inp.get("to") != "hub":
            return "at this configuration you may only borrow from the hub agent"
    # star comms
    if level.star_comms and not is_iface:
        if name in _TARGETED and inp.get("to") != "hub":
            return "at this configuration you may only interact with the hub agent"
        if name == "read_chat" and inp.get("with_agent") != "hub":
            return "at this configuration you may only interact with the hub agent"
    # pricing centralization: hub monopolizes ALL contract pricing
    if level.central_pricing:
        if name == "counter_offer":
            return "all contract prices are set by the hub agent; bargaining is disabled"
        if name == "set_price" and not is_iface:
            return "only the hub agent may set contract prices"
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
    except (BoardError, BankError, ContractError, LoanError, InsufficientFunds,
            KeyError, ValueError, IndexError) as e:
        return f"ERROR: {e}"


# ---------------- rendering helpers ----------------

def _question_line(q) -> str:
    return f"[{q.qid}] {q.text} ({q.difficulty}, reward {q.price})"


def answer_entry(q, answer: str, f1: float | None) -> str:
    """The text an answer is STORED under: self-describing, so it is findable
    both by meaning (memory_search) and by id (memory.answer)."""
    tag = f"(F1 {f1:.2f})" if f1 is not None else "(never graded, from a contract)"
    return f'[{q.qid}] {q.text} -> "{answer}" {tag}'


def _verdict(rec: dict) -> str:
    f1 = rec["f1"]
    if f1 is None:
        return "UNVERIFIED: the WORLD never scored this one, so check it before you use it"
    if f1 < _LOW_F1:
        return (f"LOW QUALITY (F1 {f1:.2f} of 1.00): re-solve it (memory_search, then "
                "reason) instead of delivering it again")
    return (f"GOOD (F1 {f1:.2f} of 1.00): deliver it as-is and spend your tokens "
            "on unanswered questions")


# ---------------- handlers ----------------

def _deliver_contract(infra, a, cid, content):
    c = infra.contracts.deliver(a, cid, content)
    # memory trigger 2: a question-bound deliverable is an answer, so BOTH
    # sides learn it -- the contractor produced it and the payer received it.
    # Ungraded: no F1 exists for internal trade.
    if c.qid:
        q = infra.bank.get(c.qid)
        entry = answer_entry(q, content, None)
        learners = [a] if infra.memory.shared else [a, c.proposer]
        for who in learners:
            infra.memory.write(who, entry, kind="answer", qid=q.qid)
    infra.chat.send(a, c.proposer, f"[deliverable for {c.cid}] {content}", infra.round)
    return f"delivered {c.cid}; escrow of {c.price} tokens released to you"


def _h_deliver_work(infra, a, inp):
    tid, content = str(inp["target_id"]).strip(), str(inp["content"])
    if _is_contract_target(tid):
        return _deliver_contract(infra, a, tid, content)
    q = infra.bank.get(tid)                     # resolve first; errors list near ids
    prev = infra.memory.answer(a, q.qid)
    r = infra.board.deliver(a, q.qid, content, infra.round)
    # memory trigger 1: the WORLD graded this, so the F1 is known
    infra.memory.write(a, answer_entry(q, r.submitted, r.f1), kind="answer",
                       qid=r.qid, f1=r.f1)
    out = f"delivered {q.qid}: F1 {r.f1:.2f} -> {r.payout} tokens"
    if prev is not None and prev["f1"] is not None:
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
    ref = str(inp["qid"]).strip()
    if _CONTRACT_ID_RE.fullmatch(ref):          # namespace mixup seen live (C5)
        return (f"ERROR: {ref} is a contract id, not a posted question - contracts "
                "are accepted (accept_contract), not claimed.")
    q = infra.board.claim(a, ref, infra.round)
    lines = [f"claimed {_question_line(q)}",
             f'deliver ONE short answer: deliver_work(target_id="{q.qid}", '
             'content="<answer>")']
    # auto-recall: the best stored ANSWER rides along (corpus entries are
    # knowledge, not answers -- they never trigger this)
    rec = infra.memory.answer(a, q.qid)
    if rec is not None:
        lines.append(f"{MEMORY_HIT_MARKER}: {rec['text']} {_verdict(rec)}")
    return "\n".join(lines)


def _h_send_message(infra, a, inp):
    if inp["to"] not in infra.agent_ids:
        return _unknown_agent(infra, inp["to"])
    infra.chat.send(a, inp["to"], inp["text"], infra.round)
    return f"sent to {inp['to']}"


def _h_read_chat(infra, a, inp):
    msgs = infra.chat.history(a, inp["with_agent"])
    return "\n".join(f"[r{m.round_no}] {m.sender}: {m.text}" for m in msgs) or "(no history)"


def _bind_question(infra, c) -> str:
    """A contract whose task text is EXACTLY a question id is bound to it: the
    deliverable is then that question's answer, and both parties learn it. Any
    other text stays a free-text contract -- `task` is chosen by the PROPOSER,
    and merely mentioning a question is not consent to bind."""
    qid = str(c.task).strip()
    if qid not in infra.bank.questions:
        return ""
    c.qid = qid
    return (f" [bound to {c.qid}: the deliverable is the short answer to "
            f"\"{infra.bank.questions[qid].text}\"]")


def _h_propose_contract(infra, a, inp):
    if inp["to"] not in infra.agent_ids:
        return _unknown_agent(infra, inp["to"])
    central = infra.cfg.level.central_pricing
    if central and a != "hub":
        # price (if any) is ignored: the hub will set it
        c = infra.contracts.propose(a, inp["to"], inp["task"])
        bound = _bind_question(infra, c)
        infra.chat.send(a, "hub",
                        f"[contract {c.cid} awaits your pricing] {a} -> {inp['to']}: "
                        f"{c.task}{bound}", infra.round)
        if inp["to"] != "hub":
            infra.chat.send(a, inp["to"],
                            f"[contract offer {c.cid}, price pending hub] "
                            f"task: {c.task}{bound}", infra.round)
        return (f"proposed {c.cid} to {inp['to']}; awaiting hub pricing"
                + (f" (bound to {c.qid})" if c.qid else ""))
    if inp.get("price") is None:
        return "ERROR: price is required (bargaining configuration)"
    c = infra.contracts.propose(a, inp["to"], inp["task"], int(inp["price"]))
    bound = _bind_question(infra, c)
    infra.chat.send(a, inp["to"],
                    f"[contract offer {c.cid}] task: {c.task} | price: {c.price}{bound}",
                    infra.round)
    return (f"proposed {c.cid} to {inp['to']} at {c.price}"
            + (f" (bound to {c.qid})" if c.qid else ""))


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


def _h_check_balance(infra, a, inp):
    return f"balance: {infra.ledger.balance(a)} tokens"


def _h_list_agents(infra, a, inp):
    return ", ".join(infra.agent_ids)


_HANDLERS = {
    "memory_search": _h_memory_search, "deliver_work": _h_deliver_work,
    "list_questions": _h_list_questions, "claim_question": _h_claim_question,
    "send_message": _h_send_message, "read_chat": _h_read_chat,
    "propose_contract": _h_propose_contract, "accept_contract": _h_accept_contract,
    "reject_contract": _h_reject_contract, "counter_offer": _h_counter_offer,
    "cancel_contract": _h_cancel_contract, "set_price": _h_set_price, "pay": _h_pay,
    "propose_loan": _h_propose_loan, "accept_loan": _h_accept_loan, "repay_loan": _h_repay_loan,
    "push_goal": _h_push_goal, "pop_goal": _h_pop_goal,
    "memory_write": _h_memory_write,
    "check_balance": _h_check_balance, "list_agents": _h_list_agents,
}
