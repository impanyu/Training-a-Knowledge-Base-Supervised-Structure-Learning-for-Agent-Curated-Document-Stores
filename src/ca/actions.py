"""Action registry: specs (tool schemas), permission gating, dispatch."""
import json
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
    "retrieve": {
        "description": "Search the external knowledge corpus. COSTS TOKENS.",
        "input_schema": _schema({"query": _S}, ["query"]),
    },
    "deliver_work": {
        "description": ("Deliver work. target_id = a JOB id ('j0007') submits the whole "
                        "job to the WORLD: `content` must be ONE JSON object mapping "
                        "EVERY question id in that job to its short answer, e.g. "
                        '\'{\"q0042\": \"Richard Strauss\", \"q0107\": \"1911\"}\'. Each '
                        "answer is the short answer itself (a name / date / phrase), "
                        "never a sentence or an explanation - each is graded by "
                        "token-overlap F1 against a short gold answer and the job pays "
                        "the sum of price x F1. Delivery is ALL-OR-NOTHING: a map that "
                        "misses a question (or names one that is not in the job) is "
                        "rejected and you keep your claim, but a complete map is your ONE "
                        "graded attempt. COSTS TOKENS. target_id starting with 'c' = "
                        "deliver an accepted contract (escrow released to you) - only the "
                        "CONTRACTOR (the agent hired to do the work) delivers a contract; "
                        "if you are the payer, wait for the deliverable to arrive in your "
                        "chat instead."),
        "input_schema": _schema({"target_id": _S, "content": _S}, ["target_id", "content"]),
    },
    # -------- admin (coordination) --------
    "list_jobs": {
        "description": ("List open jobs on the WORLD's board as "
                        "[j####] N questions, topic k##, reward R. A job is a BUNDLE of "
                        "2-10 related questions claimed and delivered as a unit; the "
                        "reward is the sum of its questions' prices. Order is arbitrary "
                        "but stable for you (other agents see a different order, so the "
                        "reward listed next to each job is what matters, not its "
                        "position). Shows one page; pass `offset` to see further pages."),
        "input_schema": _schema({"offset": _I}, []),
    },
    "claim_job": {
        "description": ("Claim an open job: this reveals the text of every question in it "
                        "and marks the ones you already have an answer for. A job holds "
                        "ONE claimant at a time."),
        "input_schema": _schema({"jid": _S}, ["jid"]),
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
                        "split a claimed job across several agents - one contract per "
                        "question you do not want to solve yourself. Any other text is "
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
        "description": "Save a note to your long-term memory (answers are saved there automatically).",
        "input_schema": _schema({"content": _S}, ["content"]),
    },
    "memory_search": {
        "description": "Search your long-term memory by meaning (notes and past answers alike).",
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

_WORLD_ACTIONS = {"list_jobs", "claim_job"}
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
_QUESTION_ID_RE = re.compile(r"q\d{4}")

_JSON_HINT = ('`content` must be ONE flat JSON object mapping every question id in '
              'the job to its short answer, e.g. {"q0042": "Richard Strauss", '
              '"q0107": "1911"} - short answers only, no sentences, no nesting')


def _is_contract_target(target: str) -> bool:
    """Contract ids are the ONE reserved namespace for deliver_work targets;
    everything else addresses the WORLD. Matched by id SHAPE (c####)."""
    return bool(_CONTRACT_ID_RE.fullmatch(str(target).strip()))


def _parse_answer_map(content: str) -> dict[str, str]:
    """A rejected map must cost the agent nothing but the turn, so every failure
    here is raised BEFORE the board is touched and the claim survives."""
    try:
        data = json.loads(content)
    except (TypeError, json.JSONDecodeError) as e:
        raise ValueError(f"could not read `content` as JSON ({e}); {_JSON_HINT}") from e
    if not isinstance(data, dict):
        raise ValueError(f"`content` is a JSON {type(data).__name__}, not an object; "
                         f"{_JSON_HINT}")
    out = {}
    for k, v in data.items():
        if isinstance(v, (dict, list)):
            raise ValueError(f"the answer for {k} is a {type(v).__name__}, not a short "
                             f"answer string; {_JSON_HINT}")
        out[str(k).strip()] = str(v)
    return out


def classify(name: str, inp: dict) -> str:
    """"solving" (answer-related) vs "admin" (coordination). EVERY action bills
    its turn's tokens (see agent.take_turn) -- this only labels *what kind* of
    work the tokens paid for, for recorder tallies / coordination_overhead."""
    if name == "retrieve":
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
        return "only the hub agent may interact with the job board"
    # retrieval is infrastructure: every agent may query the corpus at every
    # configuration (info centralization was deleted in v3).
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

def job_topic(infra, job) -> str:
    """A job is drawn from ONE topic cluster, so any member names it."""
    return infra.bank.questions[job.qids[0]].topic


def _job_line(infra, job) -> str:
    return (f"[{job.jid}] {len(job.qids)} questions, topic {job_topic(infra, job)}, "
            f"reward {job.price}")


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
        return (f"LOW QUALITY (F1 {f1:.2f} of 1.00): re-solve it (retrieve, then reason) "
                "instead of delivering it again")
    return (f"GOOD (F1 {f1:.2f} of 1.00): deliver it as-is and spend your tokens "
            "on the unanswered ones")


# ---------------- handlers ----------------

def _h_retrieve(infra, a, inp):
    hits = infra.retriever.search(inp["query"], k=infra.cfg.retrieve_k)
    return "\n\n".join(f"[{d['title']}] {d['text']}" for d in hits) or "(no results)"


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
    if _QUESTION_ID_RE.fullmatch(tid):
        return (f"ERROR: {tid} is a question, not a posted job - the WORLD pays for "
                "whole jobs. Deliver the JOB that contains it (target_id=\"j0007\") "
                "with one JSON map covering all of its questions.")
    job = infra.bank.get_job(tid)               # resolve first; errors list near ids
    answers = _parse_answer_map(content)        # raises before the board is touched
    prev = {qid: infra.memory.answer(a, qid) for qid in job.qids}
    results = infra.board.deliver(a, job.jid, answers, infra.round)
    lines = []
    for r in results:
        q = infra.bank.questions[r.qid]
        # memory trigger 1: the WORLD graded this, so the F1 is known
        infra.memory.write(a, answer_entry(q, r.submitted, r.f1), kind="answer",
                           qid=r.qid, f1=r.f1)
        line = f"  {r.qid} F1={r.f1:.2f} -> {r.payout}"
        before = prev[r.qid]
        if before is not None and before["f1"] is not None:
            line += (f" ({IMPROVED_MARKER} {before['f1']:.2f})" if r.f1 > before["f1"]
                     else f" (no better than {STORED_F1_MARKER} {before['f1']:.2f})")
        lines.append(line)
    total = sum(r.payout for r in results)
    head = (f"delivered {job.jid}: {len(results)} questions, "
            f"total F1 {sum(r.f1 for r in results):.2f} -> {total} tokens")
    return "\n".join([head] + lines)


def _h_list_jobs(infra, a, inp):
    jobs = infra.board.open_jobs(a)
    if not jobs:
        return "(no open jobs)"
    top = infra.cfg.list_top_n
    offset = max(0, int(inp.get("offset") or 0))
    page = jobs[offset:offset + top]
    if not page:
        return f"(no open jobs at offset {offset}; {len(jobs)} open in total)"
    lines = [_job_line(infra, j) for j in page]
    remaining = len(jobs) - (offset + len(page))
    if remaining > 0:
        lines.append(f"... and {remaining} more (call list_jobs with "
                     f"offset={offset + len(page)} to see them)")
    return "\n".join(lines)


def _h_claim_job(infra, a, inp):
    ref = str(inp["jid"]).strip()
    if _CONTRACT_ID_RE.fullmatch(ref):          # namespace mixup seen live (C5)
        return (f"ERROR: {ref} is a contract id, not a posted job - contracts "
                "are accepted (accept_contract), not claimed.")
    if _QUESTION_ID_RE.fullmatch(ref):
        return (f"ERROR: {ref} is a question id; the WORLD posts JOBS (j0007), and "
                "claiming one reveals its questions. Call list_jobs.")
    job = infra.board.claim(a, ref, infra.round)
    # auto-recall, now per member: whatever is stored for each question rides
    # along, so the agent sees at a glance which ones are already paid for
    rows, n_known = [], 0
    for qid in job.qids:
        q = infra.bank.questions[qid]
        rec = infra.memory.answer(a, qid)
        if rec is None:
            rows.append(f"  — {qid} {q.text} (unanswered)")
        else:
            n_known += 1
            rows.append(f"  ✓ {qid} {q.text} -> {rec['text']} {_verdict(rec)}")
    head = [f"claimed {_job_line(infra, job)}",
            f'deliver ALL {len(job.qids)} answers in ONE map: '
            f'deliver_work(target_id="{job.jid}", content=\'{{"{job.qids[0]}": "...", ...}}\')']
    if n_known:
        head.append(f"{MEMORY_HIT_MARKER} for {n_known} of these {len(job.qids)} "
                    "questions (marked ✓ below) - that part is already paid for")
    return "\n".join(head + rows)


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
    hits = infra.memory.search(a, inp["query"])
    return "\n".join(f"- {h['text']}" for h in hits) or "(no matching memories)"


def _h_check_balance(infra, a, inp):
    return f"balance: {infra.ledger.balance(a)} tokens"


def _h_list_agents(infra, a, inp):
    return ", ".join(infra.agent_ids)


_HANDLERS = {
    "retrieve": _h_retrieve, "deliver_work": _h_deliver_work,
    "list_jobs": _h_list_jobs, "claim_job": _h_claim_job,
    "send_message": _h_send_message, "read_chat": _h_read_chat,
    "propose_contract": _h_propose_contract, "accept_contract": _h_accept_contract,
    "reject_contract": _h_reject_contract, "counter_offer": _h_counter_offer,
    "cancel_contract": _h_cancel_contract, "set_price": _h_set_price, "pay": _h_pay,
    "propose_loan": _h_propose_loan, "accept_loan": _h_accept_loan, "repay_loan": _h_repay_loan,
    "push_goal": _h_push_goal, "pop_goal": _h_pop_goal,
    "memory_write": _h_memory_write, "memory_search": _h_memory_search,
    "check_balance": _h_check_balance, "list_agents": _h_list_agents,
}
