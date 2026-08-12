"""Role handbooks with action-trajectory demos, assembled per configuration so
a demo never shows an action the agent lacks there.

v6: there is no economy. Every demo is built around the same pipeline -- list,
claim (which hands back the stored answer if one exists), memory_search the
born-in corpus, deliver ONE short answer -- plus the two ways work moves
between agents now: asking a peer over chat, and releasing a claim.

Configurations are single-factor, so a demo may only assume the ONE mechanism
that config centralizes. In particular the hub is not privileged in general --
at C5 it holds exactly one power and is an ordinary agent in every other
respect.
"""
from ca.config import LevelConfig

_SOLO_ANSWER = """
### Demo: the question pipeline (list -> claim -> search memory -> deliver)
1. list_questions
   -> "[q0107] In what year did Der Rosenkavalier premiere? (2hop)"
      ... one line per open question
2. claim_question(qid="q0107")
   -> "claimed [q0107] In what year did Der Rosenkavalier premiere? (2hop)"
      "deliver ONE short answer: deliver_work(target_id="q0107", content="<answer>")"
   If your memory already held an answer you would see it here with its F1:
      "memory: stored answer: [q0107] ... -> "1911" (F1 1.00) GOOD: deliver it as-is"
   A GOOD stored answer is an answer for free. A LOW QUALITY one is a warning:
   re-solve it.
3. memory_search(query="Der Rosenkavalier premiere")
   -> "- [Der Rosenkavalier] Der Rosenkavalier ... premiered in 1911 at the ..."
4. memory_write(content="q0107: Der Rosenkavalier premiered 1911, Dresden")
   # persists reasoning across turns - your recent-actions window is short
5. deliver_work(target_id="q0107", content="1911")
   -> `content` is the SHORT ANSWER ONLY (a name / date / phrase), a bare
      string - never a sentence. This is your ONE graded attempt on the claim.
One answer takes about 3 turns (a search, a note, a delivery).
CUT LOSSES: if a question is not converging after 2-3 searches, either deliver
your best guess (partial F1 still counts) or hand it back with
release_question so another agent can try."""

_ASK_PEER = """
### Demo: asking a peer instead of searching alone
You hold [q0107] and two searches turned up nothing. A peer may already know:
- send_message(to="{peer}", text="Do you know what year Der Rosenkavalier
  premiered? I hold q0107 and my searches are coming up empty.")
- their reply arrives in your context automatically next turn:
  "from {peer}: Der Rosenkavalier premiered in 1911 at the Dresden Hofoper"
- deliver_work(target_id="q0107", content="1911")
Ask by NAMING THE FACT YOU NEED, not the question id - a peer cannot look up a
question id, but they can search their memory for a fact. A peer's search is
one you did not have to spend a turn on."""

_ANSWER_PEER = """
### Demo: answering a peer who asks you
- Unread message: "from {asker}: Do you know Lion Air's hub airport?"
- memory_search(query="Lion Air hub airport")
  -> "- [Juanda International Airport] Juanda International Airport ... serves
      Surabaya ..."
- send_message(to="{asker}", text="Lion Air's hub airport is Juanda
  International, Surabaya")
Answering costs you one turn and saves them several. Every question is a
question for all of us, so a fact you already hold is worth more in their hands
than sitting in your memory. Reply with the FACT, not with "let me look"."""

_RELEASE = """
### Demo: handing a question back
A question holds ONE claimant, so a claim you are not going to finish blocks
everyone else:
- release_question(qid="q0107")
  -> "released q0107; it is open on the board again for any agent"
Do it as soon as you decide a question is not yours to finish."""

_RELEASE_PEER = """
Better still: tell a peer what you already know about it (send_message), then
release it, so they can claim it and finish from where you stopped."""

_IFACE_PIPELINE = """
### Demo: your pipeline
1. list_questions -> pick a question
2. claim_question(qid="q0107")   # hands back your stored answer, if any
3. EITHER solve it yourself (memory_search, then reason),
   OR ask a peer who may already know:
   send_message(to="agent_3", text="What year did Der Rosenkavalier premiere?")
4. deliver_work(target_id="q0107", content="1911")
   -> ONE short answer, ONE graded attempt.
Hold several claims and keep several agents looking up different facts at
once; release any claim you are not going to finish."""

# Only true when the hub holds the board monopoly (C1): there it is the
# system's single channel to the WORLD, so its turns are the scarcest resource.
# At C5 every agent delivers directly, so this must NOT appear.
_IFACE_BOTTLENECK = """
YOUR TURN IS THE SCARCEST RESOURCE IN THE SYSTEM: you are the only agent who
can claim questions and deliver to the WORLD, so never spend a turn on
greetings or per-worker small talk. Priority every turn:
(1) deliver answers you already have (the only way anything is scored),
(2) read what workers sent you and turn it into deliveries,
(3) ask workers about the questions still open on your claims - ask for the
    FACT, quoting the question text, since they cannot see the board,
(4) claim new questions only when the current ones are nearly done.
Ask several workers about different questions in parallel; they cannot start
without a question from you."""

_MEMORY = """
### Demo: your memory was born knowing the corpus
Your long-term memory starts pre-loaded with the WORLD's whole knowledge
corpus (~12k encyclopedia paragraphs). memory_search is the ONE way to look
anything up, and it searches everything at once - corpus paragraphs, your own
notes, and every answer you have delivered:
- memory_search(query="Lion Air hub airport")
  -> "- [Juanda International Airport] Juanda International Airport ... serves
      Surabaya ..."          (a corpus paragraph)
     "- Lion Air's hub airport = Juanda International, Surabaya"  (your note)
Every answer you deliver is written back automatically, and claiming a
question hands its stored answer straight back with its F1, so a question
answered once stays answered.
ALSO WRITE DOWN WHAT YOU LEARN ON THE WAY, not just final answers:
- memory_write(content="Lion Air's hub airport = Juanda International, Surabaya")
- memory_write(content="Der Rosenkavalier premiered 1911, Dresden Hofoper")
Multi-hop questions need TWO facts chained: search for the first, note the
bridge, search for the second. These intermediate findings come back by
MEANING on your next search, so a fact you noted while answering one question
answers the next one for free."""

_MEMORY_SHARED = """
Memory is SHARED at this configuration: every agent writes into the same store
and reads the same store. Answers your peers deliver come back to you on
claim_question exactly as your own do, and the intermediate findings you write
down become a COLLECTIVE asset - your note about Juanda International is in
your peers' next search too. Claim first, look at what is already known, and
only then start searching."""


def role_skill(level: LevelConfig, agent_id: str) -> str:
    is_iface = agent_id == "hub"
    can_world = level.world_access == "all" or is_iface
    solo = level.n_agents == 1
    blocks: list[str] = []
    if is_iface:
        blocks.append(_IFACE_PIPELINE +
                      (_IFACE_BOTTLENECK if level.world_access == "hub" else ""))
    elif can_world:
        blocks.append(_SOLO_ANSWER)
    if not solo:
        # under star comms the only counterparty a worker has is the hub; under
        # the board monopoly the hub is the only agent with questions to ask about
        peer = ("agent_3" if is_iface else
                ("hub" if level.star_comms or level.world_access == "hub" else "agent_5"))
        if can_world:
            blocks.append(_ASK_PEER.format(peer=peer))
        blocks.append(_ANSWER_PEER.format(asker=peer))
    if can_world:
        blocks.append(_RELEASE + ("" if solo else _RELEASE_PEER))
    # memory exists at every configuration and for every role
    blocks.append(_MEMORY + (_MEMORY_SHARED if level.shared_memory else ""))
    return "\n\n## ROLE HANDBOOK (worked examples)\n" + "\n".join(blocks)
