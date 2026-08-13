"""Role handbook with action-trajectory demos, assembled per arm so a demo
never shows an action the agent lacks there.

v7: every agent is a domain expert on the same footing, so there is ONE
handbook. The proactive idle-cycle demo is the only arm difference (P0 only)
-- the same gate that adds record_qa to the catalog.
"""
from ca.config import LevelConfig

_ANSWER_EXTERNAL = """
### Demo: answering an external question end-to-end
Notification line in your context: "New messages: external (1)"
1. read_chat(with_agent="external")
   -> "[r12] external: [q0107] In what year did Der Rosenkavalier premiere?"
2. push_goal(note="q0107: year Der Rosenkavalier premiered")
3. memory_search(query="Der Rosenkavalier premiere")
   -> "- [Der Rosenkavalier] Der Rosenkavalier ... premiered in 1911 at the ..."
4. deliver_work(target_id="q0107", content="1911")
   -> `content` is the SHORT ANSWER ONLY (a name / date / phrase), a bare
      string - never a sentence. ONE graded attempt; the answer is appended
      to your `external` thread automatically.
5. pop_goal()
The question arrived while you were doing something else - that is normal.
The notification line tells you WHO has new mail, never what it says: read the
thread before anything else. CUT LOSSES: if a question is not converging
after 2-3 searches, deliver your best guess - partial F1 still counts."""

_PROACTIVE_CYCLE = """
### Demo: a proactive idle cycle (no external question waiting)
Your `external` thread shows what your domain has been asked so far; invent
the question most likely to arrive next:
1. push_goal(note="self: which Strauss opera premiered in Dresden in 1905?")
2. memory_search(query="Strauss opera Dresden premiere 1905")
   -> "- [Salome] Salome ... premiered at the Dresden Hofoper in 1905 ..."
3. record_qa(question="Which Strauss opera premiered in Dresden in 1905?",
   answer="Salome")
   -> "recorded to the shared knowledge base"
4. pop_goal()
When the real question arrives - to you or to a peer - the banked Q&A comes
back on the very first memory_search. An idle turn spent this way is an
answer banked in advance."""

_ASK_PEER = """
### Demo: asking a peer whose domain borders yours
You hold [q0311], which needs a fact from {peer}'s domain, and two searches
turned up nothing:
- send_message(to="{peer}", text="Where was the soprano Marie Wittich born?
  I need it for an external question.")
- next turn the notification line shows "New messages: {peer} (1)":
  read_chat(with_agent="{peer}")
  -> "[r15] {peer}: Marie Wittich was born in Giessen."
- deliver_work(target_id="q0311", content="Giessen")
Ask by NAMING THE FACT YOU NEED, not the question id - a peer cannot see your
`external` thread, but they can search the knowledge base for a fact. When a
peer asks YOU, answer with the fact, not with "let me look"."""

_OLD_PAGE = """
### Demo: reading an older page of a long thread
read_chat shows the newest 5 messages of a thread (page 0). To see what your
domain was asked earlier:
- read_chat(with_agent="external", page=1)
  -> the 5 messages before the newest 5 (page=2 is older still; the result
     says how many older messages remain)
Only page 0 clears your unread counter; older pages are pure reading."""

_MEMORY = """
### Demo: the shared knowledge base was born knowing the corpus
The cluster's ONE long-term memory starts pre-loaded with the WORLD's whole
knowledge corpus (~12k encyclopedia paragraphs). memory_search is the ONE way
to look anything up, and it searches everything at once - corpus paragraphs,
every agent's notes, and every delivered answer:
- memory_search(query="Lion Air main airport")
  -> "- [Juanda International Airport] Juanda International Airport ... serves
      Surabaya ..."          (a corpus paragraph)
     "- Lion Air's main airport = Juanda International, Surabaya"  (a note)
ALSO WRITE DOWN WHAT YOU LEARN ON THE WAY, not just final answers:
- memory_write(content="Lion Air's main airport = Juanda International, Surabaya")
Multi-hop questions need TWO facts chained: search for the first, note the
bridge, search for the second. Everything anyone writes is in EVERYONE's next
search - a fact you note while answering one question answers a peer's
question for free."""


def role_skill(level: LevelConfig, agent_id: str) -> str:
    peer = "agent_3" if agent_id != "agent_3" else "agent_2"
    blocks = [_ANSWER_EXTERNAL]
    if level.proactive:
        blocks.append(_PROACTIVE_CYCLE)
    blocks += [_ASK_PEER.format(peer=peer), _OLD_PAGE, _MEMORY]
    return "\n\n## ROLE HANDBOOK (worked examples)\n" + "\n".join(blocks)
