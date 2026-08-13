"""WeChat-model message box: one thread per partner pair, unread counters.

Peers share a thread ({a, b} unordered); `external` threads are per-agent,
because "external" is one end of the pair. Full history is kept forever --
reading never consumes or deletes anything. `read` pages backwards from the
newest message in windows of PAGE; reading page 0 clears the reader's unread
counter for that partner (and only that).
"""
from dataclasses import dataclass

EXTERNAL = "external"   # reserved partner id: the question stream / the WORLD
PAGE = 5


@dataclass
class ChatMessage:
    sender: str
    text: str
    round_no: int
    seq: int


class ChatSystem:
    def __init__(self):
        self.threads: dict[tuple[str, str], list[ChatMessage]] = {}
        self._unread: dict[tuple[str, str], int] = {}   # (reader, partner) -> n
        self._seq = 0
        # peer-to-peer traffic only: what agents CHOSE to say to each other.
        # Stream arrivals and deliver_work's auto-append travel through the
        # external threads and are part of the task itself, not coordination.
        self.n_agent_messages = 0

    @staticmethod
    def _key(a: str, b: str) -> tuple[str, str]:
        return (a, b) if a < b else (b, a)

    def send(self, sender: str, recipient: str, text: str, round_no: int) -> int:
        self._seq += 1
        msg = ChatMessage(sender, str(text), round_no, self._seq)
        self.threads.setdefault(self._key(sender, recipient), []).append(msg)
        key = (recipient, sender)
        self._unread[key] = self._unread.get(key, 0) + 1
        if EXTERNAL not in (sender, recipient):
            self.n_agent_messages += 1
        return self._seq

    def read(self, reader: str, partner: str, page: int = 0) -> tuple[list[ChatMessage], int]:
        """Page `page` of the thread (0 = the newest PAGE messages, 1 = the
        PAGE before those, ...), chronological within the page, plus how many
        OLDER messages remain beyond it. Page 0 clears the reader's unread
        counter for this partner."""
        if page == 0:
            self._unread.pop((reader, partner), None)
        thread = self.threads.get(self._key(reader, partner), [])
        end = len(thread) - page * PAGE
        if end <= 0:
            return [], 0
        start = max(0, end - PAGE)
        return thread[start:end], start

    def unread_partners(self, reader: str) -> list[tuple[str, int]]:
        """(partner, count) for every partner with unread mail -- the
        notification list. `external` first, then peers by name."""
        out = [(p, n) for (r, p), n in self._unread.items() if r == reader and n]
        out.sort(key=lambda pn: (pn[0] != EXTERNAL, pn[0]))
        return out

    def to_state(self) -> dict:
        return {"threads": {"|".join(k): [[m.sender, m.text, m.round_no, m.seq]
                                          for m in msgs]
                            for k, msgs in sorted(self.threads.items())},
                "unread": {"|".join(k): n for k, n in sorted(self._unread.items())},
                "seq": self._seq,
                "n_agent_messages": self.n_agent_messages}

    def from_state(self, state: dict) -> None:
        self.threads = {tuple(k.split("|")): [ChatMessage(*row) for row in msgs]
                        for k, msgs in state["threads"].items()}
        self._unread = {tuple(k.split("|")): n for k, n in state["unread"].items()}
        self._seq = state["seq"]
        self.n_agent_messages = state["n_agent_messages"]
