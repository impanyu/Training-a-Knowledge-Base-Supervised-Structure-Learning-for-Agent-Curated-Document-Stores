"""Point-to-point messaging with per-agent unread cursors."""
from collections import defaultdict
from dataclasses import dataclass


@dataclass
class ChatMessage:
    mid: int
    sender: str
    recipient: str
    text: str
    round_no: int


class ChatSystem:
    def __init__(self):
        self.messages: list[ChatMessage] = []
        self._cursor: dict[str, int] = defaultdict(int)  # agent -> index read up to

    def send(self, sender: str, recipient: str, text: str, round_no: int) -> int:
        mid = len(self.messages)
        self.messages.append(ChatMessage(mid, sender, recipient, text, round_no))
        return mid

    def unread(self, agent: str) -> list[ChatMessage]:
        return [m for m in self.messages[self._cursor[agent]:] if m.recipient == agent]

    def mark_read(self, agent: str) -> None:
        self._cursor[agent] = len(self.messages)

    def history(self, a: str, b: str, limit: int = 20) -> list[ChatMessage]:
        pair = [m for m in self.messages
                if {m.sender, m.recipient} == {a, b}]
        return pair[-limit:]
