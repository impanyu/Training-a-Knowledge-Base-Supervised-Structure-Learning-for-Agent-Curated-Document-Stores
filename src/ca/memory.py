"""Per-agent short-term (FIFO, goal stack) and long-term memory."""
from collections import defaultdict, deque


class FifoMemory:
    """Recent-action buffer. Results are stored in FULL; only the *rendering*
    is budgeted, so a large retrieve payload stays usable while it is fresh
    and shrinks to a reminder once it ages out of the recent window."""

    def __init__(self, k: int = 10, cap_recent: int = 4000, cap_old: int = 300,
                 recent_n: int = 2):
        self.items: deque[tuple[str, str]] = deque(maxlen=k)
        self.cap_recent = cap_recent
        self.cap_old = cap_old
        self.recent_n = recent_n

    def add(self, action: str, result: str) -> None:
        self.items.append((action, result))

    def render(self) -> str:
        if not self.items:
            return "(no recent actions)"
        n = len(self.items)
        lines = []
        for i, (a, r) in enumerate(self.items):
            cap = self.cap_recent if i >= n - self.recent_n else self.cap_old
            lines.append(f"- {a} -> {r[:cap]}")
        return "\n".join(lines)


class GoalStack:
    def __init__(self, root: str):
        self._root = root
        self._stack: list[str] = []

    def push(self, note: str) -> None:
        self._stack.append(note)

    def pop(self) -> str:
        if not self._stack:
            raise IndexError("cannot pop the root goal")
        return self._stack.pop()

    def render(self) -> str:
        lines = [f"[0] {self._root} (root, permanent)"]
        lines += [f"[{i+1}] {n}" for i, n in enumerate(self._stack)]
        lines[-1] += "   <- current focus"
        return "\n".join(lines)


class LongTermMemory:
    def __init__(self):
        self._store: dict[str, list[str]] = defaultdict(list)

    def write(self, agent: str, content: str) -> None:
        self._store[agent].append(content)

    def search(self, agent: str, query: str, k: int = 3) -> list[str]:
        qtok = set(query.lower().split())
        scored = []
        for entry in self._store[agent]:
            overlap = len(qtok & set(entry.lower().split()))
            if overlap > 0:
                scored.append((overlap, entry))
        scored.sort(key=lambda t: -t[0])
        return [e for _, e in scored[:k]]
