"""Per-agent short-term (FIFO, goal stack) and long-term memory."""
from collections import defaultdict, deque


class FifoMemory:
    """Recent-action buffer. K is the pair budget — when maxlen is reached,
    the oldest (action, result) pair is evicted. All pairs are rendered in full."""

    def __init__(self, k: int = 10):
        self.items: deque[tuple[str, str]] = deque(maxlen=k)

    def add(self, action: str, result: str) -> None:
        self.items.append((action, result))

    def render(self) -> str:
        if not self.items:
            return "(no recent actions)"
        lines = []
        for a, r in self.items:
            lines.append(f"- {a} -> {r}")
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

    def search(self, agent: str, query: str, k: int = 5) -> list[str]:
        qtok = set(query.lower().split())
        scored = []
        for entry in self._store[agent]:
            overlap = len(qtok & set(entry.lower().split()))
            if overlap > 0:
                scored.append((overlap, entry))
        scored.sort(key=lambda t: -t[0])
        return [e for _, e in scored[:k]]
