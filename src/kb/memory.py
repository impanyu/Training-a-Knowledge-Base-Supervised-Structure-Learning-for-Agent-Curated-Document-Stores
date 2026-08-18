"""Per-iteration short-term memory: pinned task block + FIFO(K=30) of
(action -> result) pairs, full results, untruncated.

Lifecycle (spec §3): reset at the start of every iteration (train AND test),
accumulate within the iteration across both phases, discarded at iteration
end. K >= N1+N2, so the in-iteration trajectory is never evicted and Phase 2
reads Phase 1's trajectory straight from the FIFO."""
from collections import deque


class IterationMemory:
    def __init__(self, k: int = 30):
        self.k = k
        self.task = ""
        self.items: deque[tuple[str, str]] = deque(maxlen=k)

    def reset(self, task: str) -> None:
        """New iteration: new task block, empty FIFO."""
        self.task = task
        self.items.clear()

    def set_task(self, task: str) -> None:
        """Phase transition within an iteration: task block replaced, FIFO kept."""
        self.task = task

    def add(self, action: str, result: str) -> None:
        self.items.append((action, result))

    def render(self, budget_line: str = "") -> str:
        parts = [self.task]
        if budget_line:
            parts.append(budget_line)
        if self.items:
            parts.append("Recent actions (oldest first):\n"
                         + "\n".join(f"- {a} -> {r}" for a, r in self.items))
        else:
            parts.append("(no actions yet)")
        return "\n\n".join(parts)
