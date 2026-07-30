"""Token ledger: balances, burn (LLM cost), mint (WORLD rewards), escrow."""


class InsufficientFunds(Exception):
    pass


class Ledger:
    def __init__(self, seed_capital: dict[str, int]):
        self.balances: dict[str, int] = dict(seed_capital)
        self.seed_total = sum(seed_capital.values())
        self.minted = 0
        self.burned = 0
        self.escrow: dict[str, int] = {}

    def balance(self, agent: str) -> int:
        return self.balances[agent]

    def is_bankrupt(self, agent: str) -> bool:
        return self.balances[agent] <= 0

    def burn(self, agent: str, amount: int) -> None:
        if amount < 0:
            raise ValueError("negative burn")
        self.balances[agent] -= amount  # may go negative -> bankruptcy
        self.burned += amount

    def mint(self, agent: str, amount: int) -> None:
        if amount < 0:
            raise ValueError("negative mint")
        self.balances[agent] += amount
        self.minted += amount

    def transfer(self, frm: str, to: str, amount: int) -> None:
        # validate BOTH keys before any mutation: a missing recipient must never
        # leave the payer debited (that would destroy tokens, breaking conservation)
        if frm not in self.balances or to not in self.balances:
            raise ValueError(f"unknown agent in transfer {frm} -> {to}")
        if amount <= 0:
            raise ValueError("transfer amount must be positive")
        if self.balances[frm] < amount:
            raise InsufficientFunds(f"{frm} has {self.balances[frm]} < {amount}")
        self.balances[frm] -= amount
        self.balances[to] += amount

    def lock_escrow(self, key: str, frm: str, amount: int) -> None:
        if frm not in self.balances:
            raise ValueError(f"unknown agent {frm}")
        if amount <= 0:
            raise ValueError("escrow amount must be positive")
        if self.balances[frm] < amount:
            raise InsufficientFunds(f"{frm} has {self.balances[frm]} < {amount}")
        self.balances[frm] -= amount
        self.escrow[key] = amount

    def release_escrow(self, key: str, to: str) -> None:
        self.balances[to] += self.escrow.pop(key)

    def refund_escrow(self, key: str, frm: str) -> None:
        self.balances[frm] += self.escrow.pop(key)

    def conservation_ok(self) -> bool:
        total = sum(self.balances.values()) + sum(self.escrow.values())
        return total == self.seed_total + self.minted - self.burned

    def to_state(self) -> dict:
        return {"balances": dict(self.balances), "seed_total": self.seed_total,
                "minted": self.minted, "burned": self.burned,
                "escrow": dict(self.escrow)}

    def from_state(self, state: dict) -> None:
        self.balances = dict(state["balances"])
        self.seed_total = state["seed_total"]
        self.minted = state["minted"]
        self.burned = state["burned"]
        self.escrow = dict(state["escrow"])
