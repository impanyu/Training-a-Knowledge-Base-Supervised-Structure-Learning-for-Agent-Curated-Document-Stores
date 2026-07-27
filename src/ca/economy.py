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
        if amount <= 0:
            raise ValueError("transfer amount must be positive")
        if self.balances[frm] < amount:
            raise InsufficientFunds(f"{frm} has {self.balances[frm]} < {amount}")
        self.balances[frm] -= amount
        self.balances[to] += amount

    def lock_escrow(self, key: str, frm: str, amount: int) -> None:
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
