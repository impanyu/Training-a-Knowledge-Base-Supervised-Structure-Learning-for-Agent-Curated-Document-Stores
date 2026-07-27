"""Internal subcontracts: propose/counter/accept(escrow)/deliver(atomic settle)."""
from dataclasses import dataclass, field

from ca.economy import InsufficientFunds, Ledger


class ContractError(Exception):
    pass


@dataclass
class Contract:
    cid: str
    proposer: str      # payer (发包方)
    contractor: str    # worker (承包方)
    task: str
    price: int
    status: str = "proposed"
    awaiting: str = ""          # who must respond while status == proposed
    deliverable: str | None = None


class ContractSystem:
    def __init__(self, ledger: Ledger):
        self.ledger = ledger
        self.contracts: dict[str, Contract] = {}
        self._n = 0

    def _next_id(self) -> str:
        self._n += 1
        return f"c{self._n:04d}"

    def get(self, cid: str) -> Contract:
        if cid not in self.contracts:
            raise ContractError(f"unknown contract {cid}")
        return self.contracts[cid]

    def propose(self, proposer: str, contractor: str, task: str,
                price: int | None = None) -> Contract:
        if proposer == contractor:
            raise ContractError("cannot contract with yourself")
        if price is None:  # central pricing: interface will set the price
            c = Contract(self._next_id(), proposer, contractor, task, 0,
                         status="unpriced", awaiting="")
        else:
            if price <= 0:
                raise ContractError("price must be positive")
            c = Contract(self._next_id(), proposer, contractor, task, price,
                         awaiting=contractor)
        self.contracts[c.cid] = c
        return c

    def set_price(self, cid: str, price: int) -> Contract:
        c = self.get(cid)
        if c.status != "unpriced":
            raise ContractError(f"{c.cid} is {c.status}, not unpriced")
        if price <= 0:
            raise ContractError("price must be positive")
        c.price = price
        c.status = "proposed"
        c.awaiting = c.contractor
        return c

    def unpriced(self) -> list[Contract]:
        return [c for c in self.contracts.values() if c.status == "unpriced"]

    def _require(self, c: Contract, status: str, agent: str | None = None):
        if c.status != status:
            raise ContractError(f"{c.cid} is {c.status}, not {status}")
        if agent is not None and c.awaiting != agent:
            raise ContractError(f"{c.cid} awaits {c.awaiting}, not {agent}")

    def counter(self, agent: str, cid: str, price: int) -> Contract:
        c = self.get(cid)
        self._require(c, "proposed", agent)
        if price <= 0:
            raise ContractError("price must be positive")
        c.price = price
        c.awaiting = c.proposer if agent == c.contractor else c.contractor
        return c

    def accept(self, agent: str, cid: str) -> Contract:
        c = self.get(cid)
        self._require(c, "proposed", agent)
        try:
            self.ledger.lock_escrow(c.cid, c.proposer, c.price)
        except InsufficientFunds as e:
            raise ContractError(f"payer cannot fund escrow: {e}") from e
        c.status = "accepted"
        c.awaiting = ""
        return c

    def reject(self, agent: str, cid: str) -> Contract:
        c = self.get(cid)
        self._require(c, "proposed", agent)
        c.status = "rejected"
        c.awaiting = ""
        return c

    def cancel(self, agent: str, cid: str) -> Contract:
        c = self.get(cid)
        if agent not in (c.proposer, c.contractor):
            raise ContractError("not a party to this contract")
        if c.status == "accepted":
            self.ledger.refund_escrow(c.cid, c.proposer)
        elif c.status not in ("proposed", "unpriced"):
            raise ContractError(f"cannot cancel a {c.status} contract")
        c.status = "cancelled"
        c.awaiting = ""
        return c

    def deliver(self, agent: str, cid: str, content: str) -> Contract:
        c = self.get(cid)
        if c.status != "accepted":
            raise ContractError(f"{c.cid} is {c.status}, not accepted")
        if agent != c.contractor:
            raise ContractError("only the contractor may deliver")
        # atomic: payment + deliverable recorded together
        self.ledger.release_escrow(c.cid, c.contractor)
        c.deliverable = content
        c.status = "delivered"
        return c

    def pending_for(self, agent: str) -> list[Contract]:
        out = []
        for c in self.contracts.values():
            if c.status == "proposed" and c.awaiting == agent:
                out.append(c)
            elif c.status == "accepted" and c.contractor == agent:
                out.append(c)
        return out
