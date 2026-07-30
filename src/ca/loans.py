"""Peer-to-peer loans: propose/accept/repay + per-round interest with capitalization."""
from dataclasses import dataclass

from ca.economy import InsufficientFunds, Ledger


class LoanError(Exception):
    pass


@dataclass
class Loan:
    lid: str
    lender: str
    borrower: str
    principal: int
    status: str = "proposed"  # proposed / active / repaid / cancelled


class LoanSystem:
    def __init__(self, ledger: Ledger, rate: float = 0.01):
        self.ledger = ledger
        self.rate = rate
        self.loans: dict[str, Loan] = {}
        self._n = 0

    def _next_id(self) -> str:
        self._n += 1
        return f"n{self._n:04d}"  # "n" for note; must not collide with q/c/t prefixes

    def get(self, lid: str) -> Loan:
        if lid not in self.loans:
            raise LoanError(f"unknown loan {lid}")
        return self.loans[lid]

    def propose(self, borrower: str, lender: str, amount: int) -> Loan:
        if borrower == lender:
            raise LoanError("cannot loan to yourself")
        if amount <= 0:
            raise LoanError("amount must be positive")
        loan = Loan(self._next_id(), lender, borrower, int(amount))
        self.loans[loan.lid] = loan
        return loan

    def _require(self, loan: Loan, status: str, agent: str):
        if loan.status != status:
            raise LoanError(f"{loan.lid} is {loan.status}, not {status}")
        if agent not in (loan.lender, loan.borrower):
            raise LoanError("not a party to this loan")

    def accept(self, agent: str, lid: str) -> Loan:
        loan = self.get(lid)
        if loan.status != "proposed":
            raise LoanError(f"{loan.lid} is {loan.status}, not proposed")
        if agent != loan.lender:
            raise LoanError(f"only the lender ({loan.lender}) may accept {loan.lid}")
        try:
            self.ledger.transfer(loan.lender, loan.borrower, loan.principal)
        except InsufficientFunds as e:
            raise LoanError(f"lender cannot fund loan: {e}") from e
        loan.status = "active"
        return loan

    def reject(self, agent: str, lid: str) -> Loan:
        loan = self.get(lid)
        if loan.status != "proposed":
            raise LoanError(f"{loan.lid} is {loan.status}, not proposed")
        if agent != loan.lender:
            raise LoanError(f"only the lender ({loan.lender}) may reject {loan.lid}")
        loan.status = "cancelled"
        return loan

    def cancel(self, agent: str, lid: str) -> Loan:
        loan = self.get(lid)
        if loan.status != "proposed":
            raise LoanError(f"{loan.lid} is {loan.status}, not proposed")
        if agent != loan.borrower:
            raise LoanError(f"only the borrower ({loan.borrower}) may cancel {loan.lid}")
        loan.status = "cancelled"
        return loan

    def repay(self, agent: str, lid: str, amount: int) -> tuple[Loan, int]:
        loan = self.get(lid)
        if loan.status != "active":
            raise LoanError(f"{loan.lid} is {loan.status}, not active")
        if agent != loan.borrower:
            raise LoanError(f"only the borrower ({loan.borrower}) may repay {loan.lid}")
        if amount <= 0:
            raise LoanError("amount must be positive")
        paid = min(int(amount), loan.principal)
        self.ledger.transfer(loan.borrower, loan.lender, paid)
        loan.principal -= paid
        if loan.principal == 0:
            loan.status = "repaid"
        return loan, paid

    def interest_tick(self) -> list[dict]:
        events = []
        for loan in self.loans.values():
            if loan.status != "active":
                continue
            interest = max(1, round(loan.principal * self.rate))
            paid = self.ledger.balance(loan.borrower) >= interest
            if paid:
                self.ledger.transfer(loan.borrower, loan.lender, interest)
            else:
                loan.principal += interest
            events.append({
                "lid": loan.lid, "borrower": loan.borrower, "lender": loan.lender,
                "interest": interest, "paid": paid, "principal_after": loan.principal,
            })
        return events

    def pending_for(self, agent: str) -> list[Loan]:
        out = []
        for loan in self.loans.values():
            if loan.status == "proposed" and loan.lender == agent:
                out.append(loan)
            elif loan.status == "active" and agent in (loan.borrower, loan.lender):
                out.append(loan)
        return out
