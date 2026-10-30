"""Spend metering and the hard budget cap (design §7.2).

Paid evals call real models, so a paid run happens inside ``metered(meter)``,
which wraps every provider in the registry: every ``Agent`` that names its
model as ``"provider:model"`` goes through it, so a suite cannot forget to
report what it spent. A client handed straight to an agent skips the registry;
wrap it with ``Meter.wrap``.

Calls are priced from their reported usage by the ``price`` you give the meter.
The framework ships no price list: rates change, and only you know yours.

The cap: before each call the meter checks spend. Once actual spend has
reached the budget, the next call raises `BudgetExceeded` instead of going
out, so overshoot is bounded by the calls already in flight when the cap was
reached. Spend the meter cannot see (a subprocess, some other vendor) is added
with ``charge_usd``.
"""

from __future__ import annotations

import threading
from collections.abc import Callable, Iterable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any, NamedTuple, Optional, Union

from ..exceptions import ModelError
from ..models.base import ModelClient, ModelRequest
from ..models.observed import ModelCall, ObservedModel
from ..models.registry import _provider_names, get_provider, register_provider
from ..usage import Usage

__all__ = ["BudgetExceeded", "CaseSpend", "Meter", "Price", "Pricer", "metered"]

Number = Union[Decimal, float, int, str]


class Price(NamedTuple):
    """What one call cost. ``known=False`` says the model has no list price
    and the pricer guessed; price such a call high, so the cap never
    under-counts. The report names those models."""

    cost: Number
    known: bool = True


Pricer = Callable[[str, Usage], Price]


class BudgetExceeded(ModelError):
    """Raised instead of a paid call once spend has reached the budget.

    A non-retryable `ModelError`, so the agent loop passes it through as is.
    Every later call raises again, so nothing more is spent. The runner reads
    the meter to tell a budget stop from a real failure, whatever the
    exception got wrapped in on the way.
    """

    def __init__(self, message: str) -> None:
        super().__init__(message, provider="eval-budget", retryable=False)


def _usd(value: Number) -> Decimal:
    return value if isinstance(value, Decimal) else Decimal(str(value))


@dataclass(eq=False)
class CaseSpend:
    """What one case has spent, and the first call the cap refused it."""

    usd: Decimal = Decimal("0")
    refused: Optional[str] = None


_CASE: ContextVar[Optional[CaseSpend]] = ContextVar("throukit_eval_case", default=None)


@dataclass
class Meter:
    """Running spend against one budget, safe across threads.

    Args:
        budget_usd: The hard cap. ``None`` meters without a cap.
        price: Prices a model call's usage. Required with a budget.
        on_spend: Called with the total after every charge, so a run that is
            killed has still said what it spent (`SessionCap` uses this).
    """

    budget_usd: Optional[Decimal] = None
    price: Optional[Pricer] = field(default=None, repr=False)
    on_spend: Optional[Callable[[Decimal], None]] = field(default=None, repr=False)
    spent_usd: Decimal = Decimal("0")
    models: dict[str, dict[str, Any]] = field(default_factory=dict)
    vendors: dict[str, Decimal] = field(default_factory=dict)
    unpriced_models: set[str] = field(default_factory=set)
    # What the cap first refused ("anthropic:claude-haiku-5-5", "voyage"),
    # or None while the run is under budget.
    tripped: Optional[str] = None
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    _cases: list[CaseSpend] = field(default_factory=list, repr=False)

    def __post_init__(self) -> None:
        if self.budget_usd is None:
            return
        self.budget_usd = _usd(self.budget_usd)
        if not self.budget_usd.is_finite() or self.budget_usd < 0:
            raise ValueError("budget must be a finite amount >= 0")
        if self.price is None:
            raise ValueError("a budget needs price= to count spend against it")

    @property
    def remaining_usd(self) -> Optional[Decimal]:
        return None if self.budget_usd is None else self.budget_usd - self.spent_usd

    @contextmanager
    def case(self) -> Iterator[CaseSpend]:
        """Count what the enclosed case spends. A charge goes to the case
        whose context it runs in; a charge from a bare thread, which carries
        no context, goes to the case in flight when there is only one."""
        spend = CaseSpend()
        token = _CASE.set(spend)
        with self._lock:
            self._cases.append(spend)
        try:
            yield spend
        finally:
            _CASE.reset(token)
            with self._lock:
                self._cases.remove(spend)

    def _current_case(self) -> Optional[CaseSpend]:
        spend = _CASE.get()
        if spend in self._cases:
            return spend
        return self._cases[0] if len(self._cases) == 1 else None

    def guard(self, what: str) -> None:
        """Refuse a paid call once spend has reached the budget."""
        with self._lock:
            if self.budget_usd is None or self.spent_usd < self.budget_usd:
                return
            if self.tripped is None:
                self.tripped = what
            case = self._current_case()
            if case is not None and case.refused is None:
                case.refused = what
            spent, budget = self.spent_usd, self.budget_usd
        raise BudgetExceeded(f"eval budget reached (${spent:.4f} of ${budget:.2f}): refused {what}")

    def _add(self, cost: Decimal) -> Decimal:
        self.spent_usd += cost
        case = self._current_case()
        if case is not None:
            case.usd += cost
        return self.spent_usd

    def charge_usage(self, model_id: str, usage: Usage) -> Decimal:
        price = self.price(model_id, usage) if self.price else Price(0, known=False)
        cost = _usd(price.cost)
        with self._lock:
            row = self.models.setdefault(
                model_id,
                {
                    "calls": 0,
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "cache_read_tokens": 0,
                    "cache_write_tokens": 0,
                    "cost_usd": Decimal("0"),
                },
            )
            row["calls"] += 1
            row["input_tokens"] += usage.input_tokens
            row["output_tokens"] += usage.output_tokens
            row["cache_read_tokens"] += usage.cache_read_tokens
            row["cache_write_tokens"] += usage.cache_write_tokens
            row["cost_usd"] += cost
            if not price.known:
                self.unpriced_models.add(model_id)
            total = self._add(cost)
        self._report(total)
        return cost

    def charge_usd(self, amount: Number, label: str) -> Decimal:
        """Spend the meter cannot see: a vendor, a subprocess."""
        cost = _usd(amount)
        with self._lock:
            self.vendors[label] = self.vendors.get(label, Decimal("0")) + cost
            total = self._add(cost)
        self._report(total)
        return cost

    def _report(self, total: Decimal) -> None:
        if self.on_spend is not None:
            self.on_spend(total)

    def wrap(self, client: ModelClient, provider: str = "model") -> ObservedModel:
        """``client``, with the cap checked before each call and its usage
        charged after."""

        def guard(request: ModelRequest) -> None:
            self.guard(f"{provider}:{request.model}")

        def charge(call: ModelCall) -> None:
            if call.response is not None:
                self.charge_usage(call.request.model, call.response.usage)

        return ObservedModel(client, before=guard, after=charge)

    def summary(self) -> dict[str, Any]:
        with self._lock:
            budget = None if self.budget_usd is None else float(self.budget_usd)
            return {
                "budget_usd": budget,
                "spent_usd": round(float(self.spent_usd), 6),
                "models": {
                    mid: {**row, "cost_usd": round(float(row["cost_usd"]), 6)}
                    for mid, row in sorted(self.models.items())
                },
                "vendors": {k: round(float(v), 6) for k, v in sorted(self.vendors.items())},
                "unpriced_models": sorted(self.unpriced_models),
                "tripped": self.tripped,
            }

    def describe(self) -> str:
        """One human line: spend vs budget and where it went."""
        s = self.summary()
        parts = [
            f"{mid} ${row['cost_usd']:.4f} ({row['calls']} calls)"
            for mid, row in s["models"].items()
        ]
        parts += [f"{name} ${cost:.4f}" for name, cost in s["vendors"].items()]
        line = f"spent ${s['spent_usd']:.4f}"
        if s["budget_usd"] is not None:
            line += f" of ${s['budget_usd']:.2f}"
        return line + (": " + ", ".join(parts) if parts else "")


@contextmanager
def metered(meter: Meter, *, providers: Optional[Iterable[str]] = None) -> Iterator[Meter]:
    """Meter (and cap) every model call made through ``providers``, by default
    every registered provider, in this process until exit."""
    names = _provider_names() if providers is None else list(providers)
    saved = {}
    for name in names:
        factory = get_provider(name)
        if factory is not None:
            saved[name] = register_provider(name, _metered_factory(name, factory, meter))
    try:
        yield meter
    finally:
        for name, factory in saved.items():
            if factory is not None:
                register_provider(name, factory)


def _metered_factory(
    provider: str, factory: Callable[[], ModelClient], meter: Meter
) -> Callable[[], ModelClient]:
    return lambda: meter.wrap(factory(), provider)
