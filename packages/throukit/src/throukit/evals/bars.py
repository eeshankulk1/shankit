"""Bars: pass/fail thresholds over a run (design §7.2)."""

from __future__ import annotations

import operator
import re
import statistics
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Optional

from .cases import CaseResult

__all__ = ["Bar", "BarResult"]

_STATS = ("rate", "mean", "median", "max", "min", "sum", "count")
_OPS = {
    ">=": operator.ge,
    "<=": operator.le,
    "==": operator.eq,
    ">": operator.gt,
    "<": operator.lt,
}
_BAR = re.compile(
    r"^\s*(?:(?P<stat>" + "|".join(_STATS) + r")\((?P<metric>[^()\s]+)\)|(?P<value>[\w./|:-]+))"
    r"\s*(?P<op>>=|<=|==|>|<)\s*(?P<threshold>-?\d+(?:\.\d+)?)\s*$"
)


@dataclass(frozen=True)
class BarResult:
    bar: str
    value: Optional[float]
    n: int
    status: str  # pass | fail | n/a


@dataclass(frozen=True)
class Bar:
    """A pass/fail threshold, written as ``Bar.parse("rate(correct) >= 0.9")``.

    ``stat(metric)`` reads per-case metrics across the run: ``rate`` is the
    share of cases whose metric is truthy (an errored case counts as a miss),
    ``mean``/``median``/``max``/``min`` skip cases without the metric, ``sum``
    adds it up, ``count`` counts truthy cases. A bare name reads a suite
    aggregate (``Suite.aggregate``). ``where`` limits a stat to cases with
    those tags. A bar with nothing to measure (its cases were filtered out)
    is reported as n/a, neither passed nor failed.
    """

    stat: str  # one of _STATS, or "value" for a suite aggregate
    metric: str
    op: str
    threshold: float
    where: tuple[tuple[str, str], ...] = ()

    @classmethod
    def parse(cls, expr: str, **where: str) -> Bar:
        m = _BAR.match(expr)
        if m is None:
            raise ValueError(
                f"bad bar {expr!r}: expected 'stat(metric) op number' (stat: {', '.join(_STATS)}) "
                "or 'aggregate op number'"
            )
        stat = m["stat"] or "value"
        metric = m["metric"] or m["value"]
        return cls(stat, metric, m["op"], float(m["threshold"]), tuple(sorted(where.items())))

    @property
    def text(self) -> str:
        lhs = self.metric if self.stat == "value" else f"{self.stat}({self.metric})"
        scope = " [" + ", ".join(f"{k}={v}" for k, v in self.where) + "]" if self.where else ""
        return f"{lhs} {self.op} {self.threshold:g}{scope}"

    def measure(
        self, results: list[CaseResult], aggregates: Mapping[str, Any]
    ) -> tuple[Optional[float], int]:
        if self.stat == "value":
            value = aggregates.get(self.metric)
            return (float(value), 1) if isinstance(value, (int, float)) else (None, 0)
        scoped = [r for r in results if all(r.tags.get(k) == v for k, v in self.where)]
        if self.stat in ("rate", "count"):
            if not scoped:
                return None, 0
            hits = sum(1 for r in scoped if r.metrics.get(self.metric))
            return (hits / len(scoped) if self.stat == "rate" else float(hits)), len(scoped)
        values = [
            float(r.metrics[self.metric])
            for r in scoped
            if isinstance(r.metrics.get(self.metric), (int, float))
        ]
        if not values:
            return None, 0
        fn = {
            "mean": statistics.fmean,
            "median": statistics.median,
            "max": max,
            "min": min,
            "sum": sum,
        }
        return float(fn[self.stat](values)), len(values)

    def check(self, results: list[CaseResult], aggregates: Mapping[str, Any]) -> BarResult:
        value, n = self.measure(results, aggregates)
        if value is None:
            return BarResult(self.text, None, 0, "n/a")
        passed = _OPS[self.op](value, self.threshold)
        return BarResult(self.text, value, n, "pass" if passed else "fail")
