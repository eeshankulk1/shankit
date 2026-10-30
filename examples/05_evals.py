"""The evaluation harness: cases, scorers, trajectory checks, pass bars and a
spend cap.

Requires: pip install throukit[anthropic]  and  ANTHROPIC_API_KEY set.
"""

import asyncio
from decimal import Decimal

from pydantic import BaseModel
from throukit import Agent, Usage, tool
from throukit.evals import Case, Price, evaluate, exact_match, trajectory


class Verdict(BaseModel):
    label: str  # "spam" | "ham"


@tool
def keyword_check(text: str) -> str:
    """Check a message for known spam keywords."""
    hits = [w for w in ("winner", "free", "urgent") if w in text.lower()]
    return f"spam keywords found: {hits}" if hits else "no spam keywords"


classifier = Agent(
    name="spam-classifier",
    model="anthropic:claude-sonnet-5-5",
    instructions="Classify messages as spam or ham. Always run keyword_check first.",
    tools=[keyword_check],
    output_type=Verdict,
)

cases = [
    Case(
        id="obvious-spam",
        input="URGENT!! You are a WINNER, claim your FREE prize",
        expected={"label": "spam"},
    ),
    Case(id="normal-mail", input="Lunch tomorrow at noon?", expected={"label": "ham"}),
]


def price(model: str, usage: Usage) -> Price:
    """Your own rates, in dollars: the framework ships no price list."""
    per_token = Decimal("0.000003"), Decimal("0.000015")
    return Price(usage.input_tokens * per_token[0] + usage.output_tokens * per_token[1])


async def main() -> None:
    report = await evaluate(
        classifier,
        cases,
        scorers=[
            exact_match,
            trajectory.used_tool("keyword_check"),
            trajectory.max_tool_calls(2),
        ],
        bars=["rate(pass) >= 0.9"],
        trials=2,
        budget_usd=0.25,
        price=price,
    )
    print(report.summary())


if __name__ == "__main__":
    asyncio.run(main())
