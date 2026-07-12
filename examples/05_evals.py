"""The evaluation harness: datasets, scorers, trajectory assertions.

Requires: pip install shankit[anthropic]  and  ANTHROPIC_API_KEY set.
"""

import asyncio

from pydantic import BaseModel
from shankit import Agent, tool
from shankit.evals import Case, Dataset, evaluate, exact_match, trajectory


class Verdict(BaseModel):
    label: str  # "spam" | "ham"


@tool
def keyword_check(text: str) -> str:
    """Check a message for known spam keywords."""
    hits = [w for w in ("winner", "free", "urgent") if w in text.lower()]
    return f"spam keywords found: {hits}" if hits else "no spam keywords"


classifier = Agent(
    name="spam-classifier",
    model="anthropic:claude-sonnet-4-5",
    instructions="Classify messages as spam or ham. Always run keyword_check first.",
    tools=[keyword_check],
    output_type=Verdict,
)

dataset = Dataset(
    cases=[
        Case(
            name="obvious-spam",
            input="URGENT!! You are a WINNER, claim your FREE prize",
            expected={"label": "spam"},
        ),
        Case(name="normal-mail", input="Lunch tomorrow at noon?", expected={"label": "ham"}),
    ]
)


async def main() -> None:
    report = await evaluate(
        classifier,
        dataset,
        scorers=[
            exact_match,
            trajectory.used_tool("keyword_check"),
            trajectory.max_tool_calls(2),
        ],
    )
    print(report.summary())


if __name__ == "__main__":
    asyncio.run(main())
