"""File-first definitions: the .md file produces the same Agent the
constructor would — the file is sugar.

Run from this directory: python run.py
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))  # make tools_email importable

from shankit import load_agent

agent = load_agent(Path(__file__).parent / "triage.md")


async def main() -> None:
    # {user_name} in the prompt body fills from the per-run context.
    result = await agent.run("Triage my inbox.", context={"user_name": "Ada"})
    print(result.output.model_dump_json(indent=2))


if __name__ == "__main__":
    asyncio.run(main())
