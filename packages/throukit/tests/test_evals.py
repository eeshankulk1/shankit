from conftest import FakeModel, final_result_response, text_response, tool_call_response
from pydantic import BaseModel
from throukit import Agent, tool
from throukit.evals import Case, evaluate, exact_match, load_cases, output_contains, trajectory
from throukit.evals.scorers import Score, llm_judge


class Verdict(BaseModel):
    label: str


@tool
def classify(text: str) -> str:
    return "spam" if "win" in text else "ham"


def scripted_agent(responses):
    return Agent(name="clf", model="fake", model_client=FakeModel(responses), tools=[classify])


async def test_evaluate_agent_with_scorers():
    agent = scripted_agent(
        [
            tool_call_response("classify", {"text": "win money"}),
            final_result_response({"label": "spam"}),
        ]
    )
    report = await evaluate(
        agent,
        [Case(id="spammy", input="win money", expected={"label": "spam"})],
        scorers=[exact_match, trajectory.used_tool("classify"), trajectory.no_tool_errors],
        output_type=Verdict,
    )
    assert report.pass_rate == 1.0
    assert report.mean("exact_match") == 1.0
    assert "PASS" in report.summary()


async def test_failing_and_erroring_cases():
    async def target(case):
        if case.id == "boom":
            raise RuntimeError("model down")
        agent = scripted_agent([text_response("looks like ham")])
        return await agent.run(case.input, output_type=str)

    cases = [Case(id="boom", input="x"), Case(id="wrong", input="y", expected="spam")]
    report = await evaluate(target, cases, scorers=[output_contains()])
    assert report.pass_rate == 0.0
    statuses = {r.case_id: (r.error, r.passed) for r in report.results}
    assert statuses["boom"][0] is not None
    assert statuses["wrong"] == (None, False)


def test_output_contains_scores_answer_not_transcript():
    """An agent that merely *mentions* the expected string in an interim
    pass must not pass; the scorer targets the deliverable."""
    from throukit import RunResult, Usage

    result = RunResult(
        output="ham",
        text="I'll check whether this is spam.\n\nham",
        usage=Usage(),
    )
    score = output_contains("spam")(Case(id="c", input="x"), result)
    assert not score.passed
    assert output_contains("ham")(Case(id="c", input="x"), result).passed


async def test_custom_scorer_shapes():
    agent = scripted_agent([text_response("hello")])

    def boolean_scorer(case, result):
        return True

    async def float_scorer(case, result):
        return 0.5

    def full_scorer(case, result):
        return Score(name="custom", value=0.9, passed=True, reason="looked good")

    report = await evaluate(
        agent,
        [Case(id="c", input="x")],
        scorers=[boolean_scorer, float_scorer, full_scorer],
        output_type=str,
    )
    result = report.results[0]
    assert result.metrics["boolean_scorer"] is True
    assert result.metrics["float_scorer"] == 0.5
    assert result.metrics["custom"] is True
    assert result.metrics["custom_score"] == 0.9
    assert result.details["scores"][2]["reason"] == "looked good"


async def test_trajectory_assertions():
    agent = scripted_agent(
        [
            tool_call_response("classify", {"text": "a"}, call_id="t1"),
            tool_call_response("classify", {"text": "b"}, call_id="t2"),
            text_response("done"),
        ]
    )
    result = await agent.run("go", output_type=str)
    case = Case(id="c", input="go")

    assert trajectory.used_tool("classify")(case, result).passed
    assert not trajectory.did_not_use_tool("classify")(case, result).passed
    assert trajectory.max_tool_calls(2)(case, result).passed
    assert not trajectory.max_tool_calls(1)(case, result).passed
    assert trajectory.tool_order(["classify", "classify"])(case, result).passed


async def test_llm_judge_scorer():
    judge_client = FakeModel(
        [final_result_response({"score": 0.9, "reason": "faithful and complete"})]
    )
    scorer = llm_judge("Is the answer correct?", model="fake-judge", model_client=judge_client)
    agent = scripted_agent([text_response("the answer")])
    result = await agent.run("q", output_type=str)
    score = await scorer(Case(id="c", input="q"), result)
    assert score.value == 0.9
    assert score.passed is True
    assert score.reason == "faithful and complete"


def test_load_cases_from_jsonl(tmp_path):
    path = tmp_path / "cases.jsonl"
    path.write_text(
        '{"id": "a", "input": "x", "expected": "y", "tags": {"group": "g"}}\n\n{"input": "z"}\n',
        encoding="utf-8",
    )
    cases = load_cases(path)
    assert [c.id for c in cases] == ["a", "case_2"]
    assert cases[0].tags == {"group": "g"}
