from conftest import FakeModel, final_result_response, text_response, tool_call_response
from pydantic import BaseModel
from shankit import Agent, tool
from shankit.evals import Case, Dataset, evaluate, exact_match, output_contains, trajectory
from shankit.evals.scorers import Score, llm_judge


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
    dataset = Dataset(cases=[Case(name="spammy", input="win money", expected={"label": "spam"})])
    report = await evaluate(
        agent,
        dataset,
        scorers=[exact_match, trajectory.used_tool("classify"), trajectory.no_tool_errors],
        output_type=Verdict,
    )
    assert report.pass_rate == 1.0
    assert report.mean("exact_match") == 1.0
    assert "PASS" in report.summary()


async def test_failing_and_erroring_cases():
    async def target(case):
        if case.name == "boom":
            raise RuntimeError("model down")
        agent = scripted_agent([text_response("looks like ham")])
        return await agent.run(case.input, output_type=str)

    dataset = Dataset(
        cases=[
            Case(name="boom", input="x"),
            Case(name="wrong", input="y", expected="spam"),
        ]
    )
    report = await evaluate(target, dataset, scorers=[output_contains()])
    assert report.pass_rate == 0.0
    statuses = {r.case.name: (r.error, r.passed) for r in report.results}
    assert statuses["boom"][0] is not None
    assert statuses["wrong"] == (None, False)


def test_output_contains_scores_answer_not_transcript():
    """An agent that merely *mentions* the expected string in an interim
    pass must not pass; the scorer targets the deliverable."""
    from shankit import RunResult, Usage

    result = RunResult(
        output="ham",
        text="I'll check whether this is spam.\n\nham",
        usage=Usage(),
    )
    score = output_contains("spam")(Case(name="c", input="x"), result)
    assert not score.passed
    assert output_contains("ham")(Case(name="c", input="x"), result).passed


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
        Dataset(cases=[Case(name="c", input="x")]),
        scorers=[boolean_scorer, float_scorer, full_scorer],
        output_type=str,
    )
    scores = report.results[0].scores
    assert scores[0].name == "boolean_scorer"
    assert scores[0].passed is True
    assert scores[1].value == 0.5
    assert scores[1].passed is None
    assert scores[2].reason == "looked good"


async def test_trajectory_assertions():
    agent = scripted_agent(
        [
            tool_call_response("classify", {"text": "a"}, call_id="t1"),
            tool_call_response("classify", {"text": "b"}, call_id="t2"),
            text_response("done"),
        ]
    )
    result = await agent.run("go", output_type=str)
    case = Case(name="c", input="go")

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
    score = await scorer(Case(name="c", input="q"), result)
    assert score.value == 0.9
    assert score.passed is True
    assert score.reason == "faithful and complete"


def test_dataset_jsonl(tmp_path):
    path = tmp_path / "cases.jsonl"
    path.write_text(
        '{"name": "a", "input": "x", "expected": "y"}\n{"input": "z"}\n', encoding="utf-8"
    )
    dataset = Dataset.from_jsonl(path)
    assert len(dataset) == 2
    assert dataset.cases[1].name == "case_1"
