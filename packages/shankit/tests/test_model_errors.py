"""The typed model-error contract: provider failures surface as ModelError
(never a provider SDK exception), and error events say what kind of failure
happened and whether retrying is reasonable."""

import pytest
from conftest import FakeModel, text_response
from shankit import ErrorEvent, ModelError
from shankit.exceptions import error_code
from shankit.models.base import model_error_for_status


class LeakyModel(FakeModel):
    """A custom client that leaks a raw exception, as third-party clients might."""

    def __init__(self, exc: Exception) -> None:
        super().__init__([])
        self.exc = exc

    def _next(self, request):  # both complete() and stream() route through here
        raise self.exc


# ------------------------------------------------------------- the loop


async def test_run_wraps_custom_client_exception(make_agent):
    agent, _ = make_agent([])
    agent.model_client = LeakyModel(RuntimeError("socket exploded"))
    with pytest.raises(ModelError) as excinfo:
        await agent.run("go", output_type=str)
    assert isinstance(excinfo.value.__cause__, RuntimeError)
    assert not excinfo.value.retryable
    # sanitized: the raw failure detail stays on __cause__, not the message
    assert "socket exploded" not in str(excinfo.value)


async def test_run_passes_through_client_model_error(make_agent):
    err = ModelError("rate limited", provider="anthropic", status=429, retryable=True)
    agent, _ = make_agent([])
    agent.model_client = LeakyModel(err)
    with pytest.raises(ModelError) as excinfo:
        await agent.run("go", output_type=str)
    assert excinfo.value is err  # not double-wrapped by the loop fallback


async def test_stream_emits_typed_model_error_event(make_agent):
    err = ModelError("provider is overloaded", provider="anthropic", status=529, retryable=True)
    agent, _ = make_agent([])
    agent.model_client = LeakyModel(err)
    events = [e async for e in agent.stream("go")]
    error = events[-1]
    assert isinstance(error, ErrorEvent)
    assert error.code == "model_error"
    assert error.retryable
    assert error.message == "provider is overloaded"


# ------------------------------------------------------- codes & statuses


def test_error_code_mapping():
    from shankit import MaxIterationsError, OutputValidationError, ShankitError

    assert error_code(ModelError("x")) == "model_error"
    assert error_code(MaxIterationsError("x")) == "max_iterations"
    assert error_code(OutputValidationError("x")) == "output_validation"
    assert error_code(ShankitError("x")) == "error"
    assert error_code(RuntimeError("x")) == "unexpected"


def test_model_error_for_status_retryability():
    assert model_error_for_status("anthropic", 429).retryable
    assert model_error_for_status("anthropic", 529).retryable
    assert model_error_for_status("openai", 500).retryable
    assert not model_error_for_status("openai", 400).retryable
    assert not model_error_for_status("anthropic", 401).retryable
    err = model_error_for_status("anthropic", 429)
    assert err.provider == "anthropic"
    assert err.status == 429


# ------------------------------------------------- provider SDK mappings


def _http_error_parts(status: int):
    httpx = pytest.importorskip("httpx")
    request = httpx.Request("POST", "https://api.example.com/v1")
    return httpx.Response(status, request=request), request


def test_anthropic_exceptions_map_to_model_error():
    anthropic = pytest.importorskip("anthropic")
    from shankit.models.anthropic import _map_error

    response, request = _http_error_parts(429)
    mapped = _map_error(anthropic.RateLimitError("rate limited", response=response, body=None))
    assert isinstance(mapped, ModelError)
    assert (mapped.provider, mapped.status, mapped.retryable) == ("anthropic", 429, True)

    mapped = _map_error(anthropic.APIConnectionError(request=request))
    assert isinstance(mapped, ModelError)
    assert mapped.retryable

    assert _map_error(RuntimeError("not the SDK's")) is None


def test_openai_exceptions_map_to_model_error():
    openai = pytest.importorskip("openai")
    from shankit.models.openai import _map_error

    response, request = _http_error_parts(500)
    mapped = _map_error(
        openai.InternalServerError("server error", response=response, body=None)
    )
    assert isinstance(mapped, ModelError)
    assert (mapped.provider, mapped.status, mapped.retryable) == ("openai", 500, True)

    mapped = _map_error(openai.APIConnectionError(request=request))
    assert isinstance(mapped, ModelError)
    assert mapped.retryable

    assert _map_error(RuntimeError("not the SDK's")) is None


async def test_anthropic_client_raises_model_error():
    anthropic = pytest.importorskip("anthropic")
    from shankit.models.anthropic import AnthropicModel
    from shankit.models.base import ModelRequest

    response, _ = _http_error_parts(429)
    err = anthropic.RateLimitError("rate limited", response=response, body=None)

    class FailingMessages:
        async def create(self, **kwargs):
            raise err

    class FakeSDKClient:
        messages = FailingMessages()

    model = AnthropicModel(client=FakeSDKClient())
    with pytest.raises(ModelError) as excinfo:
        await model.complete(ModelRequest(model="m", messages=[]))
    assert excinfo.value.retryable
    assert excinfo.value.__cause__ is err


# --------------------------------------------------------------- via SSE


async def test_sse_stream_emits_typed_error(make_agent):
    from shankit import sse_stream

    err = ModelError("rate limited", retryable=True)

    async def broken():
        yield text_response("never sent")  # any BaseModel works for format_sse
        raise err

    lines = [line async for line in sse_stream(broken())]
    assert 'event: error' in lines[-1]
    assert '"code":"model_error"' in lines[-1]
    assert '"retryable":true' in lines[-1]
