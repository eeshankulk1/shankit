import pytest
from pydantic_core import PydanticSerializationError
from shankit.durability import (
    Checkpoint,
    InMemoryCheckpointer,
    InterruptInfo,
    SqliteCheckpointer,
)


@pytest.fixture(params=["memory", "sqlite"])
def checkpointer(request, tmp_path):
    if request.param == "memory":
        return InMemoryCheckpointer()
    return SqliteCheckpointer(tmp_path / "threads.db")


async def test_roundtrip(checkpointer):
    checkpoint = Checkpoint(
        state={"draft": "hello", "count": 2},
        status="interrupted",
        interrupt=InterruptInfo(reason="approve the email", key="approval", payload={"to": "x"}),
        steps_run=["gather", "draft"],
    )
    await checkpointer.save("t1", checkpoint)
    loaded = await checkpointer.load("t1")
    assert loaded is not None
    assert loaded.state == {"draft": "hello", "count": 2}
    assert loaded.status == "interrupted"
    assert loaded.interrupt.reason == "approve the email"
    assert loaded.interrupt.payload == {"to": "x"}
    assert loaded.steps_run == ["gather", "draft"]


async def test_upsert(checkpointer):
    await checkpointer.save("t1", Checkpoint(state={"v": 1}))
    await checkpointer.save("t1", Checkpoint(state={"v": 2}))
    loaded = await checkpointer.load("t1")
    assert loaded.state == {"v": 2}


async def test_missing_thread(checkpointer):
    assert await checkpointer.load("nope") is None


async def test_delete(checkpointer):
    await checkpointer.save("t1", Checkpoint(state={}))
    await checkpointer.delete("t1")
    assert await checkpointer.load("t1") is None
    await checkpointer.delete("t1")  # no-op, no error


async def test_snapshot_semantics(checkpointer):
    state = {"items": [1]}
    await checkpointer.save("t1", Checkpoint(state=state))
    state["items"].append(2)  # mutating live state must not mutate the checkpoint
    loaded = await checkpointer.load("t1")
    assert loaded.state == {"items": [1]}


async def test_json_contract_enforced(checkpointer):
    class NotJson:
        pass

    with pytest.raises(PydanticSerializationError):
        await checkpointer.save("t1", Checkpoint(state={"bad": NotJson()}))


async def test_sqlite_persists_across_instances(tmp_path):
    path = tmp_path / "durable.db"
    first = SqliteCheckpointer(path)
    await first.save("t1", Checkpoint(state={"v": 1}, status="done"))
    second = SqliteCheckpointer(path)
    loaded = await second.load("t1")
    assert loaded.state == {"v": 1}
    assert loaded.status == "done"
