"""Tests for artifact services, artifact tools, and artifact delta tracking."""

from __future__ import annotations

import pytest

from orxhestra.agents.base_agent import BaseAgent  # noqa: F401
from orxhestra.agents.invocation_context import InvocationContext as Context
from orxhestra.artifacts.base_artifact_service import (  # noqa: F401
    ArtifactVersion,
    BaseArtifactService,
)
from orxhestra.artifacts.in_memory_artifact_service import InMemoryArtifactService

Context.model_rebuild()

from orxhestra.events.event_actions import EventActions  # noqa: E402
from orxhestra.models.part import DataPart, FilePart, TextPart  # noqa: E402
from orxhestra.tools.artifact_tools import make_artifact_tools  # noqa: E402
from orxhestra.tools.call_context import CallContext  # noqa: E402


def _make_ctx(
    *,
    artifact_service: BaseArtifactService | None = None,
    app_name: str = "test-app",
    user_id: str = "user-1",
    session_id: str = "s1",
) -> Context:
    return Context(
        session_id=session_id,
        agent_name="agent",
        app_name=app_name,
        user_id=user_id,
        artifact_service=artifact_service,
    )


# ---------------------------------------------------------------------------
# InMemoryArtifactService
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_save_and_load_artifact():
    svc = InMemoryArtifactService()
    version: int = await svc.save_artifact(
        app_name="app", user_id="u1", filename="hello.txt", data=b"hello",
    )
    assert version == 0

    loaded: bytes | None = await svc.load_artifact(
        app_name="app", user_id="u1", filename="hello.txt",
    )
    assert loaded == b"hello"


@pytest.mark.asyncio
async def test_save_increments_version():
    svc = InMemoryArtifactService()
    v0: int = await svc.save_artifact(
        app_name="app", user_id="u1", filename="f.txt", data=b"v0",
    )
    v1: int = await svc.save_artifact(
        app_name="app", user_id="u1", filename="f.txt", data=b"v1",
    )
    assert v0 == 0
    assert v1 == 1

    # Latest version returned by default
    assert await svc.load_artifact(
        app_name="app", user_id="u1", filename="f.txt",
    ) == b"v1"

    # Specific version
    assert await svc.load_artifact(
        app_name="app", user_id="u1", filename="f.txt", version=0,
    ) == b"v0"


@pytest.mark.asyncio
async def test_load_nonexistent_returns_none():
    svc = InMemoryArtifactService()
    result: bytes | None = await svc.load_artifact(
        app_name="app", user_id="u1", filename="nope.txt",
    )
    assert result is None


@pytest.mark.asyncio
async def test_delete_artifact():
    svc = InMemoryArtifactService()
    await svc.save_artifact(
        app_name="app", user_id="u1", filename="f.txt", data=b"data",
    )
    deleted: bool = await svc.delete_artifact(
        app_name="app", user_id="u1", filename="f.txt",
    )
    assert deleted is True

    # Gone after delete
    assert await svc.load_artifact(
        app_name="app", user_id="u1", filename="f.txt",
    ) is None

    # Deleting again returns False
    assert await svc.delete_artifact(
        app_name="app", user_id="u1", filename="f.txt",
    ) is False


@pytest.mark.asyncio
async def test_list_versions():
    svc = InMemoryArtifactService()
    await svc.save_artifact(
        app_name="app", user_id="u1", filename="f.txt", data=b"v0",
    )
    await svc.save_artifact(
        app_name="app", user_id="u1", filename="f.txt", data=b"v1",
    )
    versions: list[int] = await svc.list_versions(
        app_name="app", user_id="u1", filename="f.txt",
    )
    assert versions == [0, 1]


@pytest.mark.asyncio
async def test_get_artifact_version_metadata():
    svc = InMemoryArtifactService()
    await svc.save_artifact(
        app_name="app", user_id="u1", filename="report.md",
        data=b"# Hello", mime_type="text/markdown",
        metadata={"author": "agent"},
    )
    info: ArtifactVersion | None = await svc.get_artifact_version(
        app_name="app", user_id="u1", filename="report.md",
    )
    assert info is not None
    assert info.version == 0
    assert info.filename == "report.md"
    assert info.mime_type == "text/markdown"
    assert info.size_bytes == len(b"# Hello")
    assert info.metadata == {"author": "agent"}


@pytest.mark.asyncio
async def test_session_scoped_artifact():
    svc = InMemoryArtifactService()
    await svc.save_artifact(
        app_name="app", user_id="u1", filename="f.txt",
        data=b"session-data", session_id="sess-1",
    )
    # Not visible without session_id
    assert await svc.load_artifact(
        app_name="app", user_id="u1", filename="f.txt",
    ) is None

    # Visible with correct session_id
    assert await svc.load_artifact(
        app_name="app", user_id="u1", filename="f.txt", session_id="sess-1",
    ) == b"session-data"


# ---------------------------------------------------------------------------
# list_artifact_keys — merged scopes
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_list_keys_merges_user_and_session_scopes():
    svc = InMemoryArtifactService()
    # User-scoped
    await svc.save_artifact(
        app_name="app", user_id="u1", filename="user-file.txt", data=b"u",
    )
    # Session-scoped
    await svc.save_artifact(
        app_name="app", user_id="u1", filename="session-file.txt",
        data=b"s", session_id="sess-1",
    )

    # With session_id → both scopes merged
    keys: list[str] = await svc.list_artifact_keys(
        app_name="app", user_id="u1", session_id="sess-1",
    )
    assert "user-file.txt" in keys
    assert "session-file.txt" in keys

    # Without session_id → only user-scoped
    user_keys: list[str] = await svc.list_artifact_keys(
        app_name="app", user_id="u1",
    )
    assert "user-file.txt" in user_keys
    assert "session-file.txt" not in user_keys


@pytest.mark.asyncio
async def test_list_keys_deduplicates():
    svc = InMemoryArtifactService()
    # Same filename in both scopes
    await svc.save_artifact(
        app_name="app", user_id="u1", filename="shared.txt", data=b"user",
    )
    await svc.save_artifact(
        app_name="app", user_id="u1", filename="shared.txt",
        data=b"session", session_id="sess-1",
    )

    keys: list[str] = await svc.list_artifact_keys(
        app_name="app", user_id="u1", session_id="sess-1",
    )
    assert keys.count("shared.txt") == 1


# ---------------------------------------------------------------------------
# FileArtifactService
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_file_artifact_service_roundtrip(tmp_path):
    from orxhestra.artifacts.file_artifact_service import FileArtifactService

    svc = FileArtifactService(tmp_path)
    v: int = await svc.save_artifact(
        app_name="app", user_id="u1", filename="data.bin",
        data=b"\x00\x01\x02", mime_type="application/octet-stream",
    )
    assert v == 0

    loaded: bytes | None = await svc.load_artifact(
        app_name="app", user_id="u1", filename="data.bin",
    )
    assert loaded == b"\x00\x01\x02"

    info: ArtifactVersion | None = await svc.get_artifact_version(
        app_name="app", user_id="u1", filename="data.bin",
    )
    assert info is not None
    assert info.mime_type == "application/octet-stream"
    assert info.size_bytes == 3


@pytest.mark.asyncio
async def test_file_artifact_service_versioning(tmp_path):
    from orxhestra.artifacts.file_artifact_service import FileArtifactService

    svc = FileArtifactService(tmp_path)
    await svc.save_artifact(
        app_name="app", user_id="u1", filename="f.txt", data=b"v0",
    )
    await svc.save_artifact(
        app_name="app", user_id="u1", filename="f.txt", data=b"v1",
    )
    assert await svc.list_versions(
        app_name="app", user_id="u1", filename="f.txt",
    ) == [0, 1]
    assert await svc.load_artifact(
        app_name="app", user_id="u1", filename="f.txt", version=0,
    ) == b"v0"


@pytest.mark.asyncio
async def test_file_artifact_service_merges_scopes(tmp_path):
    from orxhestra.artifacts.file_artifact_service import FileArtifactService

    svc = FileArtifactService(tmp_path)
    await svc.save_artifact(
        app_name="app", user_id="u1", filename="user.txt", data=b"u",
    )
    await svc.save_artifact(
        app_name="app", user_id="u1", filename="sess.txt",
        data=b"s", session_id="sess-1",
    )

    keys: list[str] = await svc.list_artifact_keys(
        app_name="app", user_id="u1", session_id="sess-1",
    )
    assert "user.txt" in keys
    assert "sess.txt" in keys


@pytest.mark.asyncio
async def test_file_artifact_service_delete(tmp_path):
    from orxhestra.artifacts.file_artifact_service import FileArtifactService

    svc = FileArtifactService(tmp_path)
    await svc.save_artifact(
        app_name="app", user_id="u1", filename="f.txt", data=b"data",
    )
    assert await svc.delete_artifact(
        app_name="app", user_id="u1", filename="f.txt",
    ) is True
    assert await svc.load_artifact(
        app_name="app", user_id="u1", filename="f.txt",
    ) is None


# ---------------------------------------------------------------------------
# EventActions.artifact_delta
# ---------------------------------------------------------------------------


def test_event_actions_artifact_delta_default():
    actions = EventActions()
    assert actions.artifact_delta == {}


def test_event_actions_artifact_delta_populated():
    actions = EventActions(artifact_delta={"report.md": 0, "data.json": 2})
    assert actions.artifact_delta["report.md"] == 0
    assert actions.artifact_delta["data.json"] == 2


# ---------------------------------------------------------------------------
# CallContext — artifact delta tracking
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_call_context_save_tracks_artifact_delta():
    svc = InMemoryArtifactService()
    ctx: Context = _make_ctx(artifact_service=svc)
    call_ctx = CallContext(ctx)

    version: int | None = await call_ctx.save_artifact(
        "output.txt", b"hello", mime_type="text/plain",
    )
    assert version == 0
    assert call_ctx.actions.artifact_delta == {"output.txt": 0}


@pytest.mark.asyncio
async def test_call_context_save_without_service_returns_none():
    ctx: Context = _make_ctx(artifact_service=None)
    call_ctx = CallContext(ctx)

    result: int | None = await call_ctx.save_artifact("f.txt", b"data")
    assert result is None
    assert call_ctx.actions.artifact_delta == {}


@pytest.mark.asyncio
async def test_call_context_load_artifact():
    svc = InMemoryArtifactService()
    ctx: Context = _make_ctx(artifact_service=svc)
    call_ctx = CallContext(ctx)

    await call_ctx.save_artifact("f.txt", b"content")
    loaded: bytes | None = await call_ctx.load_artifact("f.txt")
    assert loaded == b"content"


@pytest.mark.asyncio
async def test_call_context_list_artifacts():
    svc = InMemoryArtifactService()
    ctx: Context = _make_ctx(artifact_service=svc)
    call_ctx = CallContext(ctx)

    await call_ctx.save_artifact("a.txt", b"a")
    await call_ctx.save_artifact("b.txt", b"b")
    keys: list[str] = await call_ctx.list_artifacts()
    assert sorted(keys) == ["a.txt", "b.txt"]


@pytest.mark.asyncio
async def test_call_context_save_string_auto_encodes():
    svc = InMemoryArtifactService()
    ctx: Context = _make_ctx(artifact_service=svc)
    call_ctx = CallContext(ctx)

    await call_ctx.save_artifact("f.txt", "hello string")
    loaded: bytes | None = await call_ctx.load_artifact("f.txt")
    assert loaded == b"hello string"


# ---------------------------------------------------------------------------
# CallContext — Part-based multimodal save/load
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_save_and_load_text_part():
    svc = InMemoryArtifactService()
    ctx: Context = _make_ctx(artifact_service=svc)
    call_ctx = CallContext(ctx)

    part = TextPart(text="Hello, world!")
    version: int | None = await call_ctx.save_artifact_part("greeting.txt", part)
    assert version == 0

    loaded = await call_ctx.load_artifact_part("greeting.txt")
    assert isinstance(loaded, TextPart)
    assert loaded.text == "Hello, world!"


@pytest.mark.asyncio
async def test_save_and_load_data_part():
    svc = InMemoryArtifactService()
    ctx: Context = _make_ctx(artifact_service=svc)
    call_ctx = CallContext(ctx)

    part = DataPart(data={"key": "value", "count": 42})
    await call_ctx.save_artifact_part("config.json", part)

    loaded = await call_ctx.load_artifact_part("config.json")
    assert isinstance(loaded, DataPart)
    assert loaded.data["key"] == "value"
    assert loaded.data["count"] == 42


@pytest.mark.asyncio
async def test_save_and_load_file_part():
    import base64

    svc = InMemoryArtifactService()
    ctx: Context = _make_ctx(artifact_service=svc)
    call_ctx = CallContext(ctx)

    raw: bytes = b"\x89PNG\r\n\x1a\n"
    part = FilePart(
        name="image.png",
        mime_type="image/png",
        inline_bytes=base64.b64encode(raw).decode("ascii"),
    )
    await call_ctx.save_artifact_part("image.png", part)

    loaded = await call_ctx.load_artifact_part("image.png")
    assert isinstance(loaded, FilePart)
    assert loaded.mime_type == "image/png"
    assert base64.b64decode(loaded.inline_bytes) == raw


@pytest.mark.asyncio
async def test_load_artifact_part_nonexistent():
    svc = InMemoryArtifactService()
    ctx: Context = _make_ctx(artifact_service=svc)
    call_ctx = CallContext(ctx)

    result = await call_ctx.load_artifact_part("nope.txt")
    assert result is None


@pytest.mark.asyncio
async def test_save_artifact_part_tracks_delta():
    svc = InMemoryArtifactService()
    ctx: Context = _make_ctx(artifact_service=svc)
    call_ctx = CallContext(ctx)

    await call_ctx.save_artifact_part("doc.txt", TextPart(text="hi"))
    assert call_ctx.actions.artifact_delta == {"doc.txt": 0}


@pytest.mark.asyncio
async def test_save_artifact_part_metadata():
    svc = InMemoryArtifactService()
    ctx: Context = _make_ctx(artifact_service=svc)
    call_ctx = CallContext(ctx)

    await call_ctx.save_artifact_part(
        "doc.txt", TextPart(text="hi"), metadata={"source": "agent"},
    )
    info: ArtifactVersion | None = await svc.get_artifact_version(
        app_name="test-app", user_id="user-1", filename="doc.txt", session_id="s1",
    )
    assert info is not None
    assert info.metadata["part_type"] == "TextPart"
    assert info.metadata["source"] == "agent"


# ---------------------------------------------------------------------------
# Artifact tools (make_artifact_tools)
# ---------------------------------------------------------------------------


def test_make_artifact_tools_returns_three_tools():
    tools = make_artifact_tools()
    assert len(tools) == 3
    names: set[str] = {t.name for t in tools}
    assert names == {"save_artifact", "load_artifact", "list_artifacts"}


def test_artifact_tools_have_inject_context():
    tools = make_artifact_tools()
    for tool in tools:
        assert hasattr(tool, "inject_context")


@pytest.mark.asyncio
async def test_artifact_tool_save_and_load():
    svc = InMemoryArtifactService()
    ctx: Context = _make_ctx(artifact_service=svc)
    tools = make_artifact_tools()

    # Inject context
    for tool in tools:
        tool.inject_context(ctx)

    save_tool = next(t for t in tools if t.name == "save_artifact")
    load_tool = next(t for t in tools if t.name == "load_artifact")
    list_tool = next(t for t in tools if t.name == "list_artifacts")

    # Save
    result: str = await save_tool.ainvoke({
        "filename": "notes.txt",
        "content": "my notes",
    })
    assert "version 0" in result
    assert "notes.txt" in result

    # Load
    loaded: str = await load_tool.ainvoke({"filename": "notes.txt"})
    assert loaded == "my notes"

    # List
    listing: str = await list_tool.ainvoke({})
    assert "notes.txt" in listing


@pytest.mark.asyncio
async def test_artifact_tool_load_nonexistent():
    svc = InMemoryArtifactService()
    ctx: Context = _make_ctx(artifact_service=svc)
    tools = make_artifact_tools()
    for tool in tools:
        tool.inject_context(ctx)

    load_tool = next(t for t in tools if t.name == "load_artifact")
    result: str = await load_tool.ainvoke({"filename": "nope.txt"})
    assert "not found" in result


@pytest.mark.asyncio
async def test_artifact_tool_list_empty():
    svc = InMemoryArtifactService()
    ctx: Context = _make_ctx(artifact_service=svc)
    tools = make_artifact_tools()
    for tool in tools:
        tool.inject_context(ctx)

    list_tool = next(t for t in tools if t.name == "list_artifacts")
    result: str = await list_tool.ainvoke({})
    assert "No artifacts" in result


@pytest.mark.asyncio
async def test_artifact_tool_no_service():
    ctx: Context = _make_ctx(artifact_service=None)
    tools = make_artifact_tools()
    for tool in tools:
        tool.inject_context(ctx)

    save_tool = next(t for t in tools if t.name == "save_artifact")
    result: str = await save_tool.ainvoke({
        "filename": "f.txt", "content": "data",
    })
    assert "no artifact service" in result.lower()


@pytest.mark.asyncio
async def test_artifact_tool_raises_without_context():
    tools = make_artifact_tools()
    save_tool = next(t for t in tools if t.name == "save_artifact")
    with pytest.raises(RuntimeError, match="context"):
        await save_tool.ainvoke({"filename": "f.txt", "content": "data"})


@pytest.mark.asyncio
async def test_artifact_tool_base64_binary():
    import base64

    svc = InMemoryArtifactService()
    ctx: Context = _make_ctx(artifact_service=svc)
    tools = make_artifact_tools()
    for tool in tools:
        tool.inject_context(ctx)

    save_tool = next(t for t in tools if t.name == "save_artifact")
    load_tool = next(t for t in tools if t.name == "load_artifact")

    raw: bytes = b"\x00\x01\x02\xff"
    encoded: str = base64.b64encode(raw).decode("ascii")

    await save_tool.ainvoke({
        "filename": "binary.dat",
        "content": encoded,
        "is_base64": True,
    })

    result: str = await load_tool.ainvoke({"filename": "binary.dat"})
    # Binary data is returned as base64
    assert "base64" in result
    assert encoded in result


# ---------------------------------------------------------------------------
# Composer builtin registration
# ---------------------------------------------------------------------------


def test_artifacts_registered_as_builtin():
    from orxhestra.composer.builders.tools import resolve_builtin

    tools = resolve_builtin("artifacts")
    assert isinstance(tools, list)
    assert len(tools) == 3
    names: set[str] = {t.name for t in tools}
    assert "save_artifact" in names
    assert "load_artifact" in names
    assert "list_artifacts" in names
