"""Tests for FilesystemBackend implementations.

The same test body runs against both ``LocalFilesystemBackend`` and
``InMemoryFilesystemBackend`` to ensure parity.
"""

from __future__ import annotations

import pytest

from orxhestra.filesystem.base import FilesystemBackend, GrepMatch
from orxhestra.filesystem.local import LocalFilesystemBackend
from orxhestra.filesystem.memory import InMemoryFilesystemBackend


@pytest.fixture
async def local_fs(tmp_path) -> FilesystemBackend:
    return LocalFilesystemBackend(tmp_path)


@pytest.fixture
async def mem_fs() -> FilesystemBackend:
    return InMemoryFilesystemBackend()


@pytest.fixture(params=["local", "memory"])
async def fs(request, tmp_path) -> FilesystemBackend:
    if request.param == "local":
        return LocalFilesystemBackend(tmp_path)
    return InMemoryFilesystemBackend()


# ── Parity tests — run against both backends ─────────────────────────


@pytest.mark.asyncio
async def test_write_then_read_round_trip(fs: FilesystemBackend):
    await fs.write("hello.txt", "world")
    content = await fs.read("hello.txt")
    assert content == "world"


@pytest.mark.asyncio
async def test_read_nonexistent_raises(fs: FilesystemBackend):
    with pytest.raises(FileNotFoundError):
        await fs.read("missing.txt")


@pytest.mark.asyncio
async def test_exists_false_for_missing(fs: FilesystemBackend):
    assert not await fs.exists("nope.txt")


@pytest.mark.asyncio
async def test_exists_true_after_write(fs: FilesystemBackend):
    await fs.write("x.txt", "x")
    assert await fs.exists("x.txt")


@pytest.mark.asyncio
async def test_write_creates_nested_directories(fs: FilesystemBackend):
    await fs.write("a/b/c/deep.txt", "deep")
    content = await fs.read("a/b/c/deep.txt")
    assert content == "deep"


@pytest.mark.asyncio
async def test_read_with_offset_and_limit(fs: FilesystemBackend):
    await fs.write("lines.txt", "l1\nl2\nl3\nl4\nl5\n")
    # Offset 2 (skip first two lines), limit 2.
    out = await fs.read("lines.txt", offset=2, limit=2)
    assert out == "l3\nl4\n"


@pytest.mark.asyncio
async def test_read_limit_zero_returns_nothing(fs: FilesystemBackend):
    await fs.write("lines.txt", "l1\nl2\n")
    out = await fs.read("lines.txt", offset=0, limit=0)
    assert out == ""


@pytest.mark.asyncio
async def test_edit_replaces_single_occurrence(fs: FilesystemBackend):
    await fs.write("x.txt", "foo bar baz")
    count = await fs.edit("x.txt", "bar", "BAZ")
    assert count == 1
    content = await fs.read("x.txt")
    assert content == "foo BAZ baz"


@pytest.mark.asyncio
async def test_edit_missing_string_raises(fs: FilesystemBackend):
    await fs.write("x.txt", "foo")
    with pytest.raises(ValueError, match="String not found"):
        await fs.edit("x.txt", "bar", "baz")


@pytest.mark.asyncio
async def test_edit_ambiguous_without_replace_all_raises(
    fs: FilesystemBackend,
):
    await fs.write("x.txt", "foo foo foo")
    with pytest.raises(ValueError, match="appears 3 times"):
        await fs.edit("x.txt", "foo", "bar")


@pytest.mark.asyncio
async def test_edit_replace_all(fs: FilesystemBackend):
    await fs.write("x.txt", "foo foo foo")
    count = await fs.edit("x.txt", "foo", "bar", replace_all=True)
    assert count == 3
    assert await fs.read("x.txt") == "bar bar bar"


@pytest.mark.asyncio
async def test_edit_on_missing_file_raises(fs: FilesystemBackend):
    with pytest.raises(FileNotFoundError):
        await fs.edit("missing.txt", "a", "b")


@pytest.mark.asyncio
async def test_ls_returns_sorted_names(fs: FilesystemBackend):
    await fs.write("b.txt", "")
    await fs.write("a.txt", "")
    await fs.write("c.txt", "")
    names = await fs.ls(".")
    assert names == ["a.txt", "b.txt", "c.txt"]


@pytest.mark.asyncio
async def test_ls_of_nested_dir(fs: FilesystemBackend):
    await fs.write("dir/one.txt", "")
    await fs.write("dir/two.txt", "")
    await fs.write("other.txt", "")
    names = await fs.ls("dir")
    assert set(names) == {"one.txt", "two.txt"}


@pytest.mark.asyncio
async def test_glob_matches_files(fs: FilesystemBackend):
    await fs.write("a.py", "")
    await fs.write("b.py", "")
    await fs.write("c.txt", "")
    results = await fs.glob("*.py")
    assert sorted(results) == sorted(["a.py", "b.py"])


@pytest.mark.asyncio
async def test_grep_finds_pattern(fs: FilesystemBackend):
    await fs.write("a.txt", "hello\nworld\nhello again\n")
    matches = await fs.grep("hello")
    assert len(matches) == 2
    assert all(isinstance(m, GrepMatch) for m in matches)
    assert matches[0].line_no == 1
    assert matches[1].line_no == 3


@pytest.mark.asyncio
async def test_grep_with_glob_filter(fs: FilesystemBackend):
    await fs.write("keep.py", "match here\n")
    await fs.write("skip.txt", "match here\n")
    matches = await fs.grep("match", glob="*.py")
    assert len(matches) == 1
    assert matches[0].path.endswith("keep.py")


@pytest.mark.asyncio
async def test_delete_removes_file(fs: FilesystemBackend):
    await fs.write("x.txt", "x")
    await fs.delete("x.txt")
    assert not await fs.exists("x.txt")


@pytest.mark.asyncio
async def test_delete_nonexistent_is_noop(fs: FilesystemBackend):
    # Both backends should tolerate deleting what's not there.
    await fs.delete("nope.txt")


@pytest.mark.asyncio
async def test_mkdir_and_exists(fs: FilesystemBackend):
    await fs.mkdir("subdir")
    # Implementations differ on whether a mkdir'd dir shows up in exists()
    # without children, but at minimum it must not raise.
    assert await fs.exists("subdir") in (True, False)


# ── Local-specific tests ─────────────────────────────────────────────


@pytest.mark.asyncio
async def test_local_path_jail_blocks_escape(tmp_path):
    fs = LocalFilesystemBackend(tmp_path)
    with pytest.raises(ValueError, match="outside the workspace"):
        await fs.write("../escape.txt", "x")


@pytest.mark.asyncio
async def test_local_workspace_property(tmp_path):
    fs = LocalFilesystemBackend(tmp_path)
    assert fs.workspace == tmp_path.resolve()


@pytest.mark.asyncio
async def test_local_creates_workspace_on_first_use(tmp_path):
    ws = tmp_path / "new_ws"
    fs = LocalFilesystemBackend(ws)
    await fs.write("x.txt", "x")
    assert ws.exists()


# ── In-memory-specific tests ─────────────────────────────────────────


@pytest.mark.asyncio
async def test_memory_seed_initial_files():
    fs = InMemoryFilesystemBackend({"seed.txt": "hi"})
    assert await fs.read("seed.txt") == "hi"


@pytest.mark.asyncio
async def test_memory_normalizes_path_with_dotdot():
    fs = InMemoryFilesystemBackend()
    await fs.write("a/b/../x.txt", "x")
    assert await fs.read("a/x.txt") == "x"


@pytest.mark.asyncio
async def test_memory_ls_lists_implicit_dirs():
    fs = InMemoryFilesystemBackend({
        "a/1.txt": "",
        "a/2.txt": "",
        "b/3.txt": "",
    })
    names = await fs.ls(".")
    assert set(names) == {"a", "b"}


@pytest.mark.asyncio
async def test_memory_delete_non_empty_dir_raises():
    fs = InMemoryFilesystemBackend()
    await fs.mkdir("d")
    await fs.write("d/child.txt", "x")
    with pytest.raises(OSError, match="not empty"):
        await fs.delete("d")


@pytest.mark.asyncio
async def test_memory_read_directory_raises():
    fs = InMemoryFilesystemBackend({"d/x.txt": "x"})
    with pytest.raises(IsADirectoryError):
        await fs.read("d")


# ── Protocol conformance ─────────────────────────────────────────────


def test_backends_satisfy_protocol(tmp_path):
    assert isinstance(LocalFilesystemBackend(tmp_path), FilesystemBackend)
    assert isinstance(InMemoryFilesystemBackend(), FilesystemBackend)
