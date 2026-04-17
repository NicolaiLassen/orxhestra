"""Tests for ``make_filesystem_tools`` with a pluggable backend.

Covers the new ``backend=`` kwarg on the rich filesystem tool factory —
the default still builds a :class:`LocalFilesystemBackend`, but any
:class:`FilesystemBackend` can be injected. Running the same tool set
against :class:`InMemoryFilesystemBackend` gives us a fast,
deterministic exercise of the backend integration.
"""

from __future__ import annotations

import pytest

from orxhestra.filesystem.memory import InMemoryFilesystemBackend
from orxhestra.tools.filesystem import make_filesystem_tools


@pytest.fixture
def tools_mem():
    fs = InMemoryFilesystemBackend()
    tools = {t.name: t for t in make_filesystem_tools(backend=fs)}
    return tools, fs


# ── Construction ─────────────────────────────────────────────────────


def test_passing_backend_returns_full_tool_set():
    fs = InMemoryFilesystemBackend()
    tools = {t.name for t in make_filesystem_tools(backend=fs)}
    assert tools == {
        "ls", "read_file", "write_file", "edit_file",
        "mkdir", "glob", "grep",
    }


def test_default_workspace_builds_local_backend(tmp_path):
    # Default path: no backend, just a workspace.
    tools = make_filesystem_tools(workspace=str(tmp_path))
    assert {t.name for t in tools} == {
        "ls", "read_file", "write_file", "edit_file",
        "mkdir", "glob", "grep",
    }


def test_passing_both_workspace_and_backend_errors():
    fs = InMemoryFilesystemBackend()
    with pytest.raises(ValueError, match="not both"):
        make_filesystem_tools(workspace="/tmp/x", backend=fs)


# ── Rich read_file formatting ───────────────────────────────────────


@pytest.mark.asyncio
async def test_read_file_adds_line_numbers(tools_mem):
    tools, fs = tools_mem
    await fs.write("a.txt", "hello\nworld\n")
    result = await tools["read_file"].ainvoke({"path": "a.txt"})
    # Header + line-numbered body
    assert result.startswith("Lines 1-2 of 2\n")
    assert "1\thello" in result
    assert "2\tworld" in result


@pytest.mark.asyncio
async def test_read_file_pagination_header(tools_mem):
    tools, fs = tools_mem
    await fs.write("big.txt", "\n".join(f"l{i}" for i in range(10)) + "\n")
    result = await tools["read_file"].ainvoke({
        "path": "big.txt", "offset": 5, "limit": 3,
    })
    assert "Lines 6-8 of 10" in result
    assert "use offset=8" in result


@pytest.mark.asyncio
async def test_read_file_missing_returns_error(tools_mem):
    tools, _ = tools_mem
    result = await tools["read_file"].ainvoke({"path": "missing.txt"})
    assert "does not exist" in result


# ── Rich edit_file diff output ──────────────────────────────────────


@pytest.mark.asyncio
async def test_edit_file_produces_diff_summary(tools_mem):
    tools, fs = tools_mem
    await fs.write("doc.md", "# Title\nold content\nend\n")
    result = await tools["edit_file"].ainvoke({
        "path": "doc.md", "old": "old content", "new": "new content",
    })
    assert "Edited doc.md" in result
    assert "- old content" in result
    assert "+ new content" in result
    assert "1 lines removed" in result
    content = await fs.read("doc.md")
    assert content == "# Title\nnew content\nend\n"


@pytest.mark.asyncio
async def test_edit_file_first_occurrence_only(tools_mem):
    """Rich factory preserves pre-refactor silent first-occurrence semantics."""
    tools, fs = tools_mem
    await fs.write("x.txt", "foo foo foo")
    result = await tools["edit_file"].ainvoke({
        "path": "x.txt", "old": "foo", "new": "BAR",
    })
    assert "Edited" in result
    content = await fs.read("x.txt")
    assert content == "BAR foo foo"


@pytest.mark.asyncio
async def test_edit_missing_substring_returns_error(tools_mem):
    tools, fs = tools_mem
    await fs.write("x.txt", "hello")
    result = await tools["edit_file"].ainvoke({
        "path": "x.txt", "old": "missing", "new": "x",
    })
    assert "not found" in result


# ── write_file / mkdir / ls ─────────────────────────────────────────


@pytest.mark.asyncio
async def test_write_file_returns_byte_count(tools_mem):
    tools, fs = tools_mem
    result = await tools["write_file"].ainvoke({
        "path": "notes.md", "content": "hello",
    })
    assert "Wrote 5 characters" in result
    assert await fs.read("notes.md") == "hello"


@pytest.mark.asyncio
async def test_mkdir_is_idempotent(tools_mem):
    tools, _ = tools_mem
    await tools["mkdir"].ainvoke({"path": "subdir"})
    result = await tools["mkdir"].ainvoke({"path": "subdir"})
    assert "Created directory" in result


@pytest.mark.asyncio
async def test_ls_lists_names(tools_mem):
    tools, fs = tools_mem
    await fs.write("a.txt", "")
    await fs.write("b.txt", "")
    result = await tools["ls"].ainvoke({"path": "."})
    names = set(result.splitlines())
    assert names == {"a.txt", "b.txt"}


@pytest.mark.asyncio
async def test_ls_empty_directory_message(tools_mem):
    tools, _ = tools_mem
    result = await tools["ls"].ainvoke({"path": "."})
    assert result == "(empty directory)"


@pytest.mark.asyncio
async def test_ls_missing_path_returns_error(tools_mem):
    tools, _ = tools_mem
    result = await tools["ls"].ainvoke({"path": "nope"})
    assert "does not exist" in result


# ── glob ────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_glob_matches_files(tools_mem):
    tools, fs = tools_mem
    await fs.write("a.py", "")
    await fs.write("b.py", "")
    await fs.write("c.txt", "")
    result = await tools["glob"].ainvoke({"pattern": "*.py"})
    lines = set(result.splitlines())
    assert lines == {"a.py", "b.py"}


@pytest.mark.asyncio
async def test_glob_no_matches(tools_mem):
    tools, _ = tools_mem
    result = await tools["glob"].ainvoke({"pattern": "*.rs"})
    assert result == "(no matches)"


@pytest.mark.asyncio
async def test_glob_truncates_over_max_results(tools_mem):
    tools, fs = tools_mem
    for i in range(5):
        await fs.write(f"f{i}.txt", "")
    result = await tools["glob"].ainvoke({
        "pattern": "*.txt", "max_results": 3,
    })
    assert "truncated" in result
    # 3 filenames + 1 truncation marker
    assert len(result.splitlines()) == 4


# ── grep with rich features (context, case_insensitive) ─────────────


@pytest.mark.asyncio
async def test_grep_basic_match(tools_mem):
    tools, fs = tools_mem
    await fs.write("a.txt", "hello\nworld\nhello again\n")
    result = await tools["grep"].ainvoke({"pattern": "hello"})
    lines = result.splitlines()
    assert len(lines) == 2
    assert "a.txt:1:" in lines[0]
    assert "a.txt:3:" in lines[1]


@pytest.mark.asyncio
async def test_grep_case_insensitive(tools_mem):
    tools, fs = tools_mem
    await fs.write("a.txt", "Hello\nworld\n")
    result = await tools["grep"].ainvoke({
        "pattern": "hello", "case_insensitive": True,
    })
    assert "a.txt:1:" in result


@pytest.mark.asyncio
async def test_grep_with_context(tools_mem):
    tools, fs = tools_mem
    await fs.write("a.txt", "one\ntwo\nMATCH\nfour\nfive\n")
    result = await tools["grep"].ainvoke({
        "pattern": "MATCH", "context": 1,
    })
    # context=1 → 1 before + 1 after. Includes "--" separator.
    assert "> MATCH" in result
    assert "two" in result
    assert "four" in result


@pytest.mark.asyncio
async def test_grep_with_glob_filter(tools_mem):
    tools, fs = tools_mem
    await fs.write("keep.py", "match me\n")
    await fs.write("skip.txt", "match me\n")
    result = await tools["grep"].ainvoke({
        "pattern": "match", "glob_filter": "*.py",
    })
    lines = result.splitlines()
    assert len(lines) == 1
    assert "keep.py" in lines[0]


@pytest.mark.asyncio
async def test_grep_no_matches(tools_mem):
    tools, fs = tools_mem
    await fs.write("x.txt", "nothing here")
    result = await tools["grep"].ainvoke({"pattern": "zzz"})
    assert result == "(no matches)"


@pytest.mark.asyncio
async def test_grep_truncates_over_max_results(tools_mem):
    tools, fs = tools_mem
    await fs.write(
        "big.txt",
        "\n".join(["match"] * 10),
    )
    result = await tools["grep"].ainvoke({
        "pattern": "match", "max_results": 3,
    })
    assert "truncated" in result


# ── default backend produces tools that hit disk ────────────────────


@pytest.mark.asyncio
async def test_default_backend_writes_real_files(tmp_path):
    tools = {t.name: t for t in make_filesystem_tools(workspace=str(tmp_path))}
    await tools["write_file"].ainvoke({
        "path": "hello.txt", "content": "world",
    })
    disk_content = (tmp_path / "hello.txt").read_text()
    assert disk_content == "world"
