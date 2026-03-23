"""Tests for Skill, InMemorySkillStore, and skill tools."""

import pytest

from orxhestra.skills import (
    InMemorySkillStore,
    Skill,
    make_list_skills_tool,
    make_load_skill_tool,
)


@pytest.fixture
def store():
    return InMemorySkillStore(
        skills=[
            Skill(name="python", description="Python coding", content="Write clean Python."),
            Skill(name="sql", description="SQL queries", content="Write efficient SQL."),
        ]
    )


def test_skill_defaults():
    skill = Skill(name="test", content="content")
    assert skill.name == "test"
    assert skill.description == ""
    assert skill.id is not None


def test_store_add_duplicate_raises():
    store = InMemorySkillStore()
    skill = Skill(name="s", content="c")
    store.add(skill)
    with pytest.raises(ValueError, match="already registered"):
        store.add(Skill(name="s", content="c2"))


@pytest.mark.asyncio
async def test_store_get_by_name(store):
    skill = await store.get_by_name("python")
    assert skill is not None
    assert skill.name == "python"


@pytest.mark.asyncio
async def test_store_get_by_name_not_found(store):
    skill = await store.get_by_name("nonexistent")
    assert skill is None


@pytest.mark.asyncio
async def test_store_list_skills(store):
    skills = await store.list_skills()
    assert len(skills) == 2


@pytest.mark.asyncio
async def test_load_skill_tool_found(store):
    tool = make_load_skill_tool(store)
    result = await tool.ainvoke({"name": "python"})
    assert "python" in result.lower()
    assert "Write clean Python" in result


@pytest.mark.asyncio
async def test_load_skill_tool_not_found(store):
    tool = make_load_skill_tool(store)
    result = await tool.ainvoke({"name": "nonexistent"})
    assert "not found" in result.lower()
    assert "python" in result  # lists available skills


@pytest.mark.asyncio
async def test_list_skills_tool(store):
    tool = make_list_skills_tool(store)
    result = await tool.ainvoke({})
    assert "python" in result
    assert "sql" in result


@pytest.mark.asyncio
async def test_list_skills_tool_empty():
    tool = make_list_skills_tool(InMemorySkillStore())
    result = await tool.ainvoke({})
    assert "No skills" in result
