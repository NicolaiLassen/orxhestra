"""Tests for PromptContext and build_system_prompt."""

from langchain_adk.prompts.catalog import build_system_prompt
from langchain_adk.prompts.context import PromptContext


def test_prompt_includes_agent_name():
    ctx = PromptContext(agent_name="ResearchAgent")
    prompt = build_system_prompt(ctx)
    assert "ResearchAgent" in prompt


def test_prompt_includes_date():
    ctx = PromptContext(agent_name="Agent", current_date="2025-01-01")
    prompt = build_system_prompt(ctx)
    assert "2025-01-01" in prompt


def test_prompt_includes_goal():
    ctx = PromptContext(agent_name="Agent", goal="Summarize papers")
    prompt = build_system_prompt(ctx)
    assert "Summarize papers" in prompt


def test_prompt_omits_empty_goal():
    ctx = PromptContext(agent_name="Agent", goal="")
    prompt = build_system_prompt(ctx)
    assert "goal" not in prompt.lower()


def test_prompt_includes_instructions():
    ctx = PromptContext(agent_name="Agent", instructions="Be concise.")
    prompt = build_system_prompt(ctx)
    assert "Be concise." in prompt


def test_prompt_skills_block():
    ctx = PromptContext(
        agent_name="Agent",
        skills=[{"name": "web_search", "description": "Search the web"}],
    )
    prompt = build_system_prompt(ctx)
    assert "web_search" in prompt
    assert "Search the web" in prompt


def test_prompt_agents_block():
    ctx = PromptContext(
        agent_name="Agent",
        agents=[{"name": "CodeAgent", "description": "Writes code"}],
    )
    prompt = build_system_prompt(ctx)
    assert "CodeAgent" in prompt
    assert "Writes code" in prompt


def test_prompt_tasks_block():
    ctx = PromptContext(
        agent_name="Agent",
        tasks=[{"tag": "t1", "title": "Do research", "description": "Find sources"}],
    )
    prompt = build_system_prompt(ctx)
    assert "Do research" in prompt
    assert "t1" in prompt


def test_prompt_workflow_block():
    ctx = PromptContext(
        agent_name="Agent",
        workflow_instructions="Step 1: Gather data\nStep 2: Analyze",
    )
    prompt = build_system_prompt(ctx)
    assert "Step 1" in prompt
    assert "Step 2" in prompt


def test_prompt_extra_sections():
    ctx = PromptContext(
        agent_name="Agent",
        extra_sections=["Extra note about safety."],
    )
    prompt = build_system_prompt(ctx)
    assert "Extra note about safety." in prompt


def test_prompt_empty_sections_excluded():
    ctx = PromptContext(agent_name="Agent")
    prompt = build_system_prompt(ctx)
    # Only the agent name and date should be present
    assert "Available skills" not in prompt
    assert "Tasks:" not in prompt
    assert "Workflow:" not in prompt
