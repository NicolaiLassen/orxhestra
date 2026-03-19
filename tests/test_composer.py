"""Tests for the Composer module: schema, model factory, tool resolver, and full builds."""

from __future__ import annotations

import textwrap
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from langchain_adk.agents.llm_agent import LlmAgent
from langchain_adk.agents.loop_agent import LoopAgent
from langchain_adk.agents.parallel_agent import ParallelAgent
from langchain_adk.agents.sequential_agent import SequentialAgent
from langchain_adk.composer.builders.models import _REGISTRY as _PROVIDER_REGISTRY
from langchain_adk.composer.builders.models import create as create_model
from langchain_adk.composer.builders.models import register as register_provider
from langchain_adk.composer.builders.tools import import_object
from langchain_adk.composer.builders.tools import register_builtin as register_builtin_tool
from langchain_adk.composer.builders.tools import resolve_builtin as resolve_builtin_tool
from langchain_adk.composer.builders.tools import resolve_function as resolve_function_tool
from langchain_adk.composer.errors import CircularReferenceError, ComposerError
from langchain_adk.composer.schema import ComposeSpec, ToolDef

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_yaml(tmp_path: Path, content: str) -> Path:
    """Write a YAML string to a temp file and return its path."""
    p = tmp_path / "compose.yaml"
    p.write_text(textwrap.dedent(content))
    return p


def _mock_llm() -> MagicMock:
    """Return a mock that passes as a BaseChatModel for construction."""
    return MagicMock()


# ---------------------------------------------------------------------------
# Schema tests
# ---------------------------------------------------------------------------


class TestSchema:
    """Tests for ComposeSpec Pydantic validation."""

    def test_minimal_valid(self):
        spec = ComposeSpec.model_validate(
            {
                "agents": {"bot": {"type": "llm"}},
                "main_agent": "bot",
            }
        )
        assert spec.main_agent == "bot"
        assert "bot" in spec.agents

    def test_root_not_in_agents_raises(self):
        with pytest.raises(ValueError, match="main_agent 'missing'"):
            ComposeSpec.model_validate(
                {
                    "agents": {"bot": {"type": "llm"}},
                    "main_agent": "missing",
                }
            )

    def test_tool_def_requires_one_type(self):
        with pytest.raises(ValueError, match="exactly one"):
            ToolDef.model_validate({})

    def test_tool_def_rejects_multiple_types(self):
        with pytest.raises(ValueError, match="exactly one"):
            ToolDef.model_validate({"function": "a.b", "builtin": "exit_loop"})

    def test_tool_def_function(self):
        td = ToolDef.model_validate({"function": "myapp.tools.search"})
        assert td.function == "myapp.tools.search"

    def test_tool_def_agent(self):
        td = ToolDef.model_validate({"agent": "ResearchAgent"})
        assert td.agent == "ResearchAgent"

    def test_tool_def_transfer(self):
        td = ToolDef.model_validate({"transfer": {"targets": ["A", "B"]}})
        assert td.transfer is not None
        assert td.transfer.targets == ["A", "B"]

    def test_full_spec(self):
        spec = ComposeSpec.model_validate(
            {
                "defaults": {"model": {"provider": "openai", "name": "gpt-4o"}},
                "tools": {"search": {"function": "myapp.tools.search"}},
                "agents": {
                    "bot": {
                        "type": "llm",
                        "instructions": "Help the user.",
                        "tools": ["search"],
                    }
                },
                "main_agent": "bot",
                "runner": {"app_name": "test"},
            }
        )
        assert spec.defaults.model.provider == "openai"
        assert "search" in spec.tools
        assert spec.runner.app_name == "test"


# ---------------------------------------------------------------------------
# Model factory tests
# ---------------------------------------------------------------------------


class TestModelFactory:
    """Tests for provider registry and create_model."""

    def setup_method(self):
        _PROVIDER_REGISTRY.clear()

    def test_register_provider(self):
        mock_cls = MagicMock()
        register_provider("custom", mock_cls)
        model = create_model("custom", "my-model", temperature=0.5)
        mock_cls.assert_called_once_with(model="my-model", temperature=0.5)
        assert model == mock_cls.return_value

    def test_cached_after_first_use(self):
        mock_cls = MagicMock()
        register_provider("cached", mock_cls)
        create_model("cached", "m1")
        create_model("cached", "m2")
        assert mock_cls.call_count == 2
        assert "cached" in _PROVIDER_REGISTRY

    def test_unknown_provider_raises(self):
        with pytest.raises(ComposerError, match="Failed to load"):
            create_model("nonexistent.provider.Cls", "model")

    def test_lazy_provider_resolved_via_registry(self):
        mock_cls = MagicMock()
        register_provider("openai", mock_cls)
        create_model("openai", "gpt-4o")
        mock_cls.assert_called_once_with(model="gpt-4o")


# ---------------------------------------------------------------------------
# Tool resolver tests
# ---------------------------------------------------------------------------


class TestToolResolver:
    """Tests for tool resolution functions."""

    def test_import_object_valid(self):
        obj = import_object("os.path.join")
        import os.path

        assert obj is os.path.join

    def test_import_object_invalid_path(self):
        with pytest.raises(ComposerError, match="Invalid import path"):
            import_object("nomodule")

    def test_import_object_missing_attr(self):
        with pytest.raises(ComposerError, match="has no attribute"):
            import_object("os.path.nonexistent_thing_xyz")

    def test_resolve_builtin_exit_loop(self):
        tool = resolve_builtin_tool("exit_loop")
        assert tool is not None
        assert tool.name == "exit_loop"

    def test_resolve_builtin_unknown(self):
        with pytest.raises(ComposerError, match="Unknown builtin"):
            resolve_builtin_tool("nonexistent_tool")

    def test_register_custom_builtin(self):
        mock_tool = MagicMock()
        register_builtin_tool("my_custom", lambda: mock_tool)
        result = resolve_builtin_tool("my_custom")
        assert result is mock_tool

    def test_resolve_function_tool(self):
        tool = resolve_function_tool(
            "os.path.join", name="path_join", description="Join paths"
        )
        assert tool.name == "path_join"


# ---------------------------------------------------------------------------
# Composer build tests
# ---------------------------------------------------------------------------


class TestComposerBuild:
    """Tests for full YAML -> agent tree builds."""

    @patch("langchain_adk.composer.builders.models._resolve_provider")
    async def test_simple_agent(self, mock_resolve, tmp_path):
        mock_resolve.return_value = lambda **kw: _mock_llm()
        yaml_path = _write_yaml(
            tmp_path,
            """\
            agents:
              bot:
                type: llm
                instructions: "You are helpful."
            main_agent: bot
            """,
        )
        from langchain_adk.composer import Composer

        agent = await Composer.from_yaml_async(yaml_path)
        assert isinstance(agent, LlmAgent)
        assert agent.name == "bot"

    @patch("langchain_adk.composer.builders.models._resolve_provider")
    async def test_sequential_agent(self, mock_resolve, tmp_path):
        mock_resolve.return_value = lambda **kw: _mock_llm()
        yaml_path = _write_yaml(
            tmp_path,
            """\
            agents:
              step1:
                type: llm
                instructions: "Step 1."
              step2:
                type: llm
                instructions: "Step 2."
              pipeline:
                type: sequential
                agents: [step1, step2]
            main_agent: pipeline
            """,
        )
        from langchain_adk.composer import Composer

        agent = await Composer.from_yaml_async(yaml_path)
        assert isinstance(agent, SequentialAgent)
        assert len(agent.sub_agents) == 2

    @patch("langchain_adk.composer.builders.models._resolve_provider")
    async def test_parallel_agent(self, mock_resolve, tmp_path):
        mock_resolve.return_value = lambda **kw: _mock_llm()
        yaml_path = _write_yaml(
            tmp_path,
            """\
            agents:
              a:
                type: llm
              b:
                type: llm
              both:
                type: parallel
                agents: [a, b]
            main_agent: both
            """,
        )
        from langchain_adk.composer import Composer

        agent = await Composer.from_yaml_async(yaml_path)
        assert isinstance(agent, ParallelAgent)

    @patch("langchain_adk.composer.builders.models._resolve_provider")
    async def test_loop_agent(self, mock_resolve, tmp_path):
        mock_resolve.return_value = lambda **kw: _mock_llm()
        yaml_path = _write_yaml(
            tmp_path,
            """\
            agents:
              worker:
                type: llm
              loop:
                type: loop
                agents: [worker]
                max_iterations: 3
            main_agent: loop
            """,
        )
        from langchain_adk.composer import Composer

        agent = await Composer.from_yaml_async(yaml_path)
        assert isinstance(agent, LoopAgent)
        assert agent.max_iterations == 3

    @patch("langchain_adk.composer.builders.models._resolve_provider")
    async def test_agent_tool_reference(self, mock_resolve, tmp_path):
        mock_resolve.return_value = lambda **kw: _mock_llm()
        yaml_path = _write_yaml(
            tmp_path,
            """\
            tools:
              researcher:
                agent: "research"
            agents:
              research:
                type: llm
                description: "Research agent."
                instructions: "Research topics."
              writer:
                type: llm
                instructions: "Write articles."
                tools: [researcher]
            main_agent: writer
            """,
        )
        from langchain_adk.composer import Composer

        agent = await Composer.from_yaml_async(yaml_path)
        assert isinstance(agent, LlmAgent)
        assert len(agent._tools) == 1
        tool = list(agent._tools.values())[0]
        assert tool.name == "research"

    @patch("langchain_adk.composer.builders.models._resolve_provider")
    async def test_transfer_tool(self, mock_resolve, tmp_path):
        mock_resolve.return_value = lambda **kw: _mock_llm()
        yaml_path = _write_yaml(
            tmp_path,
            """\
            agents:
              sales:
                type: llm
                description: "Sales agent."
              support:
                type: llm
                description: "Support agent."
              triage:
                type: llm
                instructions: "Route requests."
                tools:
                  - transfer:
                      targets: [sales, support]
            main_agent: triage
            """,
        )
        from langchain_adk.composer import Composer

        agent = await Composer.from_yaml_async(yaml_path)
        assert isinstance(agent, LlmAgent)
        assert "transfer_to_agent" in agent._tools
        # Transfer targets should be registered as sub-agents
        assert len(agent.sub_agents) == 2
        sub_names = {a.name for a in agent.sub_agents}
        assert sub_names == {"sales", "support"}

    @patch("langchain_adk.composer.builders.models._resolve_provider")
    async def test_builtin_exit_loop_tool(self, mock_resolve, tmp_path):
        mock_resolve.return_value = lambda **kw: _mock_llm()
        yaml_path = _write_yaml(
            tmp_path,
            """\
            agents:
              worker:
                type: llm
                tools:
                  - builtin: exit_loop
            main_agent: worker
            """,
        )
        from langchain_adk.composer import Composer

        agent = await Composer.from_yaml_async(yaml_path)
        assert "exit_loop" in agent._tools

    @patch("langchain_adk.composer.builders.models._resolve_provider")
    async def test_model_override_per_agent(self, mock_resolve, tmp_path):
        calls = []

        def track_provider(**kw):
            calls.append(kw)
            return _mock_llm()

        mock_resolve.return_value = track_provider
        yaml_path = _write_yaml(
            tmp_path,
            """\
            defaults:
              model:
                provider: openai
                name: gpt-4o
            agents:
              bot:
                type: llm
                model:
                  provider: anthropic
                  name: claude-sonnet-4-20250514
            main_agent: bot
            """,
        )
        from langchain_adk.composer import Composer

        await Composer.from_yaml_async(yaml_path)
        # The model should have been created with anthropic overrides
        assert len(calls) == 1
        assert calls[0]["model"] == "claude-sonnet-4-20250514"

    @patch("langchain_adk.composer.builders.models._resolve_provider")
    async def test_circular_reference_detected(self, mock_resolve, tmp_path):
        mock_resolve.return_value = lambda **kw: _mock_llm()
        yaml_path = _write_yaml(
            tmp_path,
            """\
            agents:
              a:
                type: llm
                tools:
                  - agent: b
              b:
                type: llm
                tools:
                  - agent: a
            main_agent: a
            """,
        )
        from langchain_adk.composer import Composer

        with pytest.raises(CircularReferenceError, match="Circular"):
            await Composer.from_yaml_async(yaml_path)

    async def test_missing_agent_reference(self, tmp_path):
        yaml_path = _write_yaml(
            tmp_path,
            """\
            agents:
              pipeline:
                type: sequential
                agents: [nonexistent]
            main_agent: pipeline
            """,
        )
        from langchain_adk.composer import Composer

        with pytest.raises(ComposerError, match="not defined"):
            await Composer.from_yaml_async(yaml_path)

    async def test_file_not_found(self):
        from langchain_adk.composer import Composer

        with pytest.raises(ComposerError, match="File not found"):
            await Composer.from_yaml_async("/nonexistent/compose.yaml")

    async def test_missing_runner_section(self, tmp_path):
        yaml_path = _write_yaml(
            tmp_path,
            """\
            agents:
              bot:
                type: llm
            main_agent: bot
            """,
        )
        from langchain_adk.composer import Composer

        with pytest.raises(ComposerError, match="No 'runner' section"):
            await Composer.runner_from_yaml_async(yaml_path)

    @patch("langchain_adk.composer.builders.models._resolve_provider")
    async def test_runner_build(self, mock_resolve, tmp_path):
        mock_resolve.return_value = lambda **kw: _mock_llm()
        yaml_path = _write_yaml(
            tmp_path,
            """\
            agents:
              bot:
                type: llm
            main_agent: bot
            runner:
              app_name: test-app
              session_service: memory
            """,
        )
        from langchain_adk.composer import Composer
        from langchain_adk.runner import Runner

        runner = await Composer.runner_from_yaml_async(yaml_path)
        assert isinstance(runner, Runner)

    @patch("langchain_adk.composer.builders.models._resolve_provider")
    async def test_orchestrator_missing_agents_list(self, mock_resolve, tmp_path):
        mock_resolve.return_value = lambda **kw: _mock_llm()
        yaml_path = _write_yaml(
            tmp_path,
            """\
            agents:
              pipeline:
                type: sequential
            main_agent: pipeline
            """,
        )
        from langchain_adk.composer import Composer

        with pytest.raises(ComposerError, match="must have an 'agents' list"):
            await Composer.from_yaml_async(yaml_path)

    @patch("langchain_adk.composer.builders.models._resolve_provider")
    async def test_inline_tool_def(self, mock_resolve, tmp_path):
        mock_resolve.return_value = lambda **kw: _mock_llm()
        yaml_path = _write_yaml(
            tmp_path,
            """\
            agents:
              bot:
                type: llm
                tools:
                  - function: "os.path.join"
            main_agent: bot
            """,
        )
        from langchain_adk.composer import Composer

        agent = await Composer.from_yaml_async(yaml_path)
        assert isinstance(agent, LlmAgent)
        assert len(agent._tools) == 1

    @patch("langchain_adk.composer.builders.models._resolve_provider")
    async def test_planner_plan_react(self, mock_resolve, tmp_path):
        mock_resolve.return_value = lambda **kw: _mock_llm()
        yaml_path = _write_yaml(
            tmp_path,
            """\
            agents:
              bot:
                type: llm
                planner:
                  type: plan_react
            main_agent: bot
            """,
        )
        from langchain_adk.composer import Composer
        from langchain_adk.planners.plan_re_act_planner import PlanReActPlanner

        agent = await Composer.from_yaml_async(yaml_path)
        assert isinstance(agent._planner, PlanReActPlanner)

    @patch("langchain_adk.composer.builders.models._resolve_provider")
    async def test_planner_task(self, mock_resolve, tmp_path):
        mock_resolve.return_value = lambda **kw: _mock_llm()
        yaml_path = _write_yaml(
            tmp_path,
            """\
            agents:
              bot:
                type: llm
                planner:
                  type: task
                  tasks:
                    - title: "Research"
                      required: true
            main_agent: bot
            """,
        )
        from langchain_adk.composer import Composer
        from langchain_adk.planners.task_planner import TaskPlanner

        agent = await Composer.from_yaml_async(yaml_path)
        assert isinstance(agent._planner, TaskPlanner)
        # Task planner should auto-add manage_tasks tool
        assert "manage_tasks" in agent._tools

    @patch("langchain_adk.composer.builders.models._resolve_provider")
    async def test_skills_auto_inject_tools(self, mock_resolve, tmp_path):
        mock_resolve.return_value = lambda **kw: _mock_llm()
        yaml_path = _write_yaml(
            tmp_path,
            """\
            skills:
              - name: summarize
                description: "Summarize text"
                content: "Extract 3-5 key points."
            agents:
              bot:
                type: llm
                instructions: "You are helpful."
            main_agent: bot
            """,
        )
        from langchain_adk.composer import Composer

        agent = await Composer.from_yaml_async(yaml_path)
        assert "list_skills" in agent._tools
        assert "load_skill" in agent._tools
