"""orxhestra — Multi-Agent AI Framework for Python.

Core agents::

    from orxhestra.agents import LlmAgent, ReActAgent
    from orxhestra.agents import SequentialAgent, ParallelAgent, LoopAgent

Runner::

    from orxhestra.runner import Runner
    from orxhestra.agents import AgentConfig

Planners::

    from orxhestra.planners import TaskPlanner, PlanReActPlanner

Events::

    from orxhestra.events.event import Event, EventType
    from orxhestra.events.event_actions import EventActions

Context::

    from orxhestra.agents import InvocationContext
    from orxhestra.agents import ReadonlyContext, CallbackContext

Sessions::

    from orxhestra.sessions import Session, InMemorySessionService

Tools::

    from orxhestra.tools import function_tool, AgentTool, make_transfer_tool
    from orxhestra.tools import exit_loop_tool, CallContext

Composer::

    from orxhestra.composer import Composer
"""

__version__ = "0.0.62"

from orxhestra.agents import (
    AgentConfig,
    BaseAgent,
    CallbackContext,
    InvocationContext,
    LlmAgent,
    LoopAgent,
    ParallelAgent,
    ReActAgent,
    ReadonlyContext,
    SequentialAgent,
)
from orxhestra.agents.a2a_agent import A2AAgent
from orxhestra.composer import Composer
from orxhestra.decorators.deprecation import (
    OrxhestraDeprecationWarning,
    deprecated,
    deprecated_param,
)
from orxhestra.events.event import Event, EventType
from orxhestra.events.event_actions import EventActions
from orxhestra.models.part import (
    Content,
    DataPart,
    FilePart,
    TextPart,
    ThinkingPart,
    ToolCallPart,
    ToolResponsePart,
)
from orxhestra.planners import BasePlanner, PlanReActPlanner, TaskPlanner
from orxhestra.runner import Runner
from orxhestra.sessions.base_session_service import BaseSessionService
from orxhestra.sessions.database_session_service import DatabaseSessionService
from orxhestra.sessions.in_memory_session_service import InMemorySessionService
from orxhestra.sessions.session import Session

__all__ = [
    # Agents
    "BaseAgent",
    "LlmAgent",
    "ReActAgent",
    "SequentialAgent",
    "ParallelAgent",
    "LoopAgent",
    "A2AAgent",
    # Runner + config
    "Runner",
    "AgentConfig",
    # Planners
    "BasePlanner",
    "TaskPlanner",
    "PlanReActPlanner",
    # Events
    "Event",
    "EventActions",
    "EventType",
    # Content / Parts
    "Content",
    "TextPart",
    "DataPart",
    "FilePart",
    "ThinkingPart",
    "ToolCallPart",
    "ToolResponsePart",
    # Context
    "InvocationContext",
    "ReadonlyContext",
    "CallbackContext",
    # Sessions
    "BaseSessionService",
    "DatabaseSessionService",
    "Session",
    "InMemorySessionService",
    # Composer
    "Composer",
    # Decorators
    "deprecated",
    "deprecated_param",
    "OrxhestraDeprecationWarning",
]
