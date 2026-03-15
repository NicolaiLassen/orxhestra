"""AgentTool - wrap a BaseAgent as a callable LangChain tool.

Enables an LlmAgent to delegate work to a sub-agent by calling it as a
tool. The parent agent sends a request string; the child agent runs and
its final answer is returned as the tool result.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from langchain_core.callbacks import AsyncCallbackManagerForToolRun
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from langchain_adk.agents.base_agent import BaseAgent
    from langchain_adk.agents.context import Context


class AgentToolInput(BaseModel):
    """Input schema for AgentTool - a plain request string."""
    request: str = Field(description="The request or task to send to the agent.")


class AgentTool(BaseTool):
    """Wraps a BaseAgent as a LangChain tool.

    When invoked, runs the wrapped agent with the given request and
    returns the final answer text as the tool result. The child
    agent runs with its own event queue.

    Attributes
    ----------
    agent : BaseAgent
        The agent to wrap.
    skip_summarization : bool
        If True, signals the parent to skip LLM summarization of this
        tool's result.
    _ctx : Context, optional
        The parent Context, injected at call time.
    """

    name: str
    description: str
    args_schema: type[BaseModel] = AgentToolInput
    skip_summarization: bool = False

    # Injected at runtime by LlmAgent before tool execution
    _ctx: Any | None = None

    def __init__(self, agent: BaseAgent, *, skip_summarization: bool = False) -> None:
        super().__init__(
            name=agent.name,
            description=agent.description or f"Delegate work to the {agent.name} agent.",
        )
        # Store agent as private attribute to avoid Pydantic field conflicts
        object.__setattr__(self, "_agent", agent)
        object.__setattr__(self, "skip_summarization", skip_summarization)
        object.__setattr__(self, "_ctx", None)

    def inject_context(self, ctx: Context) -> None:
        """Inject the parent invocation context before tool execution."""
        object.__setattr__(self, "_ctx", ctx)

    def _run(self, request: str, **kwargs: Any) -> str:
        """Raise an error - AgentTool is async-only."""
        raise NotImplementedError("Use async ainvoke - AgentTool is async-only.")

    async def _arun(
        self,
        request: str,
        run_manager: AsyncCallbackManagerForToolRun | None = None,
        **kwargs: Any,
    ) -> str:
        """Run the wrapped agent asynchronously and return its final answer.

        Parameters
        ----------
        request : str
            The request or task to send to the wrapped agent.
        run_manager : AsyncCallbackManagerForToolRun, optional
            LangChain callback manager (injected by the framework).

        Returns
        -------
        str
            The wrapped agent's final answer text, or an error message
            if no final answer was produced.
        """
        agent: BaseAgent = object.__getattribute__(self, "_agent")  # type: ignore[assignment]
        ctx: Context | None = object.__getattribute__(self, "_ctx")

        if ctx is None:
            raise RuntimeError(
                f"AgentTool '{self.name}' has no context. "
                "Call inject_context(ctx) before invoking."
            )

        final_answer: str | None = None
        async for event in agent.astream(request, ctx=ctx):
            if event.is_final_response():
                final_answer = event.text

        return final_answer or f"Agent '{agent.name}' produced no final answer."
