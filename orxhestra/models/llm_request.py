"""LlmRequest - a structured wrapper around the LangChain model call inputs.

Collects everything needed for a single ``BaseChatModel.ainvoke()`` call -
messages, system instruction, tools, output schema, and model name - into
one inspectable Pydantic model.

Agents build an ``LlmRequest`` and hand it to a ``BaseLlm`` backend rather
than calling LangChain directly. This keeps agent logic independent of
LangChain internals and makes the full request loggable and testable.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class LlmRequest(BaseModel):
    """Structured input for a single LangChain model call.

    Collect all context needed to call ``BaseChatModel.ainvoke()``:
    the message history, system instruction, tools, output schema, and
    model identifier. Agents build one of these per LLM call so the
    full request is inspectable and loggable.

    Attributes
    ----------
    model : str, optional
        Override the model name for this specific call. If omitted, the
        LlmAgent uses its configured model.
    system_instruction : str
        System prompt injected as the first message. May be the result of
        an instruction provider callable.
    messages : list[BaseMessage]
        The conversation history to send to the model.
    tools : list[BaseTool]
        LangChain tools available to the model for this call.
    tools_dict : dict[str, BaseTool]
        Name-keyed lookup of the same tools. Populated automatically from
        ``tools`` by agents; callers rarely need to set this directly.
    output_schema : type, optional
        Pydantic model class for structured output. When set, the agent
        binds ``.with_structured_output()`` to the model.
    config : dict[str, Any]
        Extra keyword arguments forwarded to the model invocation.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    model: str | None = None
    system_instruction: str = ""
    messages: list[Any] = Field(default_factory=list)
    tools: list[Any] = Field(default_factory=list)
    tools_dict: dict[str, Any] = Field(default_factory=dict, exclude=True)
    output_schema: Any | None = None
    config: dict[str, Any] = Field(default_factory=dict)

    def add_tool(self, tool: Any) -> None:
        """Register a tool on this request.

        Parameters
        ----------
        tool : BaseTool
            The LangChain tool to add.
        """
        if tool not in self.tools:
            self.tools.append(tool)
            self.tools_dict[tool.name] = tool

    def has_tools(self) -> bool:
        """Return True if at least one tool is registered.

        Returns
        -------
        bool
        """
        return bool(self.tools)
