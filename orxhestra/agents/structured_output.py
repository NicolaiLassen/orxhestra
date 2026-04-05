"""StructuredOutputParser — handles Pydantic-schema-based output parsing."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage
from langchain_core.output_parsers import PydanticOutputParser

if TYPE_CHECKING:
    from orxhestra.agents.invocation_context import InvocationContext


class StructuredOutputParser:
    """Parse or extract structured output from LLM responses.

    First attempts to parse the raw text via ``PydanticOutputParser``.
    If that fails, falls back to calling the LLM with
    ``with_structured_output()`` for a dedicated extraction call.

    Parameters
    ----------
    llm : BaseChatModel
        The LangChain chat model (used for the fallback extraction).
    output_schema : type
        Pydantic model class defining the expected structure.
    """

    def __init__(self, llm: BaseChatModel, output_schema: type) -> None:
        self._llm = llm
        self._output_schema = output_schema

    def build_structured_llm(self) -> Any | None:
        """Return the LLM bound with structured output, or ``None``.

        Tries ``json_schema`` then ``json_mode`` as the binding method.

        Returns
        -------
        BaseChatModel or None
            The bound LLM, or ``None`` if binding is unsupported.
        """
        for method in ("json_schema", "json_mode"):
            try:
                return self._llm.with_structured_output(
                    self._output_schema, method=method
                )
            except (NotImplementedError, TypeError, ValueError):
                continue
        return None

    async def parse(
        self,
        answer_text: str | None,
        messages: list[BaseMessage],
        ctx: InvocationContext,
    ) -> Any | None:
        """Try to parse structured output from the answer text.

        Falls back to a dedicated LLM call when text parsing fails.

        Parameters
        ----------
        answer_text : str or None
            The raw text answer from the LLM.
        messages : list[BaseMessage]
            Conversation messages for the fallback extraction call.
        ctx : InvocationContext
            Current invocation context.

        Returns
        -------
        Pydantic model instance or None
            The parsed structured output, or ``None`` if parsing
            fails entirely.
        """
        parser = PydanticOutputParser(pydantic_object=self._output_schema)
        if answer_text:
            try:
                return parser.parse(answer_text)
            except Exception:
                pass
        structured_llm: Any | None = self.build_structured_llm()
        if structured_llm is not None:
            try:
                return await structured_llm.ainvoke(
                    messages, config=ctx.run_config,
                )
            except Exception:
                pass
        return None
