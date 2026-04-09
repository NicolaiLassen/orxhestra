"""Tool confirmation example — gating dangerous operations with approval.

Shows how to use before_tool_callback with CallContext.request_confirmation()
to pause execution and require approval before running dangerous tools.

Prerequisites::

    pip install orxhestra

Run::

    python examples/tool_confirmation.py
"""

from __future__ import annotations

import asyncio

from langchain_core.tools import tool

from orxhestra import LlmAgent
from orxhestra.events.event import EventType
from orxhestra.tools.call_context import CallContext


@tool
def search_database(query: str) -> str:
    """Search the database for records matching a query."""
    return f"Found 3 records matching '{query}'."


@tool
def delete_records(table: str, condition: str) -> str:
    """Delete records from a database table matching a condition."""
    return f"Deleted records from '{table}' where {condition}."


@tool
def drop_table(table: str) -> str:
    """Drop an entire database table."""
    return f"Table '{table}' dropped."


# Dangerous operations that need human approval
DANGEROUS_TOOLS = {"delete_records", "drop_table"}


async def confirm_dangerous(ctx, tool_name: str, tool_args: dict) -> None:
    """Before-tool callback that gates dangerous operations."""
    if tool_name in DANGEROUS_TOOLS:
        call_ctx = CallContext(ctx)
        print(f"\n  ⚠ CONFIRMATION REQUIRED: {tool_name}({tool_args})")
        answer = input("  Approve? (y/n): ").strip().lower()
        if answer != "y":
            call_ctx.request_confirmation()


async def main() -> None:
    # --- Replace with a real LLM ---
    # from langchain_openai import ChatOpenAI
    # llm = ChatOpenAI(model="gpt-5.4")
    raise NotImplementedError(
        "Replace the llm= line below with a real LangChain chat model "
        "and comment out this raise."
    )

    agent = LlmAgent(
        name="SafeDBAgent",
        llm=llm,  # noqa: F821
        tools=[search_database, delete_records, drop_table],
        before_tool_callback=confirm_dangerous,
        instructions=(
            "You are a database assistant. Use search_database to find records. "
            "Use delete_records or drop_table when explicitly asked. "
            "Always confirm dangerous operations with the user first."
        ),
    )

    print(f"Running agent: {agent.name}\n{'='*40}")

    async for event in agent.astream(
        "Search for all users named 'test', then delete them from the users table."
    ):
        if event.has_tool_calls:
            print(f"[TOOL CALL] {event.tool_name}({event.tool_input})")
        elif event.type == EventType.TOOL_RESPONSE:
            print(f"[TOOL RESULT] {event.text}")
        elif event.is_final_response():
            print(f"\n[ANSWER] {event.text}")


if __name__ == "__main__":
    asyncio.run(main())
