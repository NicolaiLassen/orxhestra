"""Human-in-the-loop example - agent asks the human for input.

Demonstrates:
  - Custom `ask_human` tool that lets the agent ask the user questions
  - Agent gathers missing info interactively before taking action
  - Combines human input with regular tools
"""

from __future__ import annotations

import asyncio

from langchain_core.tools import tool

from langchain_adk import LlmAgent
from langchain_adk.events.event import Event, EventType


# --- Ask-human tool ---

@tool
def ask_human(question: str) -> str:
    """Ask the human user a question and return their response.

    Use this tool whenever you need clarification, confirmation,
    or additional information from the user before proceeding.
    """
    print(f"\n  Agent asks: {question}")
    response = input("  Your answer: ")
    return response.strip()


# --- Action tools ---

@tool
def book_meeting(title: str, date: str, attendees: str) -> str:
    """Book a meeting with the given details."""
    return f"Meeting '{title}' booked on {date} with {attendees}."


@tool
def send_email(to: str, subject: str, body: str) -> str:
    """Send an email to a recipient."""
    return f"Email sent to {to} with subject '{subject}'."


async def main() -> None:
    # --- Replace with a real LLM ---
    # from langchain_openai import ChatOpenAI
    # llm = ChatOpenAI(model="gpt-5.4")
    raise NotImplementedError(
        "Replace the llm= line below with a real LangChain chat model "
        "and comment out this raise."
    )

    agent = LlmAgent(
        name="AssistantWithHumanInput",
        llm=llm,  # noqa: F821
        tools=[ask_human, book_meeting, send_email],
        instructions=(
            "You are a helpful assistant that can book meetings and send emails. "
            "When you don't have enough information to complete a task, use the "
            "ask_human tool to ask the user for the missing details. "
            "Always confirm important actions with the user before executing them."
        ),
    )

    print("Human-in-the-Loop Agent")
    print("The agent will ask you questions when it needs more info.\n")

    query = "Book a meeting about the Q2 roadmap"

    print(f"User: {query}\n")

    async for event in agent.astream(query):
        if event.has_tool_calls:
            if event.tool_name != "ask_human":
                print(f"\n[TOOL CALL] {event.tool_name}({event.tool_input})")
        elif event.type == EventType.TOOL_RESPONSE:
            if event.tool_name != "ask_human":
                print(f"[TOOL RESULT] {event.text}")
        elif event.is_final_response():
            print(f"\n[ANSWER] {event.text}")


if __name__ == "__main__":
    asyncio.run(main())
