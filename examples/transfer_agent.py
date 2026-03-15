"""Transfer agent example - routing between specialist agents.

Demonstrates:
  - TransferTool for agent-to-agent handoff
  - Triage agent routes requests to the right specialist
  - Enum-constrained agent names prevent hallucination
"""

from __future__ import annotations

import asyncio

from langchain_core.tools import tool

from langchain_adk import LlmAgent
from langchain_adk.events.event import Event, EventType
from langchain_adk.tools.transfer_tool import make_transfer_tool


# --- Specialist tools ---

@tool
def lookup_order(order_id: str) -> str:
    """Look up an order by ID."""
    return f"Order {order_id}: 2x Widget Pro, shipped 2024-01-15, tracking #ABC123."


@tool
def check_billing(customer_id: str) -> str:
    """Check billing status for a customer."""
    return f"Customer {customer_id}: Plan=Pro, next billing=2024-02-01, balance=$0.00."


@tool
def search_docs(query: str) -> str:
    """Search the knowledge base for technical documentation."""
    return f"Doc result for '{query}': To reset your API key, go to Settings > API > Regenerate."


async def main() -> None:
    # --- Replace with a real LLM ---
    # from langchain_openai import ChatOpenAI
    # llm = ChatOpenAI(model="gpt-5.4")
    raise NotImplementedError(
        "Replace the llm= line below with a real LangChain chat model "
        "and comment out this raise."
    )

    # --- Specialist agents ---
    sales_agent = LlmAgent(
        name="SalesAgent",
        llm=llm,  # noqa: F821
        tools=[lookup_order],
        description="Handles order inquiries, shipping, and returns.",
        instructions="You are a sales specialist. Help with order-related questions.",
    )

    billing_agent = LlmAgent(
        name="BillingAgent",
        llm=llm,  # noqa: F821
        tools=[check_billing],
        description="Handles billing, payments, and subscription questions.",
        instructions="You are a billing specialist. Help with payment and subscription questions.",
    )

    support_agent = LlmAgent(
        name="SupportAgent",
        llm=llm,  # noqa: F821
        tools=[search_docs],
        description="Handles technical support and documentation questions.",
        instructions="You are a technical support specialist. Help with product and API questions.",
    )

    # --- Triage agent with transfer tool ---
    specialists = [sales_agent, billing_agent, support_agent]
    transfer_tool = make_transfer_tool(specialists)

    triage_agent = LlmAgent(
        name="TriageAgent",
        llm=llm,  # noqa: F821
        tools=[transfer_tool],
        instructions=(
            "You are a customer service triage agent. "
            "Determine which specialist can best help the customer, "
            "then transfer them using the transfer_to_agent tool.\n\n"
            "- SalesAgent: orders, shipping, returns\n"
            "- BillingAgent: payments, subscriptions, invoices\n"
            "- SupportAgent: technical issues, API, documentation"
        ),
    )

    queries = [
        "Where is my order #12345?",
        "How do I reset my API key?",
        "When is my next billing date? Customer ID: C-789",
    ]

    for query in queries:
        print(f"\nCustomer: {query}")
        print("-" * 50)

        async for event in triage_agent.astream(query):
            if event.has_tool_calls:
                print(f"  [TOOL] {event.tool_name}({event.tool_input})")
            elif event.type == EventType.TOOL_RESPONSE:
                print(f"  [RESULT] {event.text}")
            elif event.is_final_response():
                print(f"  [ANSWER] {event.text}")


if __name__ == "__main__":
    asyncio.run(main())
