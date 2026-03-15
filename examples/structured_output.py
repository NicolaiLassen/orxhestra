"""Structured output example - agent returns typed Pydantic objects.

Demonstrates:
  - output_schema parameter for type-safe structured responses
  - Pydantic model automatically parsed from LLM output
  - Combining tools with structured final output via event.data
"""

from __future__ import annotations

import asyncio

from pydantic import BaseModel, Field
from langchain_core.tools import tool

from langchain_adk import LlmAgent
from langchain_adk.events.event import Event, EventType


# --- Structured output schema ---

class CompanyAnalysis(BaseModel):
    """Structured analysis of a company."""
    name: str = Field(description="Company name")
    industry: str = Field(description="Primary industry")
    strengths: list[str] = Field(description="Key strengths")
    risks: list[str] = Field(description="Key risks")
    recommendation: str = Field(description="Buy, Hold, or Sell")
    confidence: float = Field(description="Confidence score 0-1")


# --- Research tools ---

@tool
def get_financials(company: str) -> str:
    """Get financial data for a company."""
    data = {
        "apple": "Revenue: $383B, Net Income: $97B, P/E: 28.5, Debt/Equity: 1.8",
        "tesla": "Revenue: $97B, Net Income: $15B, P/E: 62.3, Debt/Equity: 0.7",
        "microsoft": "Revenue: $236B, Net Income: $88B, P/E: 35.2, Debt/Equity: 0.4",
    }
    return data.get(company.lower(), f"No financial data for {company}.")


@tool
def get_news_sentiment(company: str) -> str:
    """Get recent news sentiment for a company."""
    data = {
        "apple": "Sentiment: Positive. AI features driving iPhone upgrades. Services growing 15% YoY.",
        "tesla": "Sentiment: Mixed. Strong EV demand but increasing competition. Margins under pressure.",
        "microsoft": "Sentiment: Very Positive. Azure growth accelerating. Copilot adoption strong.",
    }
    return data.get(company.lower(), f"No sentiment data for {company}.")


async def main() -> None:
    # --- Replace with a real LLM ---
    # from langchain_openai import ChatOpenAI
    # llm = ChatOpenAI(model="gpt-5.4")
    raise NotImplementedError(
        "Replace the llm= line below with a real LangChain chat model "
        "and comment out this raise."
    )

    # Just pass the Pydantic schema — the agent handles JSON parsing automatically
    agent = LlmAgent(
        name="AnalystAgent",
        llm=llm,  # noqa: F821
        tools=[get_financials, get_news_sentiment],
        output_schema=CompanyAnalysis,
        instructions=(
            "You are a financial analyst. Research the company using the available "
            "tools, then provide your analysis. Return ONLY a JSON object, no other text."
        ),
    )

    companies = ["Apple", "Tesla"]

    for company in companies:
        print(f"\nAnalyzing: {company}")
        print("=" * 50)

        async for event in agent.astream(f"Analyze {company}"):
            if event.has_tool_calls:
                print(f"  [TOOL] {event.tool_name}({event.tool_input})")
            elif event.type == EventType.TOOL_RESPONSE:
                print(f"  [DATA] {(event.text or '')[:60]}...")
            elif event.is_final_response():
                data = event.data  # dict from DataPart
                if data:
                    print(f"\n  Company:        {data['name']}")
                    print(f"  Industry:       {data['industry']}")
                    print(f"  Strengths:      {', '.join(data['strengths'])}")
                    print(f"  Risks:          {', '.join(data['risks'])}")
                    print(f"  Recommendation: {data['recommendation']}")
                    print(f"  Confidence:     {data['confidence']:.0%}")
                else:
                    print(f"\n  [RAW OUTPUT] {event.text}")


if __name__ == "__main__":
    asyncio.run(main())
