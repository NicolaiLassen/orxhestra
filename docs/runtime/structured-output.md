# Structured Output

Force an agent to return a typed Pydantic object instead of free-form text. Pass `output_schema` to `LlmAgent`.

```python
from pydantic import BaseModel, Field
from langchain_adk import LlmAgent
from langchain_adk.events.event import FinalAnswerEvent

class CompanyAnalysis(BaseModel):
    name: str = Field(description="Company name")
    industry: str = Field(description="Primary industry")
    strengths: list[str] = Field(description="Key strengths")
    risks: list[str] = Field(description="Key risks")
    recommendation: str = Field(description="Buy, Hold, or Sell")
    confidence: float = Field(description="Confidence score 0-1")

agent = LlmAgent(
    name="AnalystAgent",
    llm=llm,
    tools=[get_financials, get_news_sentiment],
    output_schema=CompanyAnalysis,
    instructions="You are a financial analyst.",
)
```

The parsed object is available on `FinalAnswerEvent.structured_output`:

```python
async for event in agent.run("Analyze Apple", ctx=ctx):
    if isinstance(event, FinalAnswerEvent):
        analysis = event.structured_output  # CompanyAnalysis instance
        print(f"{analysis.name}: {analysis.recommendation} ({analysis.confidence:.0%})")
```

## How it works

1. `PydanticOutputParser.get_format_instructions()` is appended to the system prompt
2. When the LLM responds, `PydanticOutputParser.parse()` extracts and validates JSON
3. If direct parsing fails, `with_structured_output()` is used as a fallback
4. Works with streaming (`StreamingMode.SSE`) and multi-agent compositions
