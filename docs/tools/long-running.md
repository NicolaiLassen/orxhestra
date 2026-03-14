# Long-Running Tools

For tools that take significant time (deployments, data processing), use `LongRunningFunctionTool`. It instructs the LLM not to re-invoke the tool while pending:

```python
from langchain_adk.tools import LongRunningFunctionTool

async def deploy_service(env: str) -> str:
    """Deploy the service to the given environment."""
    await some_long_operation(env)
    return f"Deployed to {env}"

tool = LongRunningFunctionTool(deploy_service)
agent = LlmAgent(name="deployer", llm=llm, tools=[tool.as_tool()])
```

The tool description is automatically appended with:

> NOTE: This is a long-running operation. Do not call this tool again if it has already returned some intermediate or pending status.
