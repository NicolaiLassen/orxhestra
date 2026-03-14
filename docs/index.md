---
hide:
  - navigation
  - toc
---

<div class="hero-section" markdown>

# Build agents with <span class="gradient">LangChain + structure</span>

Async event-streaming agents, composable hierarchies, session management, planners, and first-class A2A & MCP integration. No LangGraph required.

[Get Started](getting-started/quickstart.md){ .md-button .md-button--primary }
[View on GitHub](https://github.com/NicolaiLassen/langchain-adk){ .md-button }

<div class="install-command">
<code>pip install langchain-adk</code>
</div>

</div>

---

<div class="feature-grid" markdown>

<div class="feature-card" markdown>
### :zap: Async Event Streaming
Every agent is an async generator yielding typed events — thoughts, tool calls, results, streaming chunks. Built for real-time UIs.
</div>

<div class="feature-card" markdown>
### :deciduous_tree: Composable Hierarchies
Nest agents freely. Chain them sequentially, run in parallel, loop until done, or wrap as tools. Plain Python asyncio.
</div>

<div class="feature-card" markdown>
### :brain: Planners
Per-turn planning hooks inject dynamic instructions and post-process responses. Built-in PlanReAct and TaskPlanner included.
</div>

<div class="feature-card" markdown>
### :floppy_disk: Session Management
Pluggable session service persists events and state deltas automatically. In-memory included, bring your own database.
</div>

<div class="feature-card" markdown>
### :wrench: Rich Tool System
Function tools, agent-as-tool, transfer handoffs, exit loop, long-running tools, and tool confirmation — all built in.
</div>

<div class="feature-card" markdown>
### :mag: Full Observability
First-class tracing via RunConfig callbacks. Works with Langfuse, LangSmith, or any LangChain callback handler.
</div>

</div>

---

## Works with any LangChain LLM

=== "OpenAI"

    ```bash
    pip install langchain-adk[openai]
    ```

=== "Anthropic"

    ```bash
    pip install langchain-adk[anthropic]
    ```

=== "Google Gemini"

    ```bash
    pip install langchain-adk[google]
    ```

=== "Any BaseChatModel"

    ```bash
    pip install langchain-adk
    # + your preferred langchain-* provider package
    ```
