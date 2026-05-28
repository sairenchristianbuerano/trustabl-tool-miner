"""Agent constructor fixtures for the pure-Python agent scanner."""

from agents import Agent, AgentDefinition


# OpenAI-style agent, no guardrails, grants Bash.
support_agent = Agent(
    name="support",
    model="gpt-4o",
    tools=["Bash", "lookup"],
)

# Claude AgentDefinition granting WebSearch.
researcher = AgentDefinition(
    name="researcher",
    tools=["WebSearch"],
)

# Guarded OpenAI agent (should NOT trip missing_guardrails).
guarded = Agent(
    name="guarded",
    model="gpt-4o",
    input_guardrails=[object()],
    tools=["lookup"],
)
