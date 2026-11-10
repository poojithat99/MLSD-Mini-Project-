## Agent layer

Two-agent LangGraph (analysis agent + verifier) that turns a validated dispute into a verified, cited
recommendation or an escalation with a reason code. Entry point `backend/agents/graph.py:run_dispute_graph`.
Topology, tools, loop limits, escalation codes, guardrails, LLM configuration and how to run:
see [backend/agents/README.md](backend/agents/README.md).
