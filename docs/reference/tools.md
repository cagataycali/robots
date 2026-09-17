---
description: Every agent-callable tool in the package - what it does, what it takes, what it refuses.
---

# Tool reference

Generated from `strands_robots/tools/` by `docs/hooks/tool_reference.py` at build - do not edit.

Every tool returns the same envelope, `{"status": ..., "content": [{"text": "..."}]}`, read through
`result["content"][0]["text"]` and never through invented keys - see the
[tool result contract](../contracts.md). A tool marked as taking the agent's tool context can stop
and ask an operator before it acts.

{{tool_reference}}
