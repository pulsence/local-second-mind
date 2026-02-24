# lsm.agents

Description: Agent framework providing runtime, tools, sandbox, memory, scheduler, and built-in agent implementations.
Folder Path: `lsm/agents/`

## Sub Packages

- `lsm.agents.academic`: Academic agents (research, synthesis, curator)
- `lsm.agents.productivity`: Productivity agents (writing)
- `lsm.agents.meta`: Meta-agent orchestration and task-graph models
- `lsm.agents.assistants`: Assistant agent namespace (populated in later phases)
- [lsm.agents.memory](lsm.agents.memory.md): Persistent memory storage with SQLite/PostgreSQL backends and context builders
- [lsm.agents.tools](lsm.agents.tools.md): Tool registry, sandbox enforcement, runner abstraction, and built-in tools

## Modules

- [base.py](../lsm/agents/base.py): BaseAgent and AgentState lifecycle model
- [harness.py](../lsm/agents/harness.py): Runtime action loop, tool execution harness, and per-run summaries
- [interaction.py](../lsm/agents/interaction.py): Thread-safe runtime<->UI interaction channel for permission/clarification prompts
- [models.py](../lsm/agents/models.py): Runtime message/log/response models
- [log_formatter.py](../lsm/agents/log_formatter.py): Log formatting and serialization helpers
- [log_redactor.py](../lsm/agents/log_redactor.py): Secret redaction in logs
- [factory.py](../lsm/agents/factory.py): Agent registry, metadata, and factory function
- [scheduler.py](../lsm/agents/scheduler.py): Recurring schedule engine for agent runs
- [permission_gate.py](../lsm/agents/permission_gate.py): Permission gate for tool execution
