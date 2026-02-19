# lsm.agents

Description: Agent framework providing runtime, tools, sandbox, memory, scheduler, and built-in agent implementations.
Folder Path: `lsm/agents/`

## Sub Packages

- [lsm.agents.memory](lsm.agents.memory.md): Persistent memory storage with SQLite/PostgreSQL backends and context builders
- [lsm.agents.tools](lsm.agents.tools.md): Tool registry, sandbox enforcement, runner abstraction, and built-in tools

## Modules

- [base.py](../lsm/agents/base.py): BaseAgent and AgentState lifecycle model
- [harness.py](../lsm/agents/harness.py): Runtime action loop, tool execution harness, and per-run summaries
- [interaction.py](../lsm/agents/interaction.py): Thread-safe runtime<->UI interaction channel for permission/clarification prompts
- [models.py](../lsm/agents/models.py): Runtime message/log/response models
- [log_formatter.py](../lsm/agents/log_formatter.py): Log formatting and serialization helpers
- [log_redactor.py](../lsm/agents/log_redactor.py): Secret redaction in logs
- [factory.py](../lsm/agents/factory.py): Agent registry and factory function
- [research.py](../lsm/agents/research.py): Built-in research agent implementation
- [writing.py](../lsm/agents/writing.py): Built-in writing agent implementation
- [synthesis.py](../lsm/agents/synthesis.py): Built-in synthesis agent implementation
- [curator.py](../lsm/agents/curator.py): Built-in curator agent implementation
- [scheduler.py](../lsm/agents/scheduler.py): Recurring schedule engine for agent runs
- [task_graph.py](../lsm/agents/task_graph.py): AgentTask/TaskGraph orchestration graph models
- [meta.py](../lsm/agents/meta.py): Built-in meta-agent orchestrator with task graph execution
- [permission_gate.py](../lsm/agents/permission_gate.py): Permission gate for tool execution
