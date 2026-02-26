# lsm.agents

Description: Agent framework providing runtime, tools, sandbox, memory, scheduler, and built-in agent implementations.
Folder Path: `lsm/agents/`

## Sub Packages

- `lsm.agents.academic`: Academic agents (research, synthesis, curator)
- `lsm.agents.productivity`: Productivity agents (general, librarian, manuscript editor, writing)
- `lsm.agents.meta`: Meta-agent orchestration — `MetaAgent` (parallel execution, sandbox monotonicity, `final_result.md`), `AssistantMetaAgent` (validation passes, action recommendations), `TaskGraph`/`ParallelGroup` (parallel planning with dependency gates and deterministic ordering)
- `lsm.agents.assistants`: Assistant agent implementations (assistant, email_assistant, calendar_assistant, news_assistant)
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
- [workspace.py](../lsm/agents/workspace.py): Per-agent workspace layout helpers
- [scheduler.py](../lsm/agents/scheduler.py): Recurring schedule engine for agent runs
- [permission_gate.py](../lsm/agents/permission_gate.py): Permission gate for tool execution
- [phase.py](../lsm/agents/phase.py): PhaseResult dataclass for bounded phase execution

## Phase Execution (v0.7.1+)

### PhaseResult

The `PhaseResult` dataclass is the return type from `_run_phase()`. It carries operational output from one bounded execution phase but no financial or accounting data:

```python
from lsm.agents import PhaseResult

@dataclass(frozen=True)
class PhaseResult:
    final_text: str          # last LLM response text for this phase
    tool_calls: list[dict]   # all tool calls made during this phase
    stop_reason: str         # "done" | "max_iterations" | "budget_exhausted" | "stop_requested"
```

### AgentHarness.run_bounded()

The `run_bounded()` method drives at most `max_iterations` of the LLM + tool loop, then returns a `PhaseResult`. It introduces multi-context support via `context_label`:

```python
def run_bounded(
    self,
    user_message: str = "",
    tool_names: Optional[list[str]] = None,
    max_iterations: int = 10,
    continue_context: bool = True,
    context_label: Optional[str] = None,
    direct_tool_calls: Optional[list[dict]] = None,
) -> PhaseResult:
```

**Context label design**: The harness stores `_context_histories: dict[Optional[str], list]`, keyed by label.
- `context_label=None` (default) accesses the primary unnamed context
- `context_label="subtopic_A"` creates or resumes a named context
- Agents switch contexts simply by passing a different label on the next `_run_phase()` call
- `continue_context=False` resets the history for the current label only

**Tool-only mode**: When `direct_tool_calls` is provided, the harness executes tools directly without making any LLM call. Budget checks do NOT apply in this mode.

### BaseAgent._run_phase()

The `_run_phase()` method is the only method agents should use for LLM and tool activity. It creates the shared harness on first call and reuses it for subsequent calls:

```python
def _run_phase(
    self,
    system_prompt: str = "",
    user_message: str = "",
    tool_names: Optional[list[str]] = None,
    max_iterations: int = 10,
    continue_context: bool = True,
    context_label: Optional[str] = None,
    direct_tool_calls: Optional[list[dict]] = None,
) -> PhaseResult:
```

### BaseAgent Workspace Accessors

BaseAgent provides workspace path accessor methods:

```python
def _workspace_root(self) -> Path       # Returns agent workspace root, creates directories if absent
def _artifacts_dir(self) -> Path        # Returns artifacts/ subdirectory
def _logs_dir(self) -> Path             # Returns logs/ subdirectory
def _memory_dir(self) -> Path           # Returns memory/ subdirectory
def _artifact_filename(self, name: str, suffix: str = ".md") -> str  # Generates timestamped filename
```

## Tools (v0.7.1+)

### QueryKnowledgeBaseTool

The `query_knowledge_base` tool replaces the deprecated `query_embeddings` tool. It uses the full query pipeline including embedding search, reranking, and LLM synthesis:

```python
from lsm.agents.tools import QueryKnowledgeBaseTool

tool = QueryKnowledgeBaseTool(
    config=config,
    embedder=embedder,
    collection=collection,
)
result = tool.execute({"query": "your question", "top_k": 5})
# Returns JSON: {"answer": "...", "sources_display": "...", "candidates": [...]}
```

**Output format**: Returns a JSON object with `answer` (LLM-generated response), `sources_display` (formatted source list), and `candidates` (list of retrieved chunks with id, score, and text).
