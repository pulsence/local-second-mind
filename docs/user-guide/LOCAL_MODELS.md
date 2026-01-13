# Local Models (Ollama)

This guide explains how to use local LLMs with LSM via Ollama.

## Overview

LSM can run reranking, synthesis, and tagging against a local model when
`llm.provider` is set to `local`. The provider uses Ollama's HTTP API.

## Prerequisites

1. Install Ollama: https://ollama.com/
2. Start the Ollama server (default: `http://localhost:11434`)
3. Pull a model you want to use (examples below)

## Recommended Models

General purpose:

- `llama2`
- `mistral`
- `llama3`

Smaller/faster:

- `phi3`
- `gemma`

Pick a model that fits your machine and latency budget.

## Pulling Models

Example commands:

```bash
ollama pull llama2
ollama pull mistral
ollama pull llama3
```

## Configuration

Basic config:

```json
{
  "llm": {
    "provider": "local",
    "model": "llama2",
    "base_url": "http://localhost:11434",
    "temperature": 0.7,
    "max_tokens": 2000
  }
}
```

Notes:

- `base_url` defaults to `http://localhost:11434` if omitted.
- Local providers do not require an API key.

### Environment Variables

- `OLLAMA_BASE_URL` can be used instead of `llm.base_url`.

## Usage

Run queries as usual:

```bash
lsm query "What is the retention policy for project X?"
```

In the REPL, you can view provider health:

```bash
lsm
> /provider-status
```

## Performance Considerations

Local models are slower than hosted APIs, especially for large contexts.

Tips:

- Use smaller models for faster reranking and tagging.
- Reduce `query.k` and `query.k_rerank` to shrink context size.
- Lower `max_tokens` to bound response length.
- Consider running Ollama with a GPU if available.

## Troubleshooting

### Connection Errors

If LSM cannot reach Ollama:

1. Confirm Ollama is running.
2. Check `base_url` and ensure the port is correct.
3. Try `curl http://localhost:11434/api/tags` to verify the server.

### Model Not Found

If Ollama returns "model not found":

1. Pull the model with `ollama pull <model>`.
2. Verify the exact model name in your config.

### Low-Quality Output

Local models may produce weaker citations or noisy output.

Suggestions:

- Try a stronger model (e.g., `llama3` or `mistral`).
- Reduce temperature for more deterministic responses.
- Increase chunk size to give more context per source.

## Notes on Reranking

Local reranking relies on JSON output parsing. Smaller models can struggle
to produce valid JSON. If reranking quality is low:

- Switch `rerank_strategy` to `lexical` or `hybrid`.
- Use a stronger local model for reranking only via per-feature overrides.

Example override:

```json
"llm": {
  "provider": "local",
  "model": "llama2",
  "ranking": {
    "model": "mistral"
  }
}
```

## Security and Privacy

Local models keep your data on your machine. No data is sent to external
providers when using `llm.provider = local`.
