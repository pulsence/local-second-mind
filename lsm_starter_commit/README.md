# lsm (starter refactor)

This is a minimal "starter commit" refactor that:

- Exposes a unified CLI: `python -m lsm ingest` and `python -m lsm query`
- Keeps your existing implementations intact by moving them verbatim into:
  - `lsm/ingest/legacy_ingest.py`
  - `lsm/query/legacy_query.py`

## Run without installing

From the repository root:

```bash
python -m lsm ingest --config config.yaml
python -m lsm query  --config config.yaml
```

## Install as a command

```bash
pip install -e .
lsm ingest --config config.yaml
lsm query  --config config.yaml
```
