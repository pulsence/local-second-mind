#!/usr/bin/env python3
"""
query.py — retrieve + rerank + call OpenAI Responses API + produce cited output
Aligned to ingest.py (local sentence-transformers embeddings + Chroma persistent).

Key alignment with ingest.py:
- Query embedding via SentenceTransformer (same embed_model, normalize_embeddings=True).
- Chroma collection name default: "local_kb"
- Metadata keys expected: source_path, source_name, ext, mtime_ns, file_hash, chunk_index, ingested_at
- Chunk IDs are opaque (sp_hash:file_hash:chunk_index), but citations rely on metadata.

Env:
- OPENAI_API_KEY (required for rerank/answer steps)

Usage:
  python query.py "What have I written about X?" --config config.json
  python query.py "..." --config config.yaml --k 12 --k_rerank 6
  python query.py "..." --config config.json --no_rerank
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

import chromadb
from chromadb.config import Settings

from sentence_transformers import SentenceTransformer

from openai import OpenAI, OpenAIError, RateLimitError


# -----------------------------
# Defaults (must match ingest.py)
# -----------------------------
DEFAULT_COLLECTION = "local_kb"
DEFAULT_EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


# -----------------------------
# Data structures
# -----------------------------
@dataclass
class Candidate:
    cid: str
    text: str
    meta: Dict[str, Any]
    distance: Optional[float] = None


# -----------------------------
# Chroma
# -----------------------------
def chroma_collection(persist_dir: Path, name: str):
    client = chromadb.PersistentClient(
        path=str(persist_dir),
        settings=Settings(anonymized_telemetry=False),
    )
    return client.get_or_create_collection(name=name)


# -----------------------------
# Embeddings (local, matches ingest)
# -----------------------------
def build_embedder(model_name: str, device: str) -> SentenceTransformer:
    return SentenceTransformer(model_name, device=device)


def embed_query(model: SentenceTransformer, text: str, batch_size: int) -> List[float]:
    # Match ingest.py: normalize_embeddings=True
    vec = model.encode([text], batch_size=batch_size, show_progress_bar=False, normalize_embeddings=True)
    return vec[0].tolist()


# -----------------------------
# Retrieve
# -----------------------------
def is_insufficient_quota_error(e: Exception) -> bool:
    # openai-python surfaces this as RateLimitError with error.code == "insufficient_quota"
    if isinstance(e, RateLimitError):
        try:
            # e.response.json() exists in newer stacks; be defensive
            data = getattr(e, "response", None)
            if data is not None:
                j = data.json()
                return j.get("error", {}).get("code") == "insufficient_quota"
        except Exception:
            pass
        # If we can't inspect, still treat RateLimitError as retryable/degradable
        return True
    return isinstance(e, OpenAIError)

def retrieve(collection, query_embedding: List[float], k: int) -> List[Candidate]:
    res = collection.query(
        query_embeddings=[query_embedding],
        n_results=k,
        include=["documents", "metadatas", "distances"],
    )

    ids = (res.get("ids") or [[]])[0]
    docs = (res.get("documents") or [[]])[0]
    metas = (res.get("metadatas") or [[]])[0]
    dists = (res.get("distances") or [[]])[0]

    out: List[Candidate] = []
    for idx, cid in enumerate(ids):
        doc = docs[idx] if idx < len(docs) else ""
        meta = metas[idx] if idx < len(metas) else {}
        dist = dists[idx] if idx < len(dists) else None
        out.append(Candidate(cid=str(cid), text=(doc or ""), meta=(meta or {}), distance=dist))
    return out


# -----------------------------
# Rerank (Responses API)
# -----------------------------
def openai_client(api_key: Optional[str] = None) -> OpenAI:
    if api_key:
        return OpenAI(api_key=api_key)
    return OpenAI()


def llm_rerank(
    client: OpenAI,
    question: str,
    candidates: List[Candidate],
    model: str,
    top_n: int,
) -> List[Candidate]:
    if not candidates:
        return []

    # ... build items/payload as you already do ...

    try:
        resp = client.responses.create(
            model=model,
            reasoning={"effort": "low"},
            instructions=instructions,
            input=[{"role": "user", "content": json.dumps(payload)}],
        )
    except Exception as e:
        # Degrade gracefully: return top_n by vector similarity
        if is_insufficient_quota_error(e):
            return candidates[:top_n]
        return candidates[:top_n]

    raw = (resp.output_text or "").strip()
    try:
        data = json.loads(raw)
        ranking = data.get("ranking", [])
        chosen: List[Candidate] = []
        seen = set()
        for r in ranking:
            idx = int(r["index"])
            if 0 <= idx < len(candidates) and idx not in seen:
                chosen.append(candidates[idx])
                seen.add(idx)
            if len(chosen) >= top_n:
                break
        if len(chosen) < min(top_n, len(candidates)):
            for i, c in enumerate(candidates):
                if i not in seen:
                    chosen.append(c)
                if len(chosen) >= top_n:
                    break
        return chosen
    except Exception:
        return candidates[:top_n]


# -----------------------------
# Answer with citations
# -----------------------------
def build_sources_block(candidates: List[Candidate]) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Builds:
    - context block for the model (sources prefixed [S1], [S2], ...)
    - sources list for user display
    """
    sources_for_model: List[str] = []
    sources_for_user: List[Dict[str, Any]] = []

    for i, c in enumerate(candidates, start=1):
        label = f"S{i}"
        m = c.meta or {}

        source_path = m.get("source_path", "unknown")
        source_name = m.get("source_name")
        chunk_index = m.get("chunk_index")
        ext = m.get("ext")
        mtime_ns = m.get("mtime_ns")
        file_hash = m.get("file_hash")
        ingested_at = m.get("ingested_at")

        locator_bits = []
        if source_name:
            locator_bits.append(f"name={source_name}")
        if chunk_index is not None:
            locator_bits.append(f"chunk_index={chunk_index}")
        if ext:
            locator_bits.append(f"ext={ext}")
        if mtime_ns is not None:
            locator_bits.append(f"mtime_ns={mtime_ns}")
        if file_hash:
            locator_bits.append(f"file_hash={file_hash}")
        if ingested_at:
            locator_bits.append(f"ingested_at={ingested_at}")

        locator = " ".join(locator_bits)
        header = f"[{label}] {source_path}"
        if locator:
            header += f" ({locator})"

        sources_for_model.append(f"{header}\n{(c.text or '').strip()}\n")
        sources_for_user.append(
            {
                "label": label,
                "source_path": source_path,
                "source_name": source_name,
                "chunk_index": chunk_index,
                "ext": ext,
                "mtime_ns": mtime_ns,
                "file_hash": file_hash,
                "ingested_at": ingested_at,
            }
        )

    return "\n\n".join(sources_for_model), sources_for_user

def local_fallback_answer(question: str, chosen: List[Candidate], max_chars: int = 1200) -> str:
    """
    Minimal offline fallback: returns the top passages with citations.
    This is not a true synthesis, but it is useful when OpenAI is unavailable.
    """
    lines = [
        "OpenAI is unavailable (quota/credentials). Showing the most relevant excerpts instead.",
        "",
        f"Question: {question}",
        "",
        "Top excerpts:"
    ]
    for i, c in enumerate(chosen, start=1):
        label = f"S{i}"
        excerpt = (c.text or "").strip()
        if len(excerpt) > max_chars:
            excerpt = excerpt[: max_chars - 50] + "\n...[truncated]..."
        meta = c.meta or {}
        src = meta.get("source_path", "unknown")
        chunk_index = meta.get("chunk_index", "NA")
        lines.append(f"\n[{label}] {src} (chunk_index={chunk_index})\n{excerpt}")
    return "\n".join(lines)

def answer_with_citations(
    client: OpenAI,
    question: str,
    candidates: List[Candidate],
    model: str,
) -> Tuple[str, List[Dict[str, Any]]]:
    context_block, sources_list = build_sources_block(candidates)

    instructions = (
        "Answer the user's question using ONLY the provided sources.\n"
        "Citation rules:\n"
        "- Whenever you make a claim supported by a source, cite inline like [S1] or [S2].\n"
        "- If multiple sources support a sentence, include multiple citations.\n"
        "- Do not fabricate citations.\n"
        "- If the sources are insufficient, say so and specify what is missing.\n"
        "Style: concise, structured, directly responsive."
    )

    user_content = (
        f"Question:\n{question}\n\n"
        f"Sources:\n{context_block}\n\n"
        "Write the answer with inline citations."
    )

    try:
        resp = client.responses.create(
            model=model,
            reasoning={"effort": "medium"},
            instructions=instructions,
            input=[{"role": "user", "content": user_content}],
        )
        return (resp.output_text or "").strip(), sources_list
    except Exception as e:
        if is_insufficient_quota_error(e):
            return local_fallback_answer(question, candidates), sources_list
        return local_fallback_answer(question, candidates), sources_list


def format_sources(sources: List[Dict[str, Any]]) -> str:
    lines = ["", "Sources:"]
    grouped: Dict[str, Dict[str, Any]] = {}
    order: List[str] = []

    for s in sources:
        path = (s.get("source_path") or "unknown").strip()
        label = (s.get("label") or "").strip()
        name = (s.get("source_name") or Path(path).name or "unknown").strip()

        if path not in grouped:
            grouped[path] = {
                "name": name,
                "labels": [],
            }
            order.append(path)

        if label and label not in grouped[path]["labels"]:
            grouped[path]["labels"].append(label)

    for path in order:
        entry = grouped[path]
        labels = " ".join(f"[{lbl}]" for lbl in entry["labels"])
        lines.append(f"- {labels} {entry['name']} — {path}")

    return "\n".join(lines)


# -----------------------------
# CLI
# -----------------------------
def load_config(path: Path) -> Dict[str, Any]:
    """
    Load YAML or JSON config file (same as ingest.py).
    """
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")

    suffix = path.suffix.lower()
    raw = path.read_text(encoding="utf-8")

    if suffix in {".yaml", ".yml"}:
        return yaml.safe_load(raw) or {}
    if suffix == ".json":
        return json.loads(raw)

    raise ValueError("Config must be .yaml/.yml or .json")


def normalize_config(cfg: Dict[str, Any], cfg_path: Path) -> Dict[str, Any]:
    """
    Normalize and apply defaults for *all* query-time settings.

    Expected (optional) config keys:
      persist_dir: str                 (default ".chroma")
      collection: str                  (default DEFAULT_COLLECTION)
      embed_model: str                 (default DEFAULT_EMBED_MODEL)
      device: str                      (default "cpu")
      batch_size: int                  (default 32)

      openai:
        api_key: str                   (required from user)
      
      query:
        k: int                         (default 12)
        k_rerank: int                  (default 6)
        no_rerank: bool                (default False)
        model: str                     (default "gpt-5.2")
    """
    out: Dict[str, Any] = {}

    # ---- shared with ingest.py ----
    out["collection"] = str(cfg.get("collection", DEFAULT_COLLECTION))
    out["embed_model"] = str(cfg.get("embed_model", DEFAULT_EMBED_MODEL))
    out["device"] = str(cfg.get("device", "cpu"))
    out["batch_size"] = int(cfg.get("batch_size", 32))

    persist_raw = cfg.get("persist_dir", ".chroma")
    out["persist_dir"] = (cfg_path.parent / persist_raw).resolve()

    # ---- OpenAI ----
    ocfg = cfg.get("openai", {})
    out["openai_api_key"] = ocfg.get("api_key", None)

    # ---- query-specific ----
    qcfg = cfg.get("query", {})

    # Number of chunks retrieved from Chroma before reranking.
    out["k"] = int(qcfg.get("k", 12))
    # Number of chunks kept after LLM reranking and passed to the answer step.
    out["k_rerank"] = int(qcfg.get("k_rerank", 6))
    # If true, skips the reranking step entirely and uses vector similarity order.
    out["no_rerank"] = bool(qcfg.get("no_rerank", False))
    out["model"] = str(qcfg.get("model", "gpt-5.2"))

    return out

def parse_args() -> argparse.Namespace:
    """
    CLI is intentionally minimal.
    All tunables live in the config file.
    """
    p = argparse.ArgumentParser(description="Local KB query (config-driven)")
    p.add_argument(
        "--config",
        default="config.json",
        help="Path to YAML/JSON config file (single source of truth)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # Load + normalize config
    cfg_path = Path(args.config).expanduser().resolve()
    cfg_raw = load_config(cfg_path)
    cfg = normalize_config(cfg_raw, cfg_path)

    # Unpack config
    persist_dir: Path = cfg["persist_dir"]
    collection_name: str = cfg["collection"]
    embed_model_name: str = cfg["embed_model"]
    device: str = cfg["device"]
    batch_size: int = cfg["batch_size"]

    k: int = cfg["k"]
    k_rerank: int = cfg["k_rerank"]
    no_rerank: bool = cfg["no_rerank"]
    model: str = cfg["model"]


    # Build embedder + Chroma collection
    embedder = build_embedder(embed_model_name, device=device)
    col = chroma_collection(persist_dir, collection_name)

    # Build OpenAI client once (used for rerank and answer if enabled)
    openai_api_key: Optional[str] = cfg.get("openai_api_key")
    oa = openai_client(openai_api_key)

    print("Interactive query mode. Type your question and press Enter.")
    print("Commands: /exit (quit), /help (show this help)\n")

    while True:
        try:
            question = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            return

        if not question:
            continue

        if question.lower() in {"/exit", "exit", "quit", "q"}:
            print("Exiting.")
            return

        if question.lower() in {"/help", "help", "?"}:
            print("Enter a question to query your local knowledge base.")
            print("Commands: /exit (quit), /help (show this help)\n")
            continue

        # Embed + retrieve
        qvec = embed_query(embedder, question, batch_size=batch_size)
        cands = retrieve(col, qvec, k)

        if not cands:
            print("No results found in Chroma for this query.\n")
            continue

        # Optional rerank
        if no_rerank:
            chosen = cands[: min(k_rerank, len(cands))]
        else:
            chosen = llm_rerank(
                client=oa,
                question=question,
                candidates=cands,
                model=model,
                top_n=min(k_rerank, len(cands)),
            )

        # Answer with citations
        answer, sources = answer_with_citations(
            client=oa,
            question=question,
            candidates=chosen,
            model=model,
        )

        if "[S" not in answer:
            answer += (
                "\n\nNote: No inline citations were emitted. "
                "If this persists, tighten query.k / query.k_rerank or reduce chunk size."
            )

        print("\n" + answer)
        print(format_sources(sources))
        print()  # blank line between turns


if __name__ == "__main__":
    main()
