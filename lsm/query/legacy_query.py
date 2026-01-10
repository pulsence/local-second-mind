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

import os
import subprocess
import sys

import re
import math
from collections import Counter

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

from dotenv import load_dotenv

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

@dataclass
class SessionState:
    # Session-level overrides (start with config defaults)
    path_contains: Optional[Any] = None
    ext_allow: Optional[List[str]] = None
    ext_deny: Optional[List[str]] = None

    # Model override (session)
    model: str = "gpt-5.2"
    available_models: Optional[List[str]] = None

    # Last-turn artifacts...
    last_question: Optional[str] = None
    last_all_candidates: List[Candidate] = None
    last_filtered_candidates: List[Candidate] = None
    last_chosen: List[Candidate] = None
    last_label_to_candidate: Dict[str, Candidate] = None
    last_debug: Dict[str, Any] = None

    def __post_init__(self):
        self.available_models = self.available_models or []
        self.last_all_candidates = self.last_all_candidates or []
        self.last_filtered_candidates = self.last_filtered_candidates or []
        self.last_chosen = self.last_chosen or []
        self.last_label_to_candidate = self.last_label_to_candidate or {}
        self.last_debug = self.last_debug or {}

@dataclass
class Runtime:
    embedder: SentenceTransformer
    col: Any  # chromadb Collection
    oa: OpenAI
    cfg: Dict[str, Any]

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

def _normalize_ext(x: str) -> str:
    x = (x or "").strip().lower()
    if not x:
        return ""
    return x if x.startswith(".") else f".{x}"

def apply_filters(
    cands: List[Candidate],
    path_contains: Optional[Any] = None,
    ext_allow: Optional[List[str]] = None,
    ext_deny: Optional[List[str]] = None,
) -> List[Candidate]:
    """
    Post-retrieval filtering using Candidate.meta keys:
      - source_path: str
      - ext: str (e.g., ".md", ".pdf")
    """
    if not cands:
        return []

    # path_contains can be str or list[str]
    pcs: List[str] = []
    if isinstance(path_contains, str) and path_contains.strip():
        pcs = [path_contains.strip().lower()]
    elif isinstance(path_contains, list):
        pcs = [str(p).strip().lower() for p in path_contains if str(p).strip()]

    allow = {_normalize_ext(e) for e in (ext_allow or []) if _normalize_ext(e)}
    deny = {_normalize_ext(e) for e in (ext_deny or []) if _normalize_ext(e)}

    out: List[Candidate] = []
    for c in cands:
        m = c.meta or {}
        sp = str(m.get("source_path", "") or "")
        sp_l = sp.lower()

        # Path filter
        if pcs:
            if not any(p in sp_l for p in pcs):
                continue

        # Ext filter
        ext = _normalize_ext(str(m.get("ext", "") or ""))
        if allow and ext and ext not in allow:
            continue
        if deny and ext and ext in deny:
            continue

        out.append(c)

    return out

def best_relevance(cands: List[Candidate]) -> float:
    """
    Convert Chroma 'distance' (lower is better) into a simple relevance proxy.
    For cosine distance, relevance ~ 1 - distance. Clamped to [-1, 1].
    """
    if not cands:
        return -1.0

    dists = [c.distance for c in cands if c.distance is not None]
    if not dists:
        return -1.0

    best_dist = min(dists)
    rel = 1.0 - float(best_dist)
    # clamp just to keep prints sane
    if rel > 1.0:
        rel = 1.0
    if rel < -1.0:
        rel = -1.0
    return rel

_STOPWORDS = {
    "a","an","the","and","or","but","if","then","else","when","while","to","of","in","on","for","with","as",
    "at","by","from","into","about","over","under","after","before","between","through","during","without",
    "is","are","was","were","be","been","being","do","does","did","done","doing","have","has","had",
    "i","you","he","she","it","we","they","me","him","her","us","them","my","your","his","their","our",
    "this","that","these","those","there","here","not","no","yes","so","than","too","very","can","could",
    "should","would","may","might","must","will","just"
}

def _tokenize(text: str) -> List[str]:
    # simple, fast tokenizer; keep words/numbers; lowercased
    toks = re.findall(r"[a-zA-Z0-9']+", (text or "").lower())
    return [t for t in toks if len(t) >= 2 and t not in _STOPWORDS]

def lexical_score(question: str, passage: str) -> float:
    """
    Cheap lexical relevance score:
    - token overlap (weighted toward question tokens)
    - phrase bonus for contiguous query substrings appearing in passage
    """
    q_toks = _tokenize(question)
    p_toks = _tokenize(passage)

    if not q_toks or not p_toks:
        return 0.0

    q_set = set(q_toks)
    p_set = set(p_toks)

    overlap = len(q_set & p_set)
    base = overlap / max(1, len(q_set))  # 0..1

    # phrase bonus: reward if a meaningful phrase appears verbatim (lowercased)
    q_norm = " ".join(q_toks)
    p_norm = " ".join(p_toks)

    # use a few phrase lengths to avoid expensive n-gram scans
    bonus = 0.0
    for n in (4, 3):
        if len(q_toks) >= n:
            # take sliding windows but cap count to avoid huge cost
            max_windows = min(10, len(q_toks) - n + 1)
            for i in range(max_windows):
                phrase = " ".join(q_toks[i:i+n])
                if phrase and phrase in p_norm:
                    bonus += 0.10
                    break

    return base + bonus

def lexical_rerank(question: str, cands: List[Candidate]) -> List[Candidate]:
    scored = []
    for c in cands:
        s = lexical_score(question, c.text or "")
        scored.append((s, c))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [c for _, c in scored]

def dedupe_candidates(cands: List[Candidate], max_chars: int = 2000) -> List[Candidate]:
    """
    Deduplicate by normalized text hash. Keeps first occurrence (which will be
    higher-ranked if you call this after retrieval or lexical rerank).
    """
    seen = set()
    out: List[Candidate] = []

    for c in cands:
        t = (c.text or "").strip()
        if not t:
            continue

        # normalize whitespace and case; cap to keep hashing cheap
        norm = re.sub(r"\s+", " ", t).strip().lower()
        norm = norm[:max_chars]

        h = hash(norm)
        if h in seen:
            continue
        seen.add(h)
        out.append(c)

    return out

def diversify_by_file(cands: List[Candidate], max_per_file: int = 2) -> List[Candidate]:
    """
    Enforce diversity by limiting number of chunks per source_path.
    Keeps order and fills as many as possible.
    """
    if max_per_file <= 0:
        return cands

    counts: Dict[str, int] = {}
    out: List[Candidate] = []

    for c in cands:
        sp = str((c.meta or {}).get("source_path", "unknown"))
        n = counts.get(sp, 0)
        if n >= max_per_file:
            continue
        counts[sp] = n + 1
        out.append(c)

    return out

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
    """
    LLM-based reranker using OpenAI Responses API.

    Returns candidates reordered by usefulness for answering `question`.
    If the API fails (quota, 429, etc.), falls back to original vector rank.
    """
    if not candidates:
        return []

    top_n = max(1, min(top_n, len(candidates)))

    def snippet(t: str, max_chars: int = 1200) -> str:
        t = (t or "").strip()
        return t if len(t) <= max_chars else (t[: max_chars - 50] + "\n...[truncated]...")

    # Build rerank inputs (keep metadata aligned with ingest.py)
    items: List[Dict[str, Any]] = []
    for i, c in enumerate(candidates):
        m = c.meta or {}
        items.append(
            {
                "index": i,
                "source_path": m.get("source_path", "unknown"),
                "source_name": m.get("source_name"),
                "chunk_index": m.get("chunk_index"),
                "ext": m.get("ext"),
                "distance": c.distance,
                "text": snippet(c.text),
            }
        )

    instructions = (
        "You are a retrieval reranker.\n"
        "Goal: rank the candidate passages by how useful they are for answering the user's question.\n"
        "Guidance:\n"
        "- Prefer passages that directly address the question.\n"
        "- Prefer specificity, definitions, arguments, or evidence over vague mentions.\n"
        "- If multiple passages are similar, rank the most comprehensive/precise first.\n"
        "- Do NOT hallucinate facts; you are only ranking.\n\n"
        "Output requirements:\n"
        "- Return STRICT JSON only, no markdown, no extra text.\n"
        "- Schema: {\"ranking\":[{\"index\":int,\"reason\":string}...]}\n"
        "- Include at most top_n items.\n"
    )

    payload = {
        "question": question,
        "top_n": top_n,
        "candidates": items,
    }

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
        if not isinstance(ranking, list):
            return candidates[:top_n]

        chosen: List[Candidate] = []
        seen = set()

        for r in ranking:
            if not isinstance(r, dict) or "index" not in r:
                continue
            idx = int(r["index"])
            if 0 <= idx < len(candidates) and idx not in seen:
                chosen.append(candidates[idx])
                seen.add(idx)
            if len(chosen) >= top_n:
                break

        # Fill any gaps deterministically from original order
        if len(chosen) < top_n:
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
# REPL Display + command handling
# -----------------------------
def open_file(path: str) -> None:
    """
    Open a file with the system default application.
    Cross-platform: Windows, macOS, Linux.
    """
    if not path or not os.path.exists(path):
        print(f"File does not exist: {path}\n")
        return

    try:
        if sys.platform.startswith("win"):
            os.startfile(path)  # type: ignore[attr-defined]
        elif sys.platform == "darwin":
            subprocess.run(["open", path], check=False)
        else:
            subprocess.run(["xdg-open", path], check=False)
    except Exception as e:
        print(f"Failed to open file: {e}\n")

def print_banner() -> None:
    print("Interactive query mode. Type your question and press Enter.")
    print("Commands: /exit, /help, /show S#, /expand S#, /open S#, /debug, /model, /models, /set, /clear\n")

def print_help() -> None:
    print("Enter a question to query your local knowledge base.")
    print("Commands:")
    print("  /exit           Quit")
    print("  /help           Show this help")
    print("  /show S#        Show the cited chunk (e.g., /show S2)")
    print("  /expand S#      Show full chunk text (no truncation)")
    print("  /open S#        Open the source file in default app")
    print("  /models         List models available to the API key")
    print("  /model          Show current model")
    print("  /model <name>   Set model for this session")
    print("  /debug          Print retrieval diagnostics for the last query")
    print("  /set …          Set session filters (path/ext)")
    print("  /clear …        Clear session filters\n")

def print_source_chunk(label: str, c: Candidate, expanded: bool = False) -> None:
    m = c.meta or {}
    sp = m.get("source_path", "unknown")
    ci = m.get("chunk_index", "NA")
    dist = c.distance

    if expanded:
        print(f"\n{label} — {sp}")
        print(f"chunk_index={ci}, distance={dist}")
        print("=" * 80)
        print((c.text or "").strip())
        print("=" * 80 + "\n")
    else:
        print(f"\n{label} — {sp} (chunk_index={ci}, distance={dist})")
        print("-" * 80)
        print((c.text or "").strip())
        print("-" * 80 + "\n")

def list_models(client: OpenAI) -> List[str]:
    """
    Returns model IDs available to the current API key/project.
    Uses the Models API: /v1/models. :contentReference[oaicite:2]{index=2}
    """
    res = client.models.list()
    # openai-python returns objects with `.id` commonly; be defensive
    ids: List[str] = []
    for m in getattr(res, "data", []) or []:
        mid = getattr(m, "id", None)
        if isinstance(mid, str):
            ids.append(mid)
    ids.sort()
    return ids

def print_models(state: SessionState, rt: Runtime) -> None:
    try:
        ids = list_models(rt.oa)
        state.available_models = ids
        if not ids:
            print("No models returned by the API for this key/project.\n")

        print("\nAvailable models (API key scope):")
        # Print in columns-ish without extra deps
        for mid in ids:
            print(f"- {mid}")
        print()
    except Exception as e:
        print(f"Failed to list models: {e}\n")

def print_debug(state: SessionState) -> None:
    if not state.last_debug:
        print("No debug info yet. Ask a question first.\n")
        return

    print("\nDebug (last query):")
    for k_, v_ in state.last_debug.items():
        print(f"- {k_}: {v_}")

    print("\nTop candidates (post-filter):")
    for i, c in enumerate(state.last_filtered_candidates[: min(10, len(state.last_filtered_candidates))], start=1):
        m = c.meta or {}
        sp = m.get("source_path", "unknown")
        name = m.get("source_name") or Path(sp).name
        ci = m.get("chunk_index", "NA")
        print(f"  {i:02d}. {name} (chunk_index={ci}, distance={c.distance})")
    print()

def handle_command(line: str, state: SessionState, rt: Runtime) -> bool:
    """
    Returns True if the input was a command and has been handled.
    Returns False if the caller should treat it as a normal question.
    """
    q = line.strip()
    ql = q.lower()

    # Exit
    if ql in {"/exit", "exit", "quit", "q"}:
        raise SystemExit

    # Help
    if ql in {"/help", "help", "?"}:
        print_help()
        return True

    # Debug
    if ql == "/debug":
        print_debug(state)
        return True

    # List available models
    if ql.strip() == "/models":
        print_models(state, rt)
        return True

    # Show/set current model
    if ql.startswith("/model"):
        parts = q.split()
        if len(parts) == 1:
            print(f"Current model: {state.model}\n")
            return True

        if len(parts) != 2:
            print("Usage:")
            print("  /model           (show current)")
            print("  /model <name>    (set model for this session)")
            print("  /models          (list available models)\n")
            return True

        new_model = parts[1].strip()

        # Optional validation if we've fetched /models at least once
        if state.available_models:
            if new_model not in state.available_models:
                print(f"Model not found in last /models list: {new_model}")
                print("Run /models to refresh the list or set anyway by clearing cache.\n")
                return True

        state.model = new_model
        print(f"Model set to: {state.model}\n")
        return True

    # Show / Expand
    if ql.startswith("/show") or ql.startswith("/expand"):
        parts = q.split()
        if len(parts) != 2:
            usage = "/show S#   (e.g., /show S2)" if ql.startswith("/show") else "/expand S#   (e.g., /expand S2)"
            print(f"Usage: {usage}\n")
            return True

        label = parts[1].strip().upper()
        c = state.last_label_to_candidate.get(label)
        if not c:
            print(f"No such label in last results: {label}\n")
            return True

        print_source_chunk(label, c, expanded=ql.startswith("/expand"))
        return True

    # Open
    if ql.startswith("/open"):
        parts = q.split()
        if len(parts) != 2:
            print("Usage: /open S#   (e.g., /open S2)\n")
            return True

        label = parts[1].strip().upper()
        c = state.last_label_to_candidate.get(label)
        if not c:
            print(f"No such label in last results: {label}\n")
            return True

        path = (c.meta or {}).get("source_path")
        if not path:
            print("No source_path available for this citation.\n")
            return True

        open_file(path)
        return True

    # Set filters
    if ql.startswith("/set"):
        parts = q.split()
        if len(parts) < 3:
            print("Usage:")
            print("  /set path_contains <substring> [more...]")
            print("  /set ext_allow .md .pdf")
            print("  /set ext_deny .txt\n")
            return True

        key = parts[1]
        values = parts[2:]

        if key == "path_contains":
            state.path_contains = values if len(values) > 1 else values[0]
            print(f"path_contains set to: {state.path_contains}\n")
            return True

        if key == "ext_allow":
            state.ext_allow = values
            print(f"ext_allow set to: {state.ext_allow}\n")
            return True

        if key == "ext_deny":
            state.ext_deny = values
            print(f"ext_deny set to: {state.ext_deny}\n")
            return True

        print(f"Unknown filter key: {key}\n")
        return True

    # Clear filters
    if ql.startswith("/clear"):
        parts = q.split()
        if len(parts) != 2:
            print("Usage: /clear path_contains|ext_allow|ext_deny\n")
            return True

        key = parts[1]
        if key == "path_contains":
            state.path_contains = None
            print("path_contains cleared.\n")
            return True
        if key == "ext_allow":
            state.ext_allow = None
            print("ext_allow cleared.\n")
            return True
        if key == "ext_deny":
            state.ext_deny = None
            print("ext_deny cleared.\n")
            return True

        print(f"Unknown filter key: {key}\n")
        return True

    return False

def run_query_turn(question: str, rt: Runtime, state: SessionState) -> None:
    """
    Executes one query turn end-to-end, updating state and printing results.
    """
    cfg = rt.cfg

    batch_size: int = cfg["batch_size"]
    k: int = cfg["k"]
    k_rerank: int = cfg["k_rerank"]
    no_rerank: bool = cfg["no_rerank"]
    model: str =  state.model

    state.last_question = question

    # Embed + retrieve (retrieve more than k if filters are active)
    qvec = embed_query(rt.embedder, question, batch_size=batch_size)

    path_contains = state.path_contains
    ext_allow = state.ext_allow
    ext_deny = state.ext_deny

    filters_active = bool(path_contains) or bool(ext_allow) or bool(ext_deny)
    retrieve_k_cfg = cfg.get("retrieve_k")
    retrieve_k = int(retrieve_k_cfg) if retrieve_k_cfg else (max(k, k * 3) if filters_active else k)

    cands = retrieve(rt.col, qvec, retrieve_k)
    state.last_all_candidates = cands

    if not cands:
        print("No results found in Chroma for this query.\n")
        return

    # Apply filters
    filtered = apply_filters(cands, path_contains=path_contains, ext_allow=ext_allow, ext_deny=ext_deny)
    state.last_filtered_candidates = filtered

    if not filtered:
        print("No results matched the configured filters.\n")
        return

    # -----------------------------
    # Local quality passes:
    # (5) dedupe, (4) lexical rerank, (6) per-file diversity
    # -----------------------------
    # Optional knobs (use sane defaults if absent)
    max_per_file = int(cfg.get("max_per_file", 2))

    # If you want, you can separately limit how many candidates survive local rerank,
    # but defaulting to k*3 keeps recall decent.
    local_pool = int(cfg.get("local_pool", max(k * 3, k_rerank * 4)))

    # Start from filtered candidates (still in vector similarity order)
    local = filtered

    # (5) dedupe early to remove overlap noise
    local = dedupe_candidates(local)

    # (4) lexical rerank to surface proper-noun / keyword matches
    local = lexical_rerank(question, local)

    # keep a manageable pool before diversity
    local = local[: min(local_pool, len(local))]

    # (6) enforce per-file diversity
    local = diversify_by_file(local, max_per_file=max_per_file)

    # Final trim to k for downstream steps
    filtered = local[: min(k, len(local))]

    # Relevance gating
    min_rel = float(cfg.get("min_relevance", 0.25))
    rel = best_relevance(filtered)

    state.last_debug = {
        "question": question,
        "retrieve_k": retrieve_k,
        "k": k,
        "k_rerank": k_rerank,
        "filters_active": filters_active,
        "path_contains": path_contains,
        "ext_allow": ext_allow,
        "ext_deny": ext_deny,
        "best_relevance": rel,
        "min_relevance": min_rel,
        "no_rerank": no_rerank,
        "model": model,
        "max_per_file": max_per_file,
        "local_pool": local_pool,
        "post_local_count": len(filtered),
    }

    if rel < min_rel:
        chosen = filtered[: min(k_rerank, len(filtered))]
        state.last_chosen = chosen
        state.last_label_to_candidate = {f"S{i}": c for i, c in enumerate(chosen, start=1)}

        answer = local_fallback_answer(question, chosen)
        _, sources = build_sources_block(chosen)

        print("\n" + answer)
        print(format_sources(sources))
        print()
        return

    # Optional rerank
    if no_rerank:
        chosen = filtered[: min(k_rerank, len(filtered))]
    else:
        chosen = llm_rerank(
            client=rt.oa,
            question=question,
            candidates=filtered,
            model=model,
            top_n=min(k_rerank, len(filtered)),
        )

    state.last_chosen = chosen
    state.last_label_to_candidate = {f"S{i}": c for i, c in enumerate(chosen, start=1)}

    # Answer with citations
    answer, sources = answer_with_citations(
        client=rt.oa,
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
    print()


# -----------------------------
# CLI Helpers
# -----------------------------
def load_config(path: Path) -> Dict[str, Any]:
    """
    Load YAML or JSON config file (same as ingest.py).

    Also loads environment variables from .env file if present.
    """
    # Load environment variables from .env file (if it exists)
    load_dotenv()

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
        max_per_file: int              (default 2)
        local_pool: int                (default k*3 or k_rerank*4)

        model: str                     (default "gpt-5.2")

        min_relevance: float           (default 0.25)
        path_contains: str | list[str] (default None)
        ext_allow: list[str]           (default None)
        ext_deny: list[str]            (default None)
        retrieve_k: int | None         (default None)
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
    # Try config first, then fall back to environment variable
    ocfg = cfg.get("openai", {})
    out["openai_api_key"] = ocfg.get("api_key") or os.getenv("OPENAI_API_KEY")

    # ---- query-specific ----
    qcfg = cfg.get("query", {})

    # Number of chunks retrieved from Chroma before reranking.
    out["k"] = int(qcfg.get("k", 12))
    # Number of chunks kept after LLM reranking and passed to the answer step.
    out["k_rerank"] = int(qcfg.get("k_rerank", 6))
    # If true, skips the reranking step entirely and uses vector similarity order.
    out["no_rerank"] = bool(qcfg.get("no_rerank", False))
    out["max_per_file"] = int(qcfg.get("max_per_file", 2))
    out["local_pool"] = int(qcfg.get("local_pool", max(out["k"] * 3, out["k_rerank"] * 4)))

    out["model"] = str(qcfg.get("model", "gpt-5.2"))

    # Relevance gating:
    # Treat distance as cosine distance (lower is better).
    # Convert to a simple "relevance" proxy: relevance = 1 - best_distance.
    # Gate OpenAI usage if relevance < min_relevance.
    out["min_relevance"] = float(qcfg.get("min_relevance", 0.25))

    # Filters (optional)
    # If provided, applied after retrieval using metadata: source_path and ext.
    out["path_contains"] = qcfg.get("path_contains", None)  # str or list[str]
    out["ext_allow"] = qcfg.get("ext_allow", None)          # list[str] like [".md", ".pdf"]
    out["ext_deny"] = qcfg.get("ext_deny", None)            # list[str]

    # Optional: retrieve more than k so filters still leave enough candidates.
    # If not set, we'll default to max(k, k*3) whenever any filter is active.
    out["retrieve_k"] = qcfg.get("retrieve_k", None)

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

# -----------------------------
# CLI 
# -----------------------------
def main() -> None:
    args = parse_args()

    cfg_path = Path(args.config).expanduser().resolve()
    cfg_raw = load_config(cfg_path)
    cfg = normalize_config(cfg_raw, cfg_path)

    # Build embedder + Chroma once
    embedder = build_embedder(cfg["embed_model"], device=cfg["device"])
    col = chroma_collection(cfg["persist_dir"], cfg["collection"])

    # OpenAI client once
    oa = openai_client(cfg.get("openai_api_key"))

    rt = Runtime(embedder=embedder, col=col, oa=oa, cfg=cfg)

    # Session state (filters initialized from config)
    state = SessionState(
        path_contains=cfg.get("path_contains"),
        ext_allow=cfg.get("ext_allow"),
        ext_deny=cfg.get("ext_deny"),
        model=cfg.get("model")
    )

    print_banner()

    while True:
        try:
            line = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            return

        if not line:
            continue

        try:
            if handle_command(line, state, rt):
                continue
        except SystemExit:
            print("Exiting.")
            return

        run_query_turn(line, rt, state)

if __name__ == "__main__":
    main()