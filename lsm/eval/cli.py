"""
CLI entry points for retrieval evaluation commands.
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Optional

from lsm.config.models import LSMConfig
from lsm.eval.baselines import list_baselines, load_baseline, save_baseline
from lsm.eval.dataset import EvalDataset, load_bundled_dataset, load_dataset
from lsm.eval.harness import EvalHarness, EvalResult
from lsm.logging import get_logger

logger = get_logger(__name__)


def _load_eval_dataset(dataset_path: Optional[str]) -> EvalDataset:
    """Load dataset from path or use bundled."""
    if dataset_path:
        return load_dataset(Path(dataset_path))
    return load_bundled_dataset()


def _build_retrieve_fn(config: LSMConfig, profile: str):
    """
    Build a retrieval function for evaluation.

    This creates a function that queries the vector database and returns
    (doc_id, source_path) tuples in ranked order.
    """
    from lsm.providers import create_provider
    from lsm.query.pipeline import RetrievalPipeline
    from lsm.query.pipeline_types import QueryRequest
    from lsm.ingest.embeddings import load_embedder
    from lsm.vectordb import create_vectordb

    eval_config = deepcopy(config)
    eval_config.query.retrieval_profile = profile
    try:
        mode_cfg = eval_config.get_mode_config(eval_config.query.mode)
        mode_cfg.retrieval_profile = profile
    except Exception:
        # If mode lookup fails, RetrievalPipeline resolves to GROUNDED_MODE.
        pass

    embedder = load_embedder(eval_config)
    db = create_vectordb(eval_config)
    llm_provider = create_provider(eval_config.llm.resolve_service("query"))
    pipeline = RetrievalPipeline(
        db=db,
        embedder=embedder,
        config=eval_config,
        llm_provider=llm_provider,
    )

    def retrieve_fn(query_text: str):
        request = QueryRequest(
            question=query_text,
            mode=eval_config.query.mode,
            chat_mode="single",
        )
        package = pipeline.build_sources(request)
        candidates = [
            c for c in package.candidates if not (c.meta or {}).get("remote")
        ]
        return [
            (c.cid, c.source_path)
            for c in candidates
        ]

    return retrieve_fn


def _print_result(result: EvalResult) -> None:
    """Print evaluation results to stdout."""
    print(f"\nRetrieval Evaluation: {result.profile}")
    print(f"Dataset: {result.dataset_name} ({result.num_queries} queries)")
    print("-" * 50)
    print(f"  Recall@5:     {result.recall_at_5:.4f}")
    print(f"  Recall@10:    {result.recall_at_10:.4f}")
    print(f"  MRR:          {result.mrr_value:.4f}")
    print(f"  nDCG@5:       {result.ndcg_at_5:.4f}")
    print(f"  nDCG@10:      {result.ndcg_at_10:.4f}")
    print(f"  Diversity@5:  {result.diversity_at_5:.4f}")
    print(f"  Latency (ms): mean={result.latency['mean']:.1f}  "
          f"p50={result.latency['p50']:.1f}  "
          f"p95={result.latency['p95']:.1f}  "
          f"p99={result.latency['p99']:.1f}")


def run_eval(args, config: LSMConfig) -> int:
    """Dispatch eval subcommands."""
    cmd = args.eval_command

    if cmd == "retrieval":
        return _cmd_retrieval(args, config)
    elif cmd == "save-baseline":
        return _cmd_save_baseline(args, config)
    elif cmd == "list-baselines":
        return _cmd_list_baselines(config)
    else:
        print(f"Unknown eval command: {cmd}")
        return 1


def _cmd_retrieval(args, config: LSMConfig) -> int:
    """Run retrieval evaluation."""
    dataset = _load_eval_dataset(getattr(args, "dataset", None))
    profile = args.profile

    print(f"Running retrieval evaluation (profile: {profile})...")
    retrieve_fn = _build_retrieve_fn(config, profile)
    harness = EvalHarness(dataset)
    result = harness.run(retrieve_fn, profile=profile)
    _print_result(result)

    compare_name = getattr(args, "compare", None)
    if compare_name:
        baseline_dict = load_baseline(compare_name, config.global_folder)
        if baseline_dict is None:
            print(f"\nBaseline '{compare_name}' not found.")
            return 1
        baseline = EvalResult.from_dict(baseline_dict)
        report = EvalHarness.compare(result, baseline)
        print(f"\n{report.summary()}")

    return 0


def _cmd_save_baseline(args, config: LSMConfig) -> int:
    """Evaluate and save results as baseline."""
    dataset = _load_eval_dataset(getattr(args, "dataset", None))
    profile = args.profile
    name = args.name

    print(f"Running retrieval evaluation (profile: {profile})...")
    retrieve_fn = _build_retrieve_fn(config, profile)
    harness = EvalHarness(dataset)
    result = harness.run(retrieve_fn, profile=profile)
    _print_result(result)

    path = save_baseline(result.to_dict(), name, config.global_folder)
    print(f"\nBaseline '{name}' saved to {path}")
    return 0


def _cmd_list_baselines(config: LSMConfig) -> int:
    """List saved baselines."""
    names = list_baselines(config.global_folder)
    if not names:
        print("No saved baselines.")
    else:
        print("Saved baselines:")
        for name in names:
            print(f"  - {name}")
    return 0
