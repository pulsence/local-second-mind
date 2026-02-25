"""Provider and model information formatters for the query screen.

These functions extract formatting logic from QueryScreen so rendering
can be tested independently of the TUI widget tree.
"""

from __future__ import annotations

from typing import Any, Optional

from lsm.logging import get_logger
from lsm.ui.utils import display_provider_name, format_feature_label
from lsm.providers import create_provider
from lsm.vectordb import create_vectordb_provider, list_available_providers

logger = get_logger(__name__)


def _get_feature_configs(llm_config: Any) -> dict[str, Any]:
    """Collect per-feature LLM configs.

    Args:
        llm_config: The ``config.llm`` object.

    Returns:
        Dict mapping feature name to resolved config (or None).
    """
    feature_map = llm_config.get_service_map() if hasattr(llm_config, "get_service_map") else {}
    return {
        "query": llm_config.get_query_config(),
        "synthesis": (
            llm_config.get_synthesis_config()
            if "synthesis" in feature_map
            else None
        ),
        "tagging": (
            llm_config.get_tagging_config()
            if "tagging" in feature_map
            else None
        ),
        "ranking": (
            llm_config.get_ranking_config()
            if "ranking" in feature_map
            else None
        ),
    }


def format_model_selection(llm_config: Any) -> str:
    """Format the current model selection summary.

    Args:
        llm_config: The ``config.llm`` object.

    Returns:
        Formatted multi-line string.
    """
    lines: list[str] = []
    feature_configs = _get_feature_configs(llm_config)
    for feature, cfg in feature_configs.items():
        if cfg is None:
            continue
        label = format_feature_label(feature)
        provider = display_provider_name(cfg.provider)
        lines.append(f"{label}: {provider}/{cfg.model}")
    lines.append("")
    return "\n".join(lines)


def format_models_list(
    llm_config: Any,
    command: str,
    *,
    store_callback: Optional[Any] = None,
) -> str:
    """Format available models grouped by provider.

    Args:
        llm_config: The ``config.llm`` object.
        command: The raw ``/models`` command string for filter parsing.
        store_callback: Optional callable receiving ``(model_ids,)`` for
            storing available models on session state.

    Returns:
        Formatted multi-line string.
    """
    parts = command.split()
    provider_filter = parts[1].strip().lower() if len(parts) > 1 else None
    providers = llm_config.get_provider_names()
    if provider_filter:
        if provider_filter == "claude":
            providers = [p for p in providers if p in {"claude", "anthropic"}]
        else:
            providers = [p for p in providers if p.lower() == provider_filter]
        if not providers:
            return f"Provider not found in config: {provider_filter}\n"

    lines: list[str] = []
    seen_labels: set[str] = set()
    for provider_name in providers:
        label = display_provider_name(provider_name)
        if label in seen_labels:
            continue
        seen_labels.add(label)
        if not llm_config.get_provider_by_name(provider_name):
            continue
        try:
            test_config = llm_config.resolve_any_for_provider(provider_name)
            lines.append(f"{label}:")
            if not test_config:
                lines.append("  (not configured for any feature)\n")
                continue
            provider = create_provider(test_config)
            ids = provider.list_models()
            ids.sort()
            if store_callback is not None:
                store_callback(ids)
            if not ids:
                lines.append("  (no models returned or listing unsupported)")
            else:
                lines.extend([f"  - {model_id}" for model_id in ids])
            lines.append("")
        except Exception as e:
            logger.error(f"Failed to list models for {provider_name}: {e}")
            lines.append(f"  (failed to list models: {e})\n")
    return "\n".join(lines)


def format_providers(llm_config: Any) -> str:
    """Format available LLM providers with selection and status.

    Args:
        llm_config: The ``config.llm`` object.

    Returns:
        Formatted multi-line string.
    """
    lines = [
        "",
        "=" * 60,
        "AVAILABLE LLM PROVIDERS",
        "=" * 60,
        "",
    ]

    providers = llm_config.get_provider_names()

    if not providers:
        lines.append("No providers configured.")
        lines.append("")
        return "\n".join(lines)

    lines.append("Selections:")
    feature_configs = _get_feature_configs(llm_config)
    for feature, cfg in feature_configs.items():
        if cfg is None:
            continue
        label = format_feature_label(feature)
        provider = display_provider_name(cfg.provider)
        lines.append(f"  {label:7s} {provider}/{cfg.model}")
    lines.append("")

    lines.append(f"Providers ({len(providers)}):")
    lines.append("")

    seen_labels: set[str] = set()
    for provider_name in providers:
        try:
            provider_config = llm_config.get_provider_by_name(provider_name)
            test_config = (
                llm_config.resolve_any_for_provider(provider_name)
                if provider_config
                else None
            )

            if test_config:
                provider = create_provider(test_config)
                is_available = (
                    "ok" if provider.is_available() else "- (API key not configured)"
                )
            elif provider_config:
                is_available = "- (no service config)"
            else:
                is_available = "- (not configured)"

            label = display_provider_name(provider_name)
            if label in seen_labels:
                continue
            seen_labels.add(label)
            lines.append(f"  {label:20s} {is_available:30s}")

        except Exception as e:
            logger.debug(f"Error checking provider {provider_name}: {e}")
            label = display_provider_name(provider_name)
            if label in seen_labels:
                continue
            seen_labels.add(label)
            lines.append(f"  {label:20s} {'- (error)':30s}")

    lines.append("")
    lines.append("To switch providers, update your config.json:")
    lines.append('  "llms": [ { "provider_name": "provider_name", ... } ]')
    lines.append("")

    return "\n".join(lines)


def format_provider_status(llm_config: Any) -> str:
    """Format provider health status.

    Args:
        llm_config: The ``config.llm`` object.

    Returns:
        Formatted multi-line string.
    """
    lines = [
        "",
        "=" * 60,
        "PROVIDER HEALTH STATUS",
        "=" * 60,
        "",
    ]

    providers = llm_config.get_provider_names()
    if not providers:
        lines.append("No providers registered.")
        lines.append("")
        return "\n".join(lines)

    current_provider = llm_config.get_query_config().provider

    seen_labels: set[str] = set()
    for provider_name in providers:
        try:
            provider_config = llm_config.get_provider_by_name(provider_name)
            test_config = (
                llm_config.resolve_any_for_provider(provider_name)
                if provider_config
                else None
            )
            if not test_config:
                status = "not_configured" if not provider_config else "missing_config"
                label = display_provider_name(provider_name)
                if label in seen_labels:
                    continue
                seen_labels.add(label)
                lines.append(f"{label:20s} status={status:12s}")
                continue

            provider = create_provider(test_config)
            health = provider.health_check()

            status = health.get("status", "unknown")
            stats = health.get("stats", {})
            success = stats.get("success_count", 0)
            failure = stats.get("failure_count", 0)
            last_error = stats.get("last_error")
            current_label = (
                " (current)" if provider_name == current_provider else ""
            )
            label = display_provider_name(provider_name)
            if label in seen_labels:
                continue
            seen_labels.add(label)

            lines.append(
                f"{label:20s} status={status:12s} success={success:4d} "
                f"failure={failure:4d}{current_label}"
            )
            if last_error:
                lines.append(f"{'':20s} last_error={last_error}")
        except Exception as e:
            logger.debug(f"Error checking provider status {provider_name}: {e}")
            label = display_provider_name(provider_name)
            if label in seen_labels:
                continue
            seen_labels.add(label)
            lines.append(f"{label:20s} status=error        error={e}")

    lines.append("")

    return "\n".join(lines)


def format_vectordb_providers(vectordb_config: Any) -> str:
    """Format available vector DB providers.

    Args:
        vectordb_config: The ``config.vectordb`` object.

    Returns:
        Formatted multi-line string.
    """
    providers = list_available_providers()

    lines = [
        "",
        "=" * 60,
        "AVAILABLE VECTOR DB PROVIDERS",
        "=" * 60,
        "",
    ]

    if not providers:
        lines.append("No vector DB providers registered.")
        lines.append("")
        return "\n".join(lines)

    current_provider = vectordb_config.provider
    lines.append(f"Current Provider: {current_provider}")
    lines.append(f"Collection:       {vectordb_config.collection}")
    lines.append("")

    lines.append(f"Available Providers ({len(providers)}):")
    lines.append("")

    for provider_name in providers:
        is_current = "ACTIVE" if provider_name == current_provider else ""
        status = ""
        if provider_name == current_provider:
            try:
                provider = create_vectordb_provider(vectordb_config)
                status = "ok" if provider.is_available() else "unavailable"
            except Exception as e:
                status = f"error ({e})"
        lines.append(f"  {provider_name:20s} {status:20s} {is_current}")

    lines.append("")
    lines.append("To switch providers, update your config.json:")
    lines.append('  "vectordb": { "provider": "provider_name", ... }')
    lines.append("")

    return "\n".join(lines)


def format_vectordb_status(
    vectordb_config: Any,
    llm_config: Any,
) -> str:
    """Format vector DB status combined with LLM provider health.

    Args:
        vectordb_config: The ``config.vectordb`` object.
        llm_config: The ``config.llm`` object.

    Returns:
        Formatted multi-line string.
    """
    lines = [
        "",
        "=" * 60,
        "VECTOR DB STATUS",
        "=" * 60,
        "",
    ]

    try:
        provider = create_vectordb_provider(vectordb_config)
        health = provider.health_check()
        stats = provider.get_stats()

        lines.append(f"Provider: {health.get('provider', 'unknown')}")
        lines.append(f"Status:   {health.get('status', 'unknown')}")
        if health.get("error"):
            lines.append(f"Error:    {health.get('error')}")
        lines.append(f"Count:    {stats.get('count', 'n/a')}")
    except Exception as e:
        lines.append(f"Error: {e}")

    lines.append("")
    # Append provider health section
    lines.append(format_provider_status(llm_config).lstrip("\n"))

    return "\n".join(lines)


def format_remote_providers(config: Any) -> str:
    """Format remote source provider summary.

    Args:
        config: The full app config.

    Returns:
        Formatted multi-line string.
    """
    lines = [
        "",
        "=" * 60,
        "REMOTE SOURCE PROVIDERS",
        "=" * 60,
        "",
    ]

    from lsm import remote as remote_module

    registered = remote_module.get_registered_providers()
    lines.append(f"Registered Provider Types ({len(registered)}):")
    for provider_type, provider_class in sorted(registered.items()):
        lines.append(f"  {provider_type:20s} -> {provider_class.__name__}")
    lines.append("")

    configured = config.remote_providers or []
    if not configured:
        lines.append("No remote providers configured.")
        lines.append("")
        lines.append("Add providers to your config.json:")
        lines.append(
            '  "remote_providers": [{"name": "...", "type": "...", ...}]'
        )
        lines.append("")
        return "\n".join(lines)

    lines.append(f"Configured Providers ({len(configured)}):")
    lines.append("")
    lines.append(
        f"  {'NAME':<20s} {'TYPE':<20s} {'STATUS':<10s} {'WEIGHT':<8s} {'API KEY'}"
    )
    lines.append("  " + "-" * 70)

    for provider_config in configured:
        name = provider_config.name
        ptype = provider_config.type
        status = "enabled" if provider_config.enabled else "disabled"
        weight = f"{provider_config.weight:.1f}"
        api_key = provider_config.api_key
        has_key = (
            "set"
            if api_key and not api_key.startswith("INSERT")
            else "not set"
        )
        if ptype in {"wikipedia", "arxiv", "openalex", "crossref", "oai_pmh"}:
            has_key = "n/a"
        lines.append(
            f"  {name:<20s} {ptype:<20s} {status:<10s} {weight:<8s} {has_key}"
        )

    lines.append("")

    mode_config = config.get_mode_config()
    remote_policy = mode_config.source_policy.remote
    lines.append("Current Mode Remote Settings:")
    lines.append(f"  Enabled:       {remote_policy.enabled}")
    lines.append(f"  Max Results:   {remote_policy.max_results}")
    lines.append(f"  Rank Strategy: {remote_policy.rank_strategy}")
    if remote_policy.remote_providers:
        lines.append(
            f"  Mode Providers: {', '.join(str(p) for p in remote_policy.remote_providers)}"
        )
    lines.append("")

    return "\n".join(lines)
