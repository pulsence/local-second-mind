"""
Mode validation for agent pipeline tools.

Prevents agents from using invalid retrieval profiles or enabling
remote sources without URL access permission.
"""

from __future__ import annotations

from typing import Any, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from lsm.config.models import LSMConfig
    from lsm.config.models.agents import SandboxConfig

# Valid retrieval profiles (from lsm.query.pipeline)
VALID_PROFILES = (
    "dense_only",
    "hybrid_rrf",
    "hyde_hybrid",
    "dense_cross_rerank",
    "llm_rerank",
)


def validate_agent_mode(
    mode_name: Optional[str],
    config: "LSMConfig",
    sandbox_config: Optional["SandboxConfig"] = None,
) -> Optional[str]:
    """Validate a mode name for agent use.

    Returns None if valid, or an error message string if invalid.
    """
    if mode_name is None:
        return None

    # Resolve the mode config
    try:
        mode_config = config.get_mode_config(mode_name)
    except Exception:
        return f"Unknown mode: '{mode_name}'"

    # Validate retrieval profile
    profile = mode_config.retrieval_profile
    if profile and profile not in VALID_PROFILES:
        return (
            f"Invalid retrieval profile '{profile}' in mode '{mode_name}'. "
            f"Allowed profiles: {', '.join(VALID_PROFILES)}"
        )

    # Validate remote policy vs sandbox URL access
    if sandbox_config is not None:
        remote_policy = mode_config.remote_policy
        if remote_policy.enabled and not sandbox_config.allow_url_access:
            return (
                f"Mode '{mode_name}' has remote sources enabled, but the agent sandbox "
                "does not allow URL access (allow_url_access=false)"
            )

    return None
