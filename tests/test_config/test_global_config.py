import os
from pathlib import Path

import pytest

from lsm.config.models.global_config import GlobalConfig, MCPServerConfig


@pytest.fixture(autouse=True)
def _restore_env_after_test():
    original = dict(os.environ)
    yield
    os.environ.clear()
    os.environ.update(original)


def test_global_config_defaults() -> None:
    cfg = GlobalConfig()
    assert cfg.embed_model == "sentence-transformers/all-MiniLM-L6-v2"
    assert cfg.device == "cpu"
    assert cfg.batch_size == 32
    assert cfg.tui_density_mode == "auto"
    assert cfg.global_folder is not None
    assert isinstance(cfg.global_folder, Path)


def test_global_config_custom_values() -> None:
    cfg = GlobalConfig(
        global_folder=Path("/tmp/lsm-test"),
        embed_model="custom-model",
        device="cuda:0",
        batch_size=64,
        tui_density_mode="compact",
    )
    assert cfg.embed_model == "custom-model"
    assert cfg.device == "cuda:0"
    assert cfg.batch_size == 64
    assert cfg.tui_density_mode == "compact"


def test_global_config_string_path_conversion() -> None:
    cfg = GlobalConfig(global_folder="/tmp/lsm-test")
    assert isinstance(cfg.global_folder, Path)


def test_global_config_env_var_global_folder() -> None:
    os.environ["LSM_GLOBAL_FOLDER"] = "/tmp/env-lsm"
    cfg = GlobalConfig()
    assert cfg.global_folder == Path("/tmp/env-lsm").resolve()


def test_global_config_env_var_embed_model() -> None:
    os.environ["LSM_EMBED_MODEL"] = "env-model"
    cfg = GlobalConfig()
    assert cfg.embed_model == "env-model"


def test_global_config_env_var_device() -> None:
    os.environ["LSM_DEVICE"] = "cuda:1"
    cfg = GlobalConfig()
    assert cfg.device == "cuda:1"


def test_global_config_explicit_folder_overrides_env() -> None:
    os.environ["LSM_GLOBAL_FOLDER"] = "/tmp/env-lsm"
    cfg = GlobalConfig(global_folder="/tmp/explicit")
    assert cfg.global_folder == Path("/tmp/explicit").resolve()


def test_global_config_validate_valid() -> None:
    cfg = GlobalConfig(device="cpu", batch_size=32)
    cfg.validate()


def test_global_config_validate_cuda_device() -> None:
    cfg = GlobalConfig(device="cuda:0")
    cfg.validate()


def test_global_config_validate_mps_device() -> None:
    cfg = GlobalConfig(device="mps")
    cfg.validate()


def test_global_config_validate_invalid_device() -> None:
    cfg = GlobalConfig(device="tpu")
    with pytest.raises(ValueError, match="device must start with"):
        cfg.validate()


def test_global_config_validate_invalid_batch_size() -> None:
    cfg = GlobalConfig(batch_size=0)
    with pytest.raises(ValueError, match="batch_size must be positive"):
        cfg.validate()


def test_global_config_validate_negative_batch_size() -> None:
    cfg = GlobalConfig(batch_size=-1)
    with pytest.raises(ValueError, match="batch_size must be positive"):
        cfg.validate()


def test_global_config_validate_invalid_tui_density_mode() -> None:
    cfg = GlobalConfig(tui_density_mode="invalid")
    with pytest.raises(ValueError, match="tui_density_mode must be one of"):
        cfg.validate()


def test_global_config_mcp_server_normalization() -> None:
    cfg = GlobalConfig(
        mcp_servers=[
            {
                "name": " demo ",
                "command": "  run  ",
                "args": ["  --flag", ""],
                "env": {" KEY ": "VALUE"},
            }
        ]
    )
    server = cfg.mcp_servers[0]
    assert server.name == "demo"
    assert server.command == "run"
    assert server.args == ["--flag"]
    assert server.env == {"KEY": "VALUE"}


def test_global_config_mcp_server_duplicate_names() -> None:
    cfg = GlobalConfig(
        mcp_servers=[
            MCPServerConfig(name="dup", command="run"),
            MCPServerConfig(name="dup", command="run2"),
        ]
    )
    with pytest.raises(ValueError, match="Duplicate MCP server name"):
        cfg.validate()
