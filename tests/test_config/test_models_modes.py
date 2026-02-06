from lsm.config.models.modes import ModeConfig


def test_mode_config_validate_accepts_default_style() -> None:
    cfg = ModeConfig()
    cfg.validate()

