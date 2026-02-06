from lsm.logging import get_logger


def test_get_logger_prefixes_lsm_namespace() -> None:
    logger = get_logger("module")
    assert logger.name == "lsm.module"
