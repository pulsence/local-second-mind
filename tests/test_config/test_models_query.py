from lsm.config.models.query import QueryConfig


def test_query_config_sets_local_pool_when_missing() -> None:
    cfg = QueryConfig(k=3, local_pool=None)
    assert cfg.local_pool == cfg.k * 4


def test_query_config_chat_mode_validation() -> None:
    cfg = QueryConfig(chat_mode="chat")
    cfg.validate()


def test_query_config_cache_validation() -> None:
    cfg = QueryConfig(enable_query_cache=True, query_cache_ttl=10, query_cache_size=5)
    cfg.validate()


def test_query_config_llm_server_cache_fields() -> None:
    cfg = QueryConfig(enable_llm_server_cache=True)
    cfg.validate()
    assert cfg.enable_llm_server_cache is True


def test_query_config_default_llm_server_cache_enabled() -> None:
    cfg = QueryConfig()
    assert cfg.enable_llm_server_cache is True
