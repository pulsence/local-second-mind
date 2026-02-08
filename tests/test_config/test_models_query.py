from lsm.config.models.query import QueryConfig


def test_query_config_sets_local_pool_when_missing() -> None:
    cfg = QueryConfig(k=3, k_rerank=2, local_pool=None)
    assert cfg.local_pool == max(cfg.k * 3, cfg.k_rerank * 4)


def test_query_config_chat_mode_validation() -> None:
    cfg = QueryConfig(chat_mode="chat")
    cfg.validate()


def test_query_config_cache_validation() -> None:
    cfg = QueryConfig(enable_query_cache=True, query_cache_ttl=10, query_cache_size=5)
    cfg.validate()
