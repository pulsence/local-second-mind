from lsm.config.models.query import QueryConfig


def test_query_config_sets_local_pool_when_missing() -> None:
    cfg = QueryConfig(k=3, k_rerank=2, local_pool=None)
    assert cfg.local_pool == max(cfg.k * 3, cfg.k_rerank * 4)

