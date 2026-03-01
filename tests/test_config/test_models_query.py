from lsm.config.models.query import QueryConfig


def test_query_config_retrieval_profile_default() -> None:
    cfg = QueryConfig(k=3)
    assert cfg.retrieval_profile == "hybrid_rrf"


def test_query_config_chat_mode_validation() -> None:
    cfg = QueryConfig(chat_mode="chat")
    cfg.validate()


def test_query_config_llm_server_cache_fields() -> None:
    cfg = QueryConfig(enable_llm_server_cache=True)
    cfg.validate()
    assert cfg.enable_llm_server_cache is True


def test_query_config_default_llm_server_cache_enabled() -> None:
    cfg = QueryConfig()
    assert cfg.enable_llm_server_cache is True
