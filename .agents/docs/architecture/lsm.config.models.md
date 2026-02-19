# lsm.config.models

Description: Config dataclasses split by domain (global/ingest/query/llm/modes/vectordb/agents/chats).
Folder Path: `lsm/config/models/`

## Modules

- [constants.py](../lsm/config/models/constants.py): Default values and WELL_KNOWN_EMBED_MODELS dimension dictionary
- [global_config.py](../lsm/config/models/global_config.py): GlobalConfig dataclass for embed_model, device, batch_size, global_folder
- [ingest.py](../lsm/config/models/ingest.py): IngestConfig dataclass for roots, manifest, chunk_size, etc.
- [query.py](../lsm/config/models/query.py): QueryConfig dataclass for query settings
- [llm.py](../lsm/config/models/llm.py): LLMConfig dataclass for providers and services
- [modes.py](../lsm/config/models/modes.py): ModesConfig dataclass for query modes
- [vectordb.py](../lsm/config/models/vectordb.py): VectorDBConfig dataclass for vector database settings
- [agents.py](../lsm/config/models/agents.py): AgentsConfig dataclass for agent settings, sandbox, memory
- [chats.py](../lsm/config/models/chats.py): ChatsConfig dataclass for chat settings
- [lsm_config.py](../lsm/config/models/lsm_config.py): LSMConfig root dataclass combining all sections
