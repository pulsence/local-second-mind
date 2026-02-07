"""
Default values for configuration models.
"""

DEFAULT_EXTENSIONS = {
    ".txt", ".md", ".rst",
    ".pdf",
    ".docx",
    ".html", ".htm",
}

DEFAULT_EXCLUDE_DIRS = {
    ".git", ".hg", ".svn",
    "__pycache__",
    ".venv", "venv",
    "node_modules",
}

DEFAULT_COLLECTION = "local_kb"
DEFAULT_EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_DEVICE = "cpu"
DEFAULT_BATCH_SIZE = 32
DEFAULT_CHROMA_FLUSH_INTERVAL = 2000
DEFAULT_VDB_PROVIDER = "chromadb"
DEFAULT_CHROMA_HNSW_SPACE = "cosine"

DEFAULT_CHUNK_SIZE = 1800
DEFAULT_CHUNK_OVERLAP = 200

DEFAULT_K = 12
DEFAULT_K_RERANK = 6
DEFAULT_MAX_PER_FILE = 2
DEFAULT_MIN_RELEVANCE = 0.25

DEFAULT_LLM_TEMPERATURE = 0.7
DEFAULT_LLM_MAX_TOKENS = 2000

# Well-known embedding models and their output dimensions
WELL_KNOWN_EMBED_MODELS: dict[str, int] = {
    # sentence-transformers models
    "sentence-transformers/all-MiniLM-L6-v2": 384,
    "sentence-transformers/all-MiniLM-L12-v2": 384,
    "sentence-transformers/all-mpnet-base-v2": 768,
    "sentence-transformers/all-distilroberta-v1": 768,
    "sentence-transformers/paraphrase-MiniLM-L6-v2": 384,
    "sentence-transformers/paraphrase-mpnet-base-v2": 768,
    "sentence-transformers/multi-qa-MiniLM-L6-cos-v1": 384,
    "sentence-transformers/multi-qa-mpnet-base-cos-v1": 768,
    "sentence-transformers/multi-qa-mpnet-base-dot-v1": 768,
    # Multilingual models
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2": 384,
    "sentence-transformers/paraphrase-multilingual-mpnet-base-v2": 768,
    "sentence-transformers/distiluse-base-multilingual-cased-v2": 512,
    # Instructor models
    "hkunlp/instructor-large": 768,
    "hkunlp/instructor-xl": 768,
    "hkunlp/instructor-base": 768,
    # E5 models
    "intfloat/e5-small-v2": 384,
    "intfloat/e5-base-v2": 768,
    "intfloat/e5-large-v2": 1024,
    "intfloat/multilingual-e5-small": 384,
    "intfloat/multilingual-e5-base": 768,
    "intfloat/multilingual-e5-large": 1024,
    # BGE models
    "BAAI/bge-small-en-v1.5": 384,
    "BAAI/bge-base-en-v1.5": 768,
    "BAAI/bge-large-en-v1.5": 1024,
    # GTE models
    "thenlper/gte-small": 384,
    "thenlper/gte-base": 768,
    "thenlper/gte-large": 1024,
    # Nomic
    "nomic-ai/nomic-embed-text-v1.5": 768,
}
