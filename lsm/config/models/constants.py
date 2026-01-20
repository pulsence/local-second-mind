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
