from pathlib import Path

# Path settings
# DATA_DIR = Path(__file__).parent.parent / "data"
DATA_DIR = Path(__file__).parent / "data"
VECTOR_DIR = Path(__file__).parent / "vector_db"

# Default settings
DEFAULT_CHUNK_SIZE = 600
DEFAULT_CHUNK_OVERLAP = 50
DEFAULT_TOP_K = 5
DEFAULT_EMBEDDING_MODEL = "solar-embedding-1-large-query"
DEFAULT_LLM_MODEL = "solar-mini"