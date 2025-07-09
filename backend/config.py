import os
from pathlib import Path

# API Configuration
GROK_API_KEY = os.getenv("GROK_API_KEY", "your-grok-api-key-here")
GROK_BASE_URL = "https://api.x.ai/v1"

# File paths
DATA_DIR = Path("data")
DOCUMENTS_DIR = DATA_DIR / "documents"
CHAT_HISTORY_DIR = DATA_DIR / "chat_history"
EMBEDDINGS_DIR = DATA_DIR / "embeddings"

# Document processing settings
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
ALLOWED_EXTENSIONS = ['.pdf', '.txt', '.docx']
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 100

# Embedding settings
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
SIMILARITY_THRESHOLD = 0.7
MAX_SIMILAR_DOCS = 3

# Chat settings
MAX_HISTORY_MESSAGES = 10
GROK_MODEL = "grok-beta"
MAX_TOKENS = 1000
TEMPERATURE = 0.7

# Create directories if they don't exist
for directory in [DATA_DIR, DOCUMENTS_DIR, CHAT_HISTORY_DIR, EMBEDDINGS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)