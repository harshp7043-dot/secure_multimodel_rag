
import os
from pathlib import Path

from dotenv import load_dotenv


# Project paths
PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR
STORAGE_DIR = PROJECT_DIR

USERS_FILE = STORAGE_DIR / "users.json"
AUDIT_FILE = STORAGE_DIR / "audit_log.jsonl"

load_dotenv(PROJECT_DIR / ".env")


# LLM configuration
LLM_BACKEND = os.getenv("LLM_BACKEND", "local").lower()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL", "qwen/qwen3.8-27b")

HF_TOKEN = os.getenv("HF_TOKEN")
HF_MODEL = os.getenv(
    "HF_MODEL",
    "Qwen/Qwen3-VL-30B-A3B-Instruct"
)

LOCAL_MODEL = os.getenv(
    "LOCAL_MODEL",
    "Qwen/Qwen2-VL-2B-Instruct"
)

PDF_PATH = Path(
    os.getenv(
        "PDF_PATH",
        str(PROJECT_DIR / "company_secure_sample.pdf")
    )
)

# Authentication
SESSION_TIMEOUT_SEC = 10 * 60
MAX_FAILED_ATTEMPTS = 3
LOCKOUT_SEC = 60

# Classification
PUBLIC = "PUBLIC"
HR = "HR"
FINANCE = "FINANCE"
EXECUTIVE = "EXECUTIVE"

ALL_LEVELS = {PUBLIC, HR, FINANCE, EXECUTIVE}

ROLE_PERMISSIONS = {
    "anonymous": {PUBLIC},
    "employee": {PUBLIC},
    "hr": {PUBLIC, HR},
    "finance": {PUBLIC, FINANCE},
    "executive": {PUBLIC, HR, FINANCE, EXECUTIVE}
}

# Retrieval
TEXT_MIN_SCORE = 0.28
IMAGE_MIN_SCORE = 0.22

TEXT_TOP_K = 8
IMAGE_TOP_K = 3

MAX_CONTEXT_CHUNKS = 5
MAX_CONTEXT_IMAGES = 3

CHUNK_SIZE = 500
CHUNK_OVERLAP = 100

# Responses
NO_INFO = (
    "I couldn't find that information "
    "in the documents available to you."
)

LOGIN_FAILED = (
    "Access was not granted, so I can only help "
    "with general company information."
)
