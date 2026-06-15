"""Central configuration – reads from .env or environment variables."""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

ROOT_DIR = Path(__file__).parent.parent
DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
MODEL_DIR = DATA_DIR / "trained_model"
LOGS_DIR = ROOT_DIR / "logs"

for _d in (RAW_DIR, MODEL_DIR, LOGS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# Paths
DATASET_PATH = RAW_DIR / "category_dataset.csv"
VECTOR_INDEX_PATH = MODEL_DIR / "faiss_index.bin"
VECTOR_META_PATH = MODEL_DIR / "faiss_meta.json"
ML_MODEL_PATH = MODEL_DIR / "ml_classifier.joblib"
LABEL_ENCODER_PATH = MODEL_DIR / "label_encoder.joblib"
VECTORIZER_PATH = MODEL_DIR / "tfidf_vectorizer.joblib"

# LLM — Groq (free) is the default; OpenAI as fallback if key is provided
GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL: str = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

# OpenAI (optional fallback)
OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

# Decision thresholds
ML_CONFIDENCE_THRESHOLD: float = float(os.getenv("ML_CONFIDENCE_THRESHOLD", "0.75"))
FUZZY_MATCH_THRESHOLD: int = int(os.getenv("FUZZY_MATCH_THRESHOLD", "85"))
VECTOR_TOP_K: int = 5
VECTOR_SIMILARITY_THRESHOLD: float = 0.80   # cosine similarity

# Retraining
MIN_SAMPLES_TO_RETRAIN: int = int(os.getenv("MIN_SAMPLES_TO_RETRAIN", "20"))
