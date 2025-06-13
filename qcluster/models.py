from loguru import logger
from sentence_transformers.SentenceTransformer import SentenceTransformer

logger.info("Loading the SentenceTransformer model...")
MODEL = SentenceTransformer('all-MiniLM-L6-v2')
