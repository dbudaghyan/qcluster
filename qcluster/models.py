import os

from loguru import logger
from sentence_transformers.SentenceTransformer import SentenceTransformer

logger.info("Loading the SentenceTransformer model...")
MODEL = SentenceTransformer(os.environ['SENTENCE_TRANSFORMERS_MODEL'])
