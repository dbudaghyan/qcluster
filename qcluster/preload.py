import os
import random

import numpy as np
import torch
from dotenv import load_dotenv
from loguru import logger
from sentence_transformers.SentenceTransformer import SentenceTransformer

load_dotenv()
seed = 42  # You can use any integer as the seed

# Set seeds for reproducibility
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
if torch.mps.is_available():
    torch.mps.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

# Disable non-deterministic CuDNN operations
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Import for preloading models


logger.info("Loading the SentenceTransformer model...")
MODEL = SentenceTransformer(os.environ["SENTENCE_TRANSFORMERS_MODEL"])
