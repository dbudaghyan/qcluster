import os
import random
from pathlib import Path

import jinja2
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
MODEL = SentenceTransformer(os.environ['SENTENCE_TRANSFORMERS_MODEL'])


def read_prompt_template(template_path, template_name: str) -> jinja2.Template:
    """
    Reads a Jinja2 template from the the templates directory.
    """
    loader = jinja2.FileSystemLoader(os.path.dirname(template_path))
    env = jinja2.Environment(loader=loader)
    return env.get_template(template_name)

template_path = os.path.join(
    Path(__file__).parent,
    'templates',
    'description_prompt.j2'
)

logger.info(f"Loading the prompt template from {template_path}...")
LLM_TEMPLATE = read_prompt_template(
    template_path=template_path,
    template_name='description_prompt.j2'
)
