import os
from pathlib import Path

from loguru import logger

from . import preload  # noqa: F401
from .preload import MODEL  # noqa: F401

ROOT_DIR = Path(__file__).parent
PROJECT_DIR = ROOT_DIR.parent

# Set up the logger to write to a file
logger.add(
    Path(PROJECT_DIR, "logs", "qcluster.log"),
    rotation="1 MB",
    retention="10 days",
    level="DEBUG",
    format="{time} {level} {message}",
)
# Ensure the logs directory exists
os.makedirs(Path(ROOT_DIR, "logs"), exist_ok=True)


__version__ = "0.2.0"

REQUIRED_ENV_VARIABLES = [
    "TOKENIZERS_PARALLELISM",
    "OLLAMA_MODEL",
    "OLLAMA_REPORTING_MODEL",
    "OLLAMA_HOST",
    "SENTENCE_TRANSFORMERS_MODEL",
    "EVALUATION_RESULTS_DIR",
    'EVALUATION_REPORT_PROMPT_TEMPLATE',
    'DESCRIPTION_PROMPT_TEMPLATE'
]


def check_required_env_variables():
    """
    Checks if all required environment variables are set.
    Raises an error if any variable is missing.
    """
    missing_vars = [var for var in REQUIRED_ENV_VARIABLES if not os.getenv(var)]
    if missing_vars:
        raise EnvironmentError(
            f"Missing required environment variables: {', '.join(missing_vars)}"
        )


def get_tqdm():
    """
    Determines and returns the appropriate tqdm progress bar implementation
    based on the execution environment.
    """
    try:
        # noinspection PyProtectedMember
        from IPython import get_ipython

        # Check for Jupyter Notebook or JupyterLab
        shell = get_ipython().__class__.__name__
        if shell == "ZMQInteractiveShell":
            from tqdm.notebook import tqdm

            return tqdm
        # Check for IPython console
        elif shell == "TerminalInteractiveShell":
            from tqdm import tqdm

            return tqdm
        # Check for Google Colab
        elif "google.colab" in str(get_ipython()):
            from tqdm.notebook import tqdm

            return tqdm
        else:
            from tqdm import tqdm

            return tqdm
    except (NameError, ImportError):
        # Fallback for standard Python interpreter
        from tqdm import tqdm

        return tqdm


# Get the appropriate tqdm instance
tqdm = get_tqdm()
check_required_env_variables()
