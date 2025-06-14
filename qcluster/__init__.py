from pathlib import Path
from . import preload
from .preload import MODEL

ROOT_DIR = Path(__file__).parent
__version__ = "0.2.0"

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
        if shell == 'ZMQInteractiveShell':
            from tqdm.notebook import tqdm
            return tqdm
        # Check for IPython console
        elif shell == 'TerminalInteractiveShell':
            from tqdm import tqdm
            return tqdm
        # Check for Google Colab
        elif 'google.colab' in str(get_ipython()):
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
