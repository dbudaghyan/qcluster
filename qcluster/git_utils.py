import subprocess
import shutil

from loguru import logger


# --- Securely find the absolute path to the git executable ---
# This check is performed once when the script is loaded.
# If git is not found, the script will not proceed with definitions
# that would fail anyway.
GIT_EXECUTABLE_PATH = shutil.which("git")


def get_git_commit_hash():
    """Safely gets the short git commit hash."""
    if not GIT_EXECUTABLE_PATH:
        raise RuntimeError(
            "Git is not installed or the command" " is not available in the PATH."
        )

    try:
        # The command is passed as a list of arguments to avoid shell injection.
        result = subprocess.run(
            [GIT_EXECUTABLE_PATH, "rev-parse", "--short", "HEAD"],
            capture_output=True,  # Capture stdout and stderr
            text=True,  # Decode output as text using default encoding
            check=True,  # Raise CalledProcessError if the command fails
        )
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Failed to get git commit hash: {e.stderr.strip()}")


def get_git_diff(*args):
    """
    Safely gets the output of "git diff" with optional arguments.

    Args:
        *args: A sequence of strings representing arguments for git diff
               (e.g., 'HEAD~1', '--cached', 'my_file.py').

    Returns:
        The output of the git diff command as a string, or an empty
        string if an error occurs.
    """
    if not GIT_EXECUTABLE_PATH:
        logger.warning("Git is not installed or not available in the PATH.")
        return ""

    try:
        # The base command is combined with any provided arguments.
        command = [GIT_EXECUTABLE_PATH, "diff", *args]
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        return result.stdout
    except subprocess.CalledProcessError as e:
        # Print the error for debugging purposes.
        logger.warning(f"An error occurred while running git diff: {e}")
        # Return an empty string to signify no diff or an error.
        return ""
