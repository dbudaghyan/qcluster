import csv
import json
import os
import traceback
from os import PathLike
from pathlib import Path
from zipfile import ZipFile
from zlib import DEFLATED

from loguru import logger
from pycm import ConfusionMatrix
from sklearn.metrics.cluster import (
    adjusted_rand_score,
    completeness_score,
    homogeneity_score,
    v_measure_score,
)

from qcluster import REQUIRED_ENV_VARIABLES
from qcluster.custom_types import ClusterType, IdToCategoryResultType
from qcluster.datamodels.evaluation_result import EvaluationResult
from qcluster.datamodels.instruction import InstructionCollection
from qcluster.git_utils import get_git_diff


def evaluate_results(id_to_categories: IdToCategoryResultType) -> ConfusionMatrix:
    """
    Evaluates the clustering results by creating a confusion matrix.

    Args:
        id_to_categories (IdToResultsType): A dictionary mapping IDs to tuples of
                                             predicted and actual categories.

    Returns:
        ConfusionMatrix: A confusion matrix object containing the evaluation results.
    """
    y_pred = []
    y_true = []
    for id_, (actual_category, predicted_category) in id_to_categories.items():
        y_pred.append(predicted_category)
        y_true.append(actual_category)
    cm = ConfusionMatrix(actual_vector=y_true, predict_vector=y_pred)
    cm.normalize = True
    return cm


def cluster_to_class_similarity_measures(act: list, pred: list):
    """
    Calculates clustering similarity measures between actual and predicted clusters.

    Args:
        act (list): Actual cluster labels.
        pred (list): Predicted cluster labels.

    Returns:
        dict: A dictionary containing homogeneity, completeness, and V-measure scores.
    """
    return {
        "homogeneity": homogeneity_score(act, pred),
        "completeness": completeness_score(act, pred),
        "v_measure": v_measure_score(act, pred),
        "ari": adjusted_rand_score(act, pred),
    }


def store_results(
    cm: ConfusionMatrix,
    cluster_to_class_scores,
    storage_path: PathLike,
    instructions_by_cluster: dict,
):
    """
    Stores the evaluation results of a confusion matrix and cluster-to-class scores
    in a specified directory. Results are saved in various formats including CSV,
    HTML, and PyCM object. Logging provides feedback on the status of the
    saving process for each format.
    Args:
        cm (ConfusionMatrix): The confusion matrix containing evaluation results.
        cluster_to_class_scores (dict): A dictionary mapping cluster measures
         to their scores.
        storage_path (PathLike): The path where the results will be stored.
        instructions_by_cluster (dict): A dictionary mapping clusters to their
         corresponding instructions.
    """
    # Save the full git diff if available
    try:
        save_the_full_git_diff_if_any(storage_path)
    except Exception as e:
        logger.error(f"Failed to save git diff: {e}")
    storage_path = Path(storage_path)
    storage_path.mkdir(parents=True, exist_ok=True)
    logger.info(f"Storing evaluation results to {storage_path}...")
    # Save cluster to class scores
    csv_content = create_cluster_to_category_evaluation_csv(cluster_to_class_scores)
    with open(storage_path / "cluster_to_class_scores.csv", "w", newline="") as f:
        f.write(csv_content)
    status = cm.save_csv(str(storage_path / "stats"))
    if status["Status"] is False:
        logger.error(f"Failed to save statistics: {status['Message']}")
    else:
        logger.info(f"Statistics saved at {storage_path / 'stats.csv'}")
    status = cm.save_html(str(storage_path / "results"))
    if status.get("Status") is False:
        logger.error(f"Failed to save HTML statistics: {status.get('Message')}")
    else:
        logger.info(f"HTML statistics saved at {storage_path / 'html_results'}")
    status = cm.save_obj(str(storage_path / "pycm"))
    # zip pycm.obj
    try:
        with ZipFile(storage_path / "pycm.zip", "w", compression=DEFLATED) as zipf:
            zipf.write(storage_path / "pycm.obj", arcname="pycm.obj")
        # Remove the original pycm file
        (storage_path / "pycm.obj").unlink(missing_ok=True)
        logger.info(f"PyCM object zipped and saved at {storage_path / 'pycm.zip'}")
    except Exception as e:
        logger.error(f"Failed to zip PyCM object: {e}")
    if status.get("Status") is False:
        logger.error(f"Failed to save PyCM object: {status.get('Message')}")
    else:
        logger.info(f"PyCM object saved at {storage_path / 'pycm'}")
    status = cm.save_stat(str(storage_path / "stats"), sparse=True, summary=True)
    if status.get("Status") is False:
        logger.error(f"Failed to save stats: {status.get('Message')}")
    else:
        logger.info(f"Stats saved at {storage_path / 'stats'}")

    # Save the current notebook or script
    # try:
    save_notebook_or_the_currently_running_script(storage_path)
    # except Exception as e:
    #     logger.error(f"Failed to save notebook or script: {e}")
    save_env_variables(storage_path)
    evaluation_results = EvaluationResult.from_folder_path(storage_path)
    save_cluster_data(
        instructions_by_cluster=instructions_by_cluster,
        file_path=storage_path / "clusters.json",
    )
    evaluation_results.add_final_report()


def create_cluster_to_category_evaluation_csv(
    cluster_to_class_scores: dict[str, float],
):
    """
    Creates a CSV file in-memory containing cluster-to-class evaluation scores.

    Args:
        cluster_to_class_scores (dict): A dictionary mapping cluster measures
         to their scores.
    Returns:
        str: The CSV content as a string.
    """
    import io

    output = io.StringIO()
    writer = csv.writer(output)
    descriptions = {
        "homogeneity": "Homogeneity score measures how much each cluster contains"
        " only members of a single class.",
        "completeness": "Completeness score measures how much all members"
        " of a given class"
        " are assigned to the same cluster.",
        "v_measure": "V-measure is the harmonic mean of homogeneity and completeness.",
        "ari": "Adjusted Rand Index (ARI) is a measure of the similarity"
        " between two data clustering results.",
    }
    writer.writerow(["Measure", "Score", "Description"])
    for measure, score in cluster_to_class_scores.items():
        description = descriptions.get(measure, "No description available")
        writer.writerow([measure, score, description])
    return output.getvalue()


def save_notebook_or_the_currently_running_script(storage_path: PathLike):
    """
    Saves the current Jupyter notebook or Python script to a specified path.

    Args:
        storage_path (PathLike): The path where the notebook or script will be saved.
    """
    try:
        # noinspection PyProtectedMember,PyUnresolvedReferences
        from IPython import get_ipython

        ipython = get_ipython()
    except ImportError:
        ipython = None
    if ipython is not None:
        try:
            from IPython.display import FileLink

            notebook_path = ipython.get_parent()["metadata"]["path"]
            storage_path = Path(storage_path) / Path(notebook_path).name
            with open(storage_path, "w") as f:
                f.write(ipython.get_notebook().to_string())
            logger.info(f"Notebook saved to {storage_path}")
            return FileLink(str(storage_path))
        except Exception as e:
            logger.error(f"Failed to save notebook: {e}")
            logger.error(f"Full traceback: {traceback.format_exc()}")
    try:
        logger.warning(
            "Could not save to a Jupyter environment,"
            " will save the main running script instead."
        )
        import shutil
        import sys

        main_module = sys.modules.get("__main__")
        if not (main_module and hasattr(main_module, "__file__")):
            logger.warning(
                "Could not determine the entrypoint script." " No script will be saved."
            )
            return None
        main_script_path = main_module.__file__
        new_name = f"main_script_{Path(main_script_path).name}"
        destination_path = Path(storage_path) / new_name
        shutil.copy(main_script_path, destination_path)
        logger.info(f"Script saved to {destination_path}")
        return destination_path
    except Exception as e:
        logger.error(f"Failed to save the main script: {e}")
        return None


def save_the_full_git_diff_if_any(storage_path: PathLike):
    """
    Saves the full git diff of the current repository to a specified path.

    Args:
        storage_path (PathLike): The path where the git diff will be saved.
    """
    storage_path = Path(storage_path)
    storage_path.mkdir(parents=True, exist_ok=True)
    try:
        git_diff = get_git_diff()
        with open(storage_path / "git_diff.txt", "w") as f:
            f.write(git_diff)
        logger.info(f"Git diff saved to {storage_path / 'git_diff.txt'}")
    except Exception as e:
        logger.error(f"Failed to save git diff: {e}")
        logger.error(f"Full traceback: {traceback.format_exc()}")


def save_env_variables(storage_path: PathLike):
    """
    Saves the current environment variables to a specified path.

    Args:
        storage_path (PathLike): The path where the environment variables will be saved.
    """
    storage_path = Path(storage_path)
    storage_path.mkdir(parents=True, exist_ok=True)
    env_file_path = storage_path / ".env"
    with open(env_file_path, "w") as f:
        for key, value in os.environ.items():
            if key not in REQUIRED_ENV_VARIABLES:
                continue
            f.write(f"{key}={value}\n")
    logger.info(f"Environment variables saved to {env_file_path}")


def deserialize_from_cm_obj_zip(zip_path: PathLike) -> ConfusionMatrix:
    """
    Deserializes a ConfusionMatrix object from a zip file.

    Args:
        zip_path (PathLike): The path to the zip file containing the PyCM object.

    Returns:
        ConfusionMatrix: The deserialized ConfusionMatrix object.
    """
    with ZipFile(zip_path, "r", compression=DEFLATED) as zipf:
        with zipf.open("pycm.obj", "r") as f:
            cm = ConfusionMatrix(file=f)
    return cm


def save_cluster_data(
    instructions_by_cluster: dict[ClusterType, InstructionCollection],
    file_path: PathLike,
):
    """
    Saves cluster data to a JSON file.
    """
    logger.info(f"Saving cluster data to {file_path}")
    clusters_data = []
    for instruction_collection in instructions_by_cluster.values():
        clusters_data.append(
            {
                "name": instruction_collection.title,
                "description": instruction_collection.description,
                "count": instruction_collection.count,
            }
        )
    with open(file_path, "w") as f:
        json.dump(clusters_data, f, indent=4)
    logger.info(f"Cluster data saved to {file_path}")
