from pycm import ConfusionMatrix

from qcluster.custom_types import IdToCategoryResultType
from sklearn.metrics.cluster import (
    homogeneity_score,
    completeness_score,
    v_measure_score
)


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


def cluster_to_class_similarity_measures(act, pred):
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
        "v_measure": v_measure_score(act, pred)
    }
