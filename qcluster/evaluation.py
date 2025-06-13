from pycm import ConfusionMatrix

from qcluster.types import IdToCategoryResultType


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
    for id_, (predicted_category, actual_category) in id_to_categories.items():
        y_pred.append(predicted_category)
        y_true.append(actual_category)
    cm = ConfusionMatrix(actual_vector=y_true, predict_vector=y_pred)
    cm.normalize = True
    return cm
