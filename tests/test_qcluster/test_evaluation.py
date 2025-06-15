import json
import os
import zipfile
from pathlib import Path
from unittest.mock import MagicMock, patch
from zlib import DEFLATED

import pytest
from pycm import ConfusionMatrix

from qcluster.custom_types import IdToCategoryResultType
from qcluster.datamodels.instruction import InstructionCollection
from qcluster.evaluation import (
    cluster_to_class_similarity_measures,
    create_cluster_to_category_evaluation_csv,
    deserialize_from_cm_obj_zip,
    evaluate_results,
    save_cluster_data,
    save_env_variables,
    save_notebook_or_the_currently_running_script,
    save_the_full_git_diff_if_any,
    store_results,
)


@pytest.fixture
def id_to_categories_fixture() -> IdToCategoryResultType:
    return {
        0: ("CAT_A", "CAT_A"),
        1: ("CAT_A", "CAT_B"),
        2: ("CAT_B", "CAT_B"),
        3: ("CAT_B", "CAT_A"),
        4: ("CAT_C", "CAT_C"),
    }


def test_evaluate_results(id_to_categories_fixture):
    cm = evaluate_results(id_to_categories_fixture)
    assert isinstance(cm, ConfusionMatrix)
    assert cm.actual_vector == ["CAT_A", "CAT_A", "CAT_B", "CAT_B", "CAT_C"]
    assert cm.predict_vector == ["CAT_A", "CAT_B", "CAT_B", "CAT_A", "CAT_C"]


def test_cluster_to_class_similarity_measures():
    act = [0, 0, 1, 1]
    pred = [0, 0, 1, 1]
    scores = cluster_to_class_similarity_measures(act, pred)
    assert scores["homogeneity"] == 1.0
    assert scores["completeness"] == 1.0
    assert scores["v_measure"] == 1.0
    assert scores["ari"] == 1.0


def test_create_cluster_to_category_evaluation_csv():
    scores = {"homogeneity": 1.0, "completeness": 0.5}
    csv_content = create_cluster_to_category_evaluation_csv(scores)
    assert "homogeneity,1.0" in csv_content
    assert "completeness,0.5" in csv_content
    assert "Measure,Score,Description" in csv_content


@patch("qcluster.evaluation.save_the_full_git_diff_if_any")
@patch("qcluster.evaluation.create_cluster_to_category_evaluation_csv")
@patch("qcluster.evaluation.save_notebook_or_the_currently_running_script")
@patch("qcluster.evaluation.save_env_variables")
@patch("qcluster.evaluation.save_cluster_data")
@patch("qcluster.evaluation.EvaluationResult")
@patch("pycm.ConfusionMatrix")
def test_store_results(
    mock_cm,
    mock_eval_result,
    mock_save_cluster_data,
    mock_save_env,
    mock_save_notebook,
    mock_create_csv,
    mock_save_git_diff,
    tmp_path,
):
    mock_cm_instance = mock_cm.return_value
    mock_cm_instance.save_csv.return_value = {"Status": True}
    mock_cm_instance.save_html.return_value = {"Status": True}
    mock_cm_instance.save_obj.return_value = {"Status": True}
    mock_cm_instance.save_stat.return_value = {"Status": True}
    mock_create_csv.return_value = "csv,content"

    cluster_scores = {"homogeneity": 1.0}
    instructions = {"cluster_1": MagicMock()}

    with (
        patch("builtins.open", MagicMock()),
        patch("zipfile.ZipFile", MagicMock()),
        patch.object(Path, "unlink", MagicMock()),
    ):
        store_results(mock_cm_instance, cluster_scores, tmp_path, instructions)

    mock_save_git_diff.assert_called_once_with(tmp_path)
    mock_create_csv.assert_called_once_with(cluster_scores)
    mock_save_notebook.assert_called_once_with(tmp_path)
    mock_save_env.assert_called_once_with(tmp_path)
    mock_save_cluster_data.assert_called_once_with(
        instructions_by_cluster=instructions,
        file_path=tmp_path / "clusters.json",
    )
    mock_eval_result.from_folder_path.assert_called_once_with(tmp_path)


def test_save_the_full_git_diff_if_any(tmp_path):
    with patch("qcluster.git_utils.get_git_diff") as mock_popen:
        mock_popen.return_value.read.return_value = "git diff content"
        save_the_full_git_diff_if_any(tmp_path)
        expected_file = tmp_path / "git_diff.txt"
        assert expected_file.exists()


@patch.dict(
    os.environ,
    {"MY_VAR": "my_value", "ANOTHER_VAR": "another_value"},
    clear=True,
)
@patch("qcluster.evaluation.REQUIRED_ENV_VARIABLES", ["MY_VAR"])
def test_save_env_variables(tmp_path):
    save_env_variables(tmp_path)
    expected_file = tmp_path / ".env"
    assert expected_file.exists()
    content = expected_file.read_text()
    assert "MY_VAR=my_value" in content
    assert "ANOTHER_VAR" not in content


def test_deserialize_from_cm_obj_zip(tmp_path):
    zip_path = tmp_path / "pycm.zip"
    temp_obj_path_wo_prefix = tmp_path / "pycm"
    temp_obj_path = temp_obj_path_wo_prefix.with_suffix(".obj")
    cm = ConfusionMatrix([1, 2, 3], [3, 2, 1])
    cm.save_obj(str(temp_obj_path_wo_prefix))
    obj_content = temp_obj_path.read_bytes()
    with zipfile.ZipFile(zip_path, "w", compression=DEFLATED) as zf:
        zf.writestr("pycm.obj", obj_content)
    with patch("zipfile.ZipFile", return_value=zipfile.ZipFile(zip_path, "r")):
        cm_deserialized = deserialize_from_cm_obj_zip(zip_path)
    assert isinstance(cm_deserialized, ConfusionMatrix)


def test_save_cluster_data(tmp_path):
    mock_collection = MagicMock(spec=InstructionCollection)
    mock_collection.title = "Test Cluster"
    mock_collection.description = "A description"
    mock_collection.count = 10
    instructions_by_cluster = {"cluster_1": mock_collection}
    file_path = tmp_path / "clusters.json"

    save_cluster_data(instructions_by_cluster, file_path)

    assert file_path.exists()
    with open(file_path, "r") as f:
        data = json.load(f)
    assert data == [
        {
            "name": "Test Cluster",
            "description": "A description",
            "count": 10,
        }
    ]


@patch("sys.modules")
@patch("shutil.copy")
def test_save_notebook_or_the_currently_running_script_as_script(
    mock_copy, mock_sys_modules, tmp_path
):
    with patch("qcluster.evaluation.get_ipython", return_value=None):
        main_module = MagicMock()
        main_module.__file__ = "/path/to/script.py"
        mock_sys_modules.get.return_value = main_module

        save_notebook_or_the_currently_running_script(tmp_path)

        expected_dest = tmp_path / "main_script_script.py"
        mock_copy.assert_called_once_with("/path/to/script.py", expected_dest)
