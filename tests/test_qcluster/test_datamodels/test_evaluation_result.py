import os
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from zipfile import ZipFile
from zlib import DEFLATED

from pycm import ConfusionMatrix

from qcluster.custom_types import ClusteringReport
from qcluster.datamodels.evaluation_result import EvaluationResult
from qcluster.datamodels.filesystem import File


class TestEvaluationResult(unittest.TestCase):
    def setUp(self):
        """Set up a temporary directory and dummy files for testing."""
        self.root_dir = Path(tempfile.mkdtemp())
        self.test_dir = self.root_dir / "test_evaluation_results_temp"
        self.test_dir.mkdir(exist_ok=True)

        # Create dummy files required by EvaluationResult
        (self.test_dir / "cluster_to_class_scores.csv").write_text(
            "col1,col2\n1,2\n3,4"
        )
        (self.test_dir / "git_diff.txt").write_text("git diff content")
        (self.test_dir / "entrypoint.py").write_text("# entrypoint.py")
        (self.test_dir / "results.html").write_text("<h1>Results</h1>")
        (self.test_dir / "stats_matrix.csv").write_text("a,b,c\n1,2,3\n4,5,6\n7,8,9")
        (self.test_dir / "stats.csv").write_text("label1,label2\n10,20\n30,40")
        (self.test_dir / "clusters.json").write_text('{"cluster1": [1, 2]}')

        # Create a dummy ConfusionMatrix object and save it
        y_actual = [1, 2, 3, 1, 2, 3]
        y_predict = [1, 2, 3, 3, 2, 1]
        cm = ConfusionMatrix(y_actual, y_predict)

        pycm_obj_path = self.test_dir / "pycm.obj"
        cm.save_obj(str(self.test_dir / "pycm"))  # This creates pycm.obj
        cm.save_stat(str(self.test_dir / "stats"), summary=True)

        # Now zip it and remove the original
        with ZipFile(self.test_dir / "pycm.zip", "w", compression=DEFLATED) as zipf:
            zipf.write(pycm_obj_path, arcname="pycm.obj")
        pycm_obj_path.unlink()

    def tearDown(self):
        """Remove the temporary directory and its contents after tests."""
        shutil.rmtree(self.test_dir)

    def test_from_folder_path(self):
        """Test creating an EvaluationResult from a folder path."""
        evaluation_result = EvaluationResult.from_folder_path(self.test_dir)
        self.assertIsInstance(evaluation_result, EvaluationResult)
        self.assertEqual(evaluation_result.name, self.test_dir.name)
        self.assertEqual(evaluation_result.path, self.test_dir)
        self.assertIsNone(evaluation_result.final_report)

    def test_from_folder_path_with_final_report(self):
        """Test creating an EvaluationResult from a folder path that includes
        a final report."""
        (self.test_dir / EvaluationResult.final_report_filename()).write_text(
            "This is a final report."
        )
        evaluation_result = EvaluationResult.from_folder_path(self.test_dir)
        self.assertIsNotNone(evaluation_result.final_report)
        self.assertEqual(
            evaluation_result.final_report.name,
            EvaluationResult.final_report_filename(),
        )
        self.assertEqual(
            evaluation_result.final_report.content, "This is a final report."
        )

    def test_cm_cached_property(self):
        """Test the lazy-loaded confusion matrix property."""
        evaluation_result = EvaluationResult.from_folder_path(self.test_dir)
        self.assertIsInstance(evaluation_result.cm, ConfusionMatrix)

    def test_num_samples_property(self):
        """Test the num_samples property."""
        evaluation_result = EvaluationResult.from_folder_path(self.test_dir)
        self.assertEqual(evaluation_result.num_samples, 3)

    def test_labels_property(self):
        """Test the `labels` property."""
        evaluation_result = EvaluationResult.from_folder_path(self.test_dir)
        self.assertEqual(evaluation_result.labels, ["label1", "label2"])

    def test_clustering_summary_html_property(self):
        """Test the clustering_summary_html property."""
        evaluation_result = EvaluationResult.from_folder_path(self.test_dir)
        self.assertEqual(evaluation_result.clustering_summary_html, "<h1>Results</h1>")

    def test_additional_metrics_property(self):
        """Test the additional_metrics property."""
        evaluation_result = EvaluationResult.from_folder_path(self.test_dir)
        expected_metrics = str([{"col1": 1, "col2": 2}, {"col1": 3, "col2": 4}])
        self.assertEqual(evaluation_result.additional_metrics, expected_metrics)

    def test_cluster_json_property(self):
        """Test the cluster_json property."""
        evaluation_result = EvaluationResult.from_folder_path(self.test_dir)
        self.assertEqual(evaluation_result.cluster_json, '{"cluster1": [1, 2]}')

    def test_to_template_args(self):
        """Test the to_template_args method."""
        evaluation_result = EvaluationResult.from_folder_path(self.test_dir)
        template_args = evaluation_result.to_template_args()
        self.assertEqual(template_args["num_samples"], 3)
        self.assertEqual(template_args["labels"], ["label1", "label2"])
        self.assertEqual(template_args["clustering_summary_html"], "<h1>Results</h1>")
        self.assertEqual(template_args["cluster_json"], '{"cluster1": [1, 2]}')

    @patch("qcluster.datamodels.evaluation_result.create_report")
    def test_add_final_report(self, mock_create_report):
        """Test the add_final_report method."""
        mock_create_report.return_value = "Mocked report content"
        os.environ["EVALUATION_REPORT_PROMPT_TEMPLATE"] = "dummy_template"

        evaluation_result = EvaluationResult.from_folder_path(self.test_dir)
        self.assertIsNone(evaluation_result.final_report)

        report = evaluation_result.add_final_report()

        # Check that the final report file is created and has the correct content
        self.assertIsNotNone(evaluation_result.final_report)
        self.assertIsInstance(evaluation_result.final_report, File)
        self.assertTrue(
            (self.test_dir / EvaluationResult.final_report_filename()).exists()
        )
        self.assertEqual(
            evaluation_result.final_report.content, "Mocked report content"
        )

        # Check the returned report object
        self.assertIsInstance(report, ClusteringReport)
        self.assertEqual(report.report, "Mocked report content")
        self.assertTrue(report.title.startswith("final_report_"))

        # Check that create_report was called correctly
        mock_create_report.assert_called_once_with(
            template_name="dummy_template", evaluation_result=evaluation_result
        )
        del os.environ["EVALUATION_REPORT_PROMPT_TEMPLATE"]
