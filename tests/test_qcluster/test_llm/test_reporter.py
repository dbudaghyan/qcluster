import os
import unittest
from unittest.mock import patch, Mock

from qcluster import ROOT_DIR
from qcluster.llm.reporter import create_report


class TestReporter(unittest.TestCase):

    def setUp(self):
        template_path = ROOT_DIR / "templates"
        self.template_name = "test_template"
        self.template_content = """
        This is a test template for reporting.
        """
        self.template_path = os.path.join(template_path, f"{self.template_name}.j2")
        with open(self.template_path, "w") as f:
            f.write(self.template_content)

    def tearDown(self):
        # Clean up the template file after each test
        if os.path.exists(self.template_path):
            os.remove(self.template_path)

    @patch("qcluster.llm.reporter.query_llm")
    def test_create_report(self, mock_query_llm):
        """
        Tests the create_report function.
        """
        # Arrange
        os.environ["OLLAMA_REPORTING_MODEL"] = "test_model"

        # Mock EvaluationResult
        mock_eval_result = Mock()
        mock_eval_result.to_template_args.return_value = {
            "num_samples": 100,
            "labels": ["label1", "label2"],
            "clustering_summary_html": "<p>Summary</p>",
            "additional_metrics": {"metric1": 0.95, "metric2": 0.85},
            "cluster_json": '{"clusters": [{"id": 1, "size": 50}]}',
        }
        # Configure the mock for query_llm
        expected_report = "This is a test report."
        mock_query_llm.return_value = expected_report
        template_name = "test_template"

        # Act
        report = create_report(template_name, mock_eval_result)
        # Assert
        self.assertEqual(report, expected_report)
        # Cleanup
        del os.environ["OLLAMA_REPORTING_MODEL"]


if __name__ == "__main__":
    unittest.main()
