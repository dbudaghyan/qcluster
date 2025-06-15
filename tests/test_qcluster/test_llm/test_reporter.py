import os
import unittest
from unittest.mock import patch, Mock

from qcluster.llm.reporter import create_report


class TestReporter(unittest.TestCase):
    @patch("qcluster.llm.ollama.query_llm")
    def test_create_report(self, mock_query_llm):
        """
        Tests the create_report function.
        """
        # Arrange
        os.environ["OLLAMA_REPORTING_MODEL"] = "test_model"

        # Mock EvaluationResult
        mock_eval_result = Mock()
        mock_eval_result.to_template_args.return_value = {"key": "value"}

        # Configure the mock for query_llm
        expected_report = "This is a test report."
        mock_query_llm.return_value = expected_report

        template_name = "test_template"

        # Act
        report = create_report(template_name, mock_eval_result)

        # Assert
        self.assertEqual(report, expected_report)
        mock_query_llm.assert_called_once_with(
            template_name=template_name,
            data={"key": "value"},
            model="test_model",
        )

        # Cleanup
        del os.environ["OLLAMA_REPORTING_MODEL"]


if __name__ == "__main__":
    unittest.main()
