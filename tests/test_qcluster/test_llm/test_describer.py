from unittest.mock import patch

from qcluster.custom_types import ClusterDescription
from qcluster.llm.describer import get_description


@patch.dict("os.environ", {"OLLAMA_MODEL": "test_model"})
@patch("qcluster.llm.describer.query_llm")
def test_get_description(mock_query_llm):
    """
    Tests that get_description correctly calls query_llm and returns
     a ClusterDescription object.
    """
    # Arrange
    document = "This is a test document."
    template_name = "test_template"
    expected_cluster_description = ClusterDescription(
        title="Test Title", description="Test Description"
    )
    mock_query_llm.return_value = expected_cluster_description

    # Act
    result = get_description(document=document, template_name=template_name)

    # Assert
    assert result == expected_cluster_description
    mock_query_llm.assert_called_once_with(
        template_name=template_name,
        data={"document": document},
        model="test_model",
        output_model=ClusterDescription,
    )


@patch.dict("os.environ", {"OLLAMA_MODEL": "test_model"})
@patch("qcluster.llm.describer.query_llm")
def test_get_description_with_limit(mock_query_llm):
    """
    Tests that get_description correctly truncates the document when a limit is provided.
    """
    # Arrange
    document = "This is a long document that will be truncated."
    template_name = "test_template"
    limit = 10
    truncated_document = document[:limit]
    expected_cluster_description = ClusterDescription(
        title="Test Title", description="Test Description"
    )
    mock_query_llm.return_value = expected_cluster_description

    # Act
    result = get_description(
        document=document, template_name=template_name, limit=limit
    )

    # Assert
    assert result == expected_cluster_description
    mock_query_llm.assert_called_once_with(
        template_name=template_name,
        data={"document": truncated_document},
        model="test_model",
        output_model=ClusterDescription,
    )
