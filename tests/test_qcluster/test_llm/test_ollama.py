import os
from unittest.mock import patch, MagicMock

from qcluster.llm.ollama import query_llm


@patch("qcluster.llm.ollama.read_prompt_template")
@patch.dict(os.environ, {"OLLAMA_HOST": "mock_host"})
@patch("qcluster.llm.ollama.Client")
def test_query_llm(mock_client_constructor, mock_read_prompt_template):
    """
    Tests the query_llm function with a mocked ollama client.
    """
    # Mock the template reading
    mock_template = MagicMock()
    mock_template.render.return_value = "Rendered prompt"
    mock_read_prompt_template.return_value = mock_template

    # Mock the entire ollama client and its response
    mock_ollama_client = MagicMock()
    mock_response = MagicMock()
    mock_response.message.content = "Mocked response from LLM"
    mock_ollama_client.chat.return_value = mock_response
    mock_client_constructor.return_value = mock_ollama_client

    # Act
    result = query_llm(
        template_name="test_template", data={"key": "value"}, model="test_model"
    )

    # Assert
    # Verify that the client was initialized and used as expected
    mock_client_constructor.assert_called_once_with(host="mock_host")
    mock_ollama_client.chat.assert_called_once_with(
        messages=[{"role": "user", "content": "Rendered prompt"}],
        model="test_model",
        format=None,
        options={"temperature": 0.0, "max_tokens": 2048},
    )

    # Verify that the template was rendered
    mock_read_prompt_template.assert_called_once_with("test_template")
    mock_template.render.assert_called_once_with(key="value")

    # Verify that the function returns the content of the mocked response
    assert result == "Mocked response from LLM"
