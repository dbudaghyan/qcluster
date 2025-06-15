import unittest
from unittest.mock import patch
import torch

from qcluster.algorithms.similarity import get_top_n_similar_embeddings


class TestGetTopNSimilarEmbeddings(unittest.TestCase):

    @patch("qcluster.algorithms.feature_extractors.create_embeddings")
    def test_get_top_n_similar_embeddings(self, mock_create_embedding):
        """
        Tests the basic functionality of get_top_n_similar_embeddings.
        """
        # Arrange
        query = "test query"
        documents = ["doc1", "doc2", "doc3"]
        n = 2

        # Mocking create_embedding to return a torch tensor.
        # shape[0] will be equal to the length of its input (number of strings).
        def create_embedding_side_effect(input_data):
            if isinstance(input_data, str):
                # For a single query string, we assume it's a batch of 1.
                # So the shape of the returned tensor is (1, embedding_dim).
                return torch.tensor([[0.9, 0.8, 0.7]])  # Query embedding
            elif isinstance(input_data, list):
                # For a list of documents, shape[0] is the number of documents.
                embeddings = torch.tensor(
                    [
                        [0.1, 0.2, 0.3],  # doc1
                        [0.9, 0.8, 0.7],  # doc2 (most similar)
                        [0.4, 0.5, 0.6],  # doc3
                    ]
                )
                self.assertEqual(embeddings.shape[0], len(input_data))
                return embeddings
            return None

        mock_create_embedding.side_effect = create_embedding_side_effect
        # Act
        result = get_top_n_similar_embeddings(query, documents, n)
        # Assert
        self.assertEqual(len(result), n)
        self.assertTrue(isinstance(result[0][1], float))

    @patch("qcluster.algorithms.feature_extractors.create_embeddings")
    def test_n_larger_than_number_of_documents(self, mock_create_embedding):
        """
        Tests the case where n is larger than the number of available documents.
        """
        # Arrange
        query = "test query"
        documents = ["doc1", "doc2"]
        n = 5

        def create_embedding_side_effect(input_data):
            if isinstance(input_data, str):
                return torch.randn(1, 128)
            elif isinstance(input_data, list):
                m = torch.randn(len(input_data), 128)
                self.assertEqual(m.shape[0], len(input_data))
                return m
            return None

        mock_create_embedding.side_effect = create_embedding_side_effect

        # Act
        result = get_top_n_similar_embeddings(query, documents, n)

        # Assert
        self.assertEqual(len(result), len(documents))

    @patch("qcluster.algorithms.feature_extractors.create_embeddings")
    def test_empty_document_list(self):
        """
        Tests the case where the input document list is empty.
        """
        # Arrange
        query = "test query"
        documents = []
        n = 3
        # Act
        with self.assertRaises(ValueError):
            get_top_n_similar_embeddings(query, documents, n)


if __name__ == "__main__":
    unittest.main()
