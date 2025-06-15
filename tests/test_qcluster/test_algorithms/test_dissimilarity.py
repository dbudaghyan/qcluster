import unittest

import numpy as np

from qcluster.algorithms.dissimilarity import select_mmr


class TestSelectMMR(unittest.TestCase):
    def setUp(self):
        """Set up test data for the tests."""
        self.sentences = [
            "This is the first sentence.",
            "This is the second sentence, quite similar to the first.",
            "A completely different sentence.",
            "Another sentence, also very different.",
            "A final sentence, somewhat related to the third.",
        ]
        # Using a simple, predictable set of embeddings for testing logic
        self.test_embeddings = np.array(
            [
                [1.0, 0.0, 0.0, 0.0],  # 0: A
                [0.9, 0.1, 0.0, 0.0],  # 1: B (highly similar to A)
                [0.0, 1.0, 0.0, 0.0],  # 2: C (different from A/B)
                [0.0, 0.0, 1.0, 0.0],  # 3: D (different from all)
            ]
        )
        # The function returns the original items from the list,
        # so we need a list of "sentences"
        self.embedding_labels = [
            "Embedding A",
            "Embedding B",
            "Embedding C",
            "Embedding D",
        ]

    def test_select_mmr_with_strings(self):
        """Test MMR selection with a list of sentence strings."""
        selected = select_mmr(self.sentences, n=3)
        self.assertEqual(len(selected), 3)
        self.assertIsInstance(selected[0], tuple, "Each item should be a tuple")
        self.assertIsInstance(
            selected[0][0], int, "The first element of the tuple should be an index"
        )
        self.assertIsInstance(
            selected[0][1], str, "The second element of the tuple should be a sentence"
        )

    def test_select_mmr_with_embeddings(self):
        """Test MMR selection with a list of pre-computed embeddings."""
        selected = select_mmr(self.test_embeddings.tolist(), n=2)
        self.assertEqual(len(selected), 2)
        self.assertIsInstance(selected[0], tuple, "Each item should be a tuple")
        self.assertIsInstance(
            selected[0][0], int, "The first element of the tuple should be an index"
        )
        self.assertIsInstance(
            selected[0][1],
            list,
            "The second element of the tuple should be an embedding",
        )

    def test_n_greater_than_length(self):
        """Test selecting more items than available in the list."""
        selected = select_mmr(self.sentences, n=10)
        self.assertEqual(
            len(selected),
            len(self.sentences),
            "Should return all sentences if n is larger than list size",
        )

    def test_n_zero(self):
        """Test selecting zero items."""
        selected = select_mmr(self.sentences, n=0)
        self.assertEqual(len(selected), 0, "Should return an empty list for n=0")

    def test_invalid_lambda(self):
        """Test that an out-of-bounds lambda_param raises a ValueError."""
        with self.assertRaises(ValueError):
            select_mmr(self.sentences, n=3, lambda_param=-0.1)
        with self.assertRaises(ValueError):
            select_mmr(self.sentences, n=3, lambda_param=1.1)

    def test_lambda_one_prioritizes_relevance(self):
        """
        Test with lambda=1, which should select the most relevant items,
        ignoring diversity.
        """
        # With lambda=1, the MMR score is just the relevance score.
        # The algorithm will pick the top N most relevant items.
        # In our test_embeddings, B (1) and A (0) are most similar to the cluster center.
        selected = select_mmr(self.test_embeddings.tolist(), n=2, lambda_param=1.0)
        selected_indices = sorted([item[0] for item in selected])
        self.assertEqual(
            selected_indices, [0, 1], "Should select the two most similar items (A, B)"
        )

    def test_lambda_zero_prioritizes_diversity(self):
        """
        Test with lambda=0, which should select the most diverse items,
        ignoring relevance (after the first pick).
        """
        # With lambda=0, the MMR score is based purely on dissimilarity from
        # already selected items.
        # 1. The first item is the most relevant (B, index 1).
        # 2. The next items should be those least similar to B. D and C are
        #    the most dissimilar.
        selected = select_mmr(self.test_embeddings.tolist(), n=3, lambda_param=0.0)
        selected_indices = sorted([item[0] for item in selected])
        self.assertEqual(
            selected_indices,
            [1, 2, 3],
            "Should select B, then the most dissimilar items C and D",
        )
