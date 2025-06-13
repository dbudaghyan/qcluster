from typing import Optional, Union

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from qcluster.consts import EmbeddingType


def select_mmr(sentences: list[Union[str, EmbeddingType]], n: int,
               lambda_param: Optional[float] = 0.5) -> list[
    tuple[int, str]]:
    """
    Selects a diverse subset of N sentences from a list using
     Maximal Marginal Relevance (MMR).

    Args:
        sentences (list[str]): A list of sentence strings or embeddings.
        n (int): The number of sentences to select.
        lambda_param (float): The hyperparameter lambda for MMR, balancing relevance
                              and diversity. 0 <= lambda_param <= 1.
                              A value of 1 means only relevance is considered.
                              A value of 0 means only diversity is considered.

    Returns:
        list[tuple[str, int]]: A list of tuples, where each tuple contains the
                               selected sentence and its original index in the
                               input list.

    Raises:
        ValueError: If n is out of bounds, or lambda_param is not between 0 and 1.
    """
    if not (0 <= lambda_param <= 1):
        raise ValueError("lambda_param must be between 0 and 1.")
    if n <= 0:
        return []
    if n > len(sentences):
        raise ValueError(
            f"N ({n}) cannot be greater than the number of sentences ({len(sentences)}).")

    # 1. Generate Embeddings
    # Using a popular and effective pre-trained model.
    # The model will be downloaded automatically on first use.
    if isinstance(sentences[0], str):
        from qcluster.models import MODEL
        embeddings = MODEL.encode(sentences)
    else:
        embeddings = np.array(sentences)

    # 2. Calculate Similarity Matrix
    similarity_matrix = cosine_similarity(embeddings)

    # 3. Calculate Relevance (Query-Sentence Similarity)
    # Here, we define relevance as the sentence's average similarity to all others.
    # This represents how "central" or "representative" a sentence is.
    relevance_scores = similarity_matrix.mean(axis=1)

    # 4. Iteratively select sentences using MMR
    selected_indices = []

    # Start with the most relevant sentence
    # This avoids a random start and makes the process deterministic
    current_best_idx = np.argmax(relevance_scores)
    selected_indices.append(current_best_idx)

    # Keep track of unselected indices
    unselected_indices = [i for i in range(len(sentences)) if i not in selected_indices]

    for _ in range(n - 1):
        mmr_scores = {}

        for i in unselected_indices:
            relevance = relevance_scores[i]
            # Calculate the maximum similarity to any already selected sentence
            # Note: similarity_matrix[i, selected_indices] gives a row slice
            max_similarity_to_selected = np.max(similarity_matrix[i, selected_indices])
            # Calculate MMR score
            mmr_score = lambda_param * relevance - (
                        1 - lambda_param) * max_similarity_to_selected
            mmr_scores[i] = mmr_score
        # Select the sentence with the highest MMR score
        best_next_idx = max(mmr_scores, key=mmr_scores.get)
        # Add to `selected` and remove from unselected
        selected_indices.append(best_next_idx)
        unselected_indices.remove(best_next_idx)
    # 5. Return the selected sentences and their original indices
    return [(i, sentences[i]) for i in selected_indices]
