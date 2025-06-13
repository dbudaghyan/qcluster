from typing import Union

from sklearn.metrics.pairwise import cosine_similarity

from qcluster.consts import EmbeddingType
from qcluster.feature_extractors import create_embeddings
from qcluster.models import MODEL


def get_top_n_similar_embeddings(
    query_embedding: Union[str, EmbeddingType],
    embeddings: list[Union[str, EmbeddingType]],
    top_n: int = 5,
) -> list[tuple[int, float]]:
    """
    Get the top N most similar embeddings to a query embedding.

    Args:
        query_embedding (EmbeddingType): The embedding to compare against.
        embeddings (list[EmbeddingType]): List of embeddings to search.
        top_n (int): Number of top similar embeddings to return.

    Returns:
        list[tuple[int, EmbeddingType]]: List of tuples containing the index and
                                          the corresponding embedding.
    """
    if isinstance(query_embedding, str):
        query_embedding = create_embeddings(
            [query_embedding], model=MODEL
        )[0]
    if not embeddings:
        raise ValueError("embeddings list cannot be empty.")
    if isinstance(embeddings[0], str):
        embeddings = create_embeddings(embeddings, model=MODEL)

    similarities = [
        (i, float(cosine_similarity([query_embedding], [embedding])[0][0]))
        for i, embedding in enumerate(embeddings)
    ]
    similarities.sort(key=lambda x: x[1], reverse=True)
    return similarities[:top_n]
