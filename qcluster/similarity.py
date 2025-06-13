from typing import Union

from sklearn.metrics.pairwise import cosine_similarity

from qcluster.dissimilarity import select_mmr
from qcluster.custom_types import EmbeddingType
from qcluster.feature_extractors import create_embeddings
from qcluster.models import MODEL
from qcluster.utils import calculate_centroid_embedding


def get_top_n_similar_embeddings(
    query_embedding: Union[str, EmbeddingType],
    embeddings: list[Union[str, EmbeddingType]],
    top_n: int = 5,
    use_mmr: bool = False,
    mmr_lambda: float = 0.1,
    mmr_top_n: int = 10
) -> list[tuple[int, float]]:
    """
    Get the top N most similar embeddings to a query embedding.

    Args:
        query_embedding (Union[str, EmbeddingType]): The query embedding or a string
                                                     to be converted into an embedding.
        embeddings (list[Union[str, EmbeddingType]]): List of embeddings or strings
                                                      to be converted into embeddings.
        top_n (int): The number of top similar embeddings to return.
        use_mmr (bool): Whether to use Maximal Marginal Relevance (MMR) for diversity.
        mmr_lambda (float): The lambda parameter for MMR, balancing relevance
         and diversity.
        mmr_top_n (int): The number of top candidates to consider for MMR.

    Returns:
        list[tuple[int, EmbeddingType]]: List of tuples containing the index and
                                          the corresponding embedding.
    """
    if use_mmr:
        embeddings = select_mmr(embeddings, n=mmr_top_n, lambda_param=mmr_lambda)
        embeddings = [emb[1] if isinstance(emb, tuple) else emb for emb in embeddings]
    if isinstance(query_embedding, str):
        query_embedding = create_embeddings([query_embedding], model=MODEL)[0]
    if not embeddings:
        raise ValueError("embeddings list cannot be empty.")
    if isinstance(embeddings[0], (str, tuple)):
        embeddings = create_embeddings(embeddings, model=MODEL)
    similarities = [
        (i, float(cosine_similarity([query_embedding], [embedding])[0][0]))
        for i, embedding in enumerate(embeddings)
    ]
    similarities.sort(key=lambda x: x[1], reverse=True)
    return similarities[:top_n]
