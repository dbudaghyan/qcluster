from typing import Callable, Any

EmbeddingType = Any
ClusterType = Any
EmbeddingFunctionType = Callable[[list[str]], list[EmbeddingType]]
ClusteringFunctionType = Callable[[list[EmbeddingType]], list[ClusterType]]

# Dissimilarity function takes a list of strings and an integer n, as input,
# returns a list of tuples containing the index and the corresponding string.
DissimilarityFunctionType = Callable[[list[str], int], list[tuple[int, str]]]
