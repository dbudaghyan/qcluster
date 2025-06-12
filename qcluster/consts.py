from typing import Callable, Any

EmbeddingType = Any
ClusterType = Any
EmbeddingFunctionType = Callable[[list[str]], list[EmbeddingType]]
ClusteringFunctionType = Callable[[list[EmbeddingType]], list[ClusterType]]
