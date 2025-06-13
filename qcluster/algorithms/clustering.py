import os
from typing import Optional

import numpy as np
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from sklearn.cluster import (
    KMeans,
    DBSCAN,
    AgglomerativeClustering
)

from hdbscan import HDBSCAN
from sklearn.feature_extraction.text import CountVectorizer

from qcluster.custom_types import EmbeddingType


def kmeans_clustering(embeddings: EmbeddingType, n_clusters) -> list[int]:
    """
    Generates clusters using the K-Means algorithm.
    Args:
        embeddings: A list of embeddings to cluster.
        n_clusters: The number of clusters to find.
    Returns:
        A list of cluster labels for each embedding.
    """
    embeddings_array = np.array(embeddings)
    kmeans = KMeans(n_clusters=n_clusters,
                    random_state=42,
                    n_init='auto',
                    )
    kmeans.fit(embeddings_array)
    return kmeans.labels_.tolist()


def dbscan_clustering(embeddings: EmbeddingType, eps=0.5) -> list[int]:
    """
    Generates clusters using the DBSCAN algorithm.

    Args:
        embeddings: A list of embeddings to cluster.
        eps: The maximum distance between two samples for one to be considered
             as in the neighborhood of the other.
    Returns:
        A list of cluster labels for each embedding.
    """
    embeddings_array = np.array(embeddings)
    dbscan = DBSCAN(eps=eps, min_samples=5, metric='euclidean')
    dbscan.fit(embeddings_array)
    return dbscan.labels_.tolist()

def hdbscan_clustering(embeddings: EmbeddingType,
                       min_cluster_size=5,
                       min_samples=5,
                       cluster_selection_epsilon=0.5,
                       max_cluster_size=0,
                       metric="euclidean",
                       alpha=1.0,
                       p=None,
                       algorithm="best",
                       leaf_size=40,
                       ) -> list[int]:
    """
    Generates clusters using the HDBSCAN algorithm.
    Args:
        embeddings: A list of embeddings to cluster.
        min_cluster_size: The minimum size of clusters.
        min_samples: The number of samples in a neighborhood for a point to be
                     considered as a core point.
        cluster_selection_epsilon: The epsilon value for cluster selection.
        max_cluster_size: The maximum size of clusters.
        metric: The distance metric to use.
        alpha: The alpha parameter for the HDBSCAN algorithm.
        p: The p parameter for the Minkowski distance.
        algorithm: The algorithm to use for clustering.
        leaf_size: The leaf size for the tree structure used in clustering.
    Returns:
        A list of cluster labels for each embedding.
    """
    embeddings_array = np.array(embeddings)
    hdbscan = HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        cluster_selection_epsilon=cluster_selection_epsilon,
        max_cluster_size=max_cluster_size,
        metric=metric,
        alpha=alpha,
        p=p,
        algorithm=algorithm,
        leaf_size=leaf_size
    )
    hdbscan.fit(embeddings_array)
    return hdbscan.labels_.tolist()


def agglomerative_clustering(embeddings: EmbeddingType, n_clusters=8) -> list[int]:
    """
    Generates clusters using the Agglomerative Clustering algorithm.
    Args:
        embeddings: A list of embeddings to cluster.
        n_clusters: The number of clusters to find.
    Returns:
        A list of cluster labels for each embedding.
    """
    embeddings_array = np.array(embeddings)
    agglomerative = AgglomerativeClustering(n_clusters=n_clusters)
    agglomerative.fit(embeddings_array)
    return agglomerative.labels_.tolist()


def bert_topic_extraction(
    embeddings: EmbeddingType,
    n_topics: Optional[int],
    model: SentenceTransformer = None,
) -> list[int]:
    """"""
    if len(embeddings) == 0:
        raise ValueError("embeddings list cannot be empty.")
    if isinstance(embeddings[0], str):
        if not model:
            model = SentenceTransformer(os.environ['SENTENCE_TRANSFORMERS_MODEL'])
        topic_model = BERTopic(embedding_model=model,
                               nr_topics=n_topics,
                               verbose=True)
        topics, _ = topic_model.fit_transform(documents=embeddings)
    else:
        embeddings = np.array(embeddings)
        dummy_documents = [f"embedding_{i}" for i in range(len(embeddings))]
        dummy_vectorizer = CountVectorizer(tokenizer=lambda x: [x])
        topic_model = BERTopic(vectorizer_model=dummy_vectorizer,
                               embedding_model=None,
                               nr_topics=n_topics,
                               verbose=True)
        topics, _ = topic_model.fit_transform(documents=dummy_documents,
                                              embeddings=embeddings)
    return topics
