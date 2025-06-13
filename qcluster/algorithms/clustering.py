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

from qcluster.algorithms.feature_extractors import create_embeddings
from qcluster.custom_types import EmbeddingType


def kmeans_clustering(embeddings: EmbeddingType, n_clusters=8):
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


def dbscan_clustering(embeddings, eps=0.5, min_samples=5):
    """
    Generates clusters using the DBSCAN algorithm.

    Args:
        embeddings: A list of embeddings to cluster.
        eps: The maximum distance between two samples for one to be considered
             as in the neighborhood of the other.
        min_samples: The number of samples in a neighborhood for a point to be
                     considered as a core point.
    Returns:
        A list of cluster labels for each embedding.
    """
    embeddings_array = np.array(embeddings)
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    dbscan.fit(embeddings_array)
    return dbscan.labels_.tolist()

def hdbscan_clustering(embeddings,
                       min_cluster_size=5,
                       min_samples=5,
                       cluster_selection_epsilon=0.5,
                       max_cluster_size=0,
                       metric="euclidean",
                       alpha=1.0,
                       p=None,
                       algorithm="best",
                       leaf_size=40,
                       ):
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


def agglomerative_clustering(embeddings, n_clusters=8):
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
    texts: list[str],
    model: SentenceTransformer = None,
    n_topics: Optional[int] = 5,
) -> list[tuple[int, str]]:
    if not model:
        model = SentenceTransformer(os.environ['SENTENCE_TRANSFORMERS_MODEL'])
    topic_model = BERTopic(
        embedding_model=model,
        nr_topics=n_topics,
        verbose=True,
    )
    topics, _ = topic_model.fit_transform(texts)
    texts_with_topics = [
        (i, topic_model.get_topic(topic)) for i, topic in enumerate(topics)
    ]
    raise texts_with_topics
