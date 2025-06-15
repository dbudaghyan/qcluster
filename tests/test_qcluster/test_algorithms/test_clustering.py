from functools import partial
from typing import Callable

import numpy as np
import pytest

from qcluster.algorithms.clustering import (agglomerative_clustering,
                                            bert_topic_extraction,
                                            dbscan_clustering,
                                            hdbscan_clustering,
                                            kmeans_clustering,
                                            spectral_clustering)

# --- Test Data ---

# Sample embeddings with 3 distinct clusters
SAMPLE_EMBEDDINGS = np.array(
    [
        [1.0, 1.0],
        [1.1, 1.2],
        [0.9, 0.8],
        [1.0, 1.1],
        [1.2, 1.0],
        [0.8, 0.9],
        [1.1, 1.0],
        [1.0, 0.9],
        [0.9, 1.1],
        [1.2, 1.2],
        [10.0, 10.0],
        [10.1, 10.2],
        [9.9, 9.8],
        [10.0, 10.1],
        [10.2, 10.0],
        [9.8, 9.9],
        [10.1, 10.0],
        [10.0, 9.9],
        [9.9, 10.1],
        [10.2, 10.2],
        [20.0, 20.0],
        [20.1, 20.2],
        [19.9, 19.8],
        [20.0, 20.1],
        [20.2, 20.0],
        [19.8, 19.9],
        [20.1, 20.0],
        [20.0, 19.9],
        [19.9, 20.1],
        [20.2, 20.2],
    ]
).tolist()

# Densely packed data for DBSCAN/HDBSCAN tests
DENSE_EMBEDDINGS = np.array(
    [
        [1.0, 1.0],
        [1.1, 1.2],
        [0.9, 0.8],
        [1.2, 0.9],
        [1.3, 1.1],
        [1.0, 1.1],
        [1.2, 1.0],
        [0.8, 0.9],
        [1.1, 1.0],
        [1.0, 0.9],
        [10.0, 10.0],  # This point should be noise
    ]
).tolist()

EMPTY_EMBEDDINGS = []
SINGLE_EMBEDDING = [[5.0, 5.0]]


# --- Conformance Test Helper ---


def check_clustering_function_conformance(clustering_fn: Callable, embeddings: list):
    """
    Checks if a clustering function conforms to the ClusteringFunctionType
    by verifying its input/output types and dimensions.
    """
    assert callable(clustering_fn)

    # The function should handle empty lists gracefully or raise a specific error
    if not embeddings:
        # bert_topic_extraction has a specific check for empty lists
        if "bert_topic" in str(clustering_fn):
            with pytest.raises(ValueError):
                clustering_fn(embeddings)
            return
        # Other functions might raise various errors; we just ensure they don't hang
        with pytest.raises(Exception):
            clustering_fn(embeddings)
        return

    labels = clustering_fn(embeddings)

    assert isinstance(labels, list)
    assert len(labels) == len(embeddings)
    if labels:
        assert all(isinstance(label, int) for label in labels)


# --- Test Cases ---


def test_kmeans_clustering():
    """Tests the kmeans_clustering function."""
    labels = kmeans_clustering(SAMPLE_EMBEDDINGS, n_clusters=3)
    assert len(set(labels)) == 3
    assert len(labels) == len(SAMPLE_EMBEDDINGS)

    labels_single = kmeans_clustering(SINGLE_EMBEDDING, n_clusters=1)
    assert labels_single == [0]

    partial_fn = partial(kmeans_clustering, n_clusters=3)
    check_clustering_function_conformance(partial_fn, SAMPLE_EMBEDDINGS)
    check_clustering_function_conformance(partial_fn, EMPTY_EMBEDDINGS)


def test_spectral_clustering():
    """Tests the spectral_clustering function."""
    labels = spectral_clustering(SAMPLE_EMBEDDINGS, n_clusters=3)
    assert len(set(labels)) == 3
    assert len(labels) == len(SAMPLE_EMBEDDINGS)

    # Spectral clustering requires at least 2 samples, so we expect a ValueError
    with pytest.raises(ValueError):
        spectral_clustering(SINGLE_EMBEDDING, n_clusters=1)

    partial_fn = partial(spectral_clustering, n_clusters=3)
    check_clustering_function_conformance(partial_fn, SAMPLE_EMBEDDINGS)
    check_clustering_function_conformance(partial_fn, EMPTY_EMBEDDINGS)


def test_agglomerative_clustering():
    """Tests the agglomerative_clustering function."""
    labels = agglomerative_clustering(SAMPLE_EMBEDDINGS, n_clusters=3)
    assert len(set(labels)) == 3
    assert len(labels) == len(SAMPLE_EMBEDDINGS)

    # Agglomerative clustering requires at least 2 samples, so we expect a ValueError
    with pytest.raises(ValueError):
        agglomerative_clustering(SINGLE_EMBEDDING, n_clusters=1)

    partial_fn = partial(agglomerative_clustering, n_clusters=3)
    check_clustering_function_conformance(partial_fn, SAMPLE_EMBEDDINGS)
    check_clustering_function_conformance(partial_fn, EMPTY_EMBEDDINGS)


def test_dbscan_clustering():
    """Tests the dbscan_clustering function."""
    labels = dbscan_clustering(DENSE_EMBEDDINGS, eps=1.0)
    # Expects one cluster (label 0) and one noise point (label -1)
    assert set(labels) == {0, -1}
    assert len(labels) == len(DENSE_EMBEDDINGS)

    labels_single = dbscan_clustering(SINGLE_EMBEDDING, eps=0.5)
    assert labels_single == [-1]  # Single point is considered noise

    partial_fn = partial(dbscan_clustering, eps=1.0)
    check_clustering_function_conformance(partial_fn, DENSE_EMBEDDINGS)
    check_clustering_function_conformance(partial_fn, EMPTY_EMBEDDINGS)


def test_hdbscan_clustering():
    """Tests the hdbscan_clustering function."""
    labels = hdbscan_clustering(DENSE_EMBEDDINGS, min_cluster_size=3, min_samples=1)
    # Expects one cluster and one noise point
    assert 0 in labels and -1 in labels
    assert len(labels) == len(DENSE_EMBEDDINGS)
    partial_fn = partial(hdbscan_clustering, min_cluster_size=5)
    check_clustering_function_conformance(partial_fn, DENSE_EMBEDDINGS)
    check_clustering_function_conformance(partial_fn, EMPTY_EMBEDDINGS)


def test_bert_topic_extraction():
    """Tests the bert_topic_extraction function with pre-computed embeddings."""
    # Test with pre-computed embeddings
    labels = bert_topic_extraction(SAMPLE_EMBEDDINGS, n_clusters=3)
    assert len(labels) == len(SAMPLE_EMBEDDINGS)
    # BERTopic may produce fewer clusters or an outlier topic (-1)
    assert len(set(labels)) <= 4

    # Test edge case: single embedding
    with pytest.raises(ValueError):
        bert_topic_extraction(SINGLE_EMBEDDING, n_clusters=1)

    # Test edge case: an empty list raises ValueError
    with pytest.raises(ValueError, match="embeddings list cannot be empty"):
        bert_topic_extraction(EMPTY_EMBEDDINGS, n_clusters=1)

    # Conformance Check
    partial_fn = partial(bert_topic_extraction, n_clusters=3)
    check_clustering_function_conformance(partial_fn, SAMPLE_EMBEDDINGS)
    check_clustering_function_conformance(partial_fn, EMPTY_EMBEDDINGS)
