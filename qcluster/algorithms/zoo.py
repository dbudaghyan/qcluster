from qcluster.algorithms.clustering import (
    kmeans_clustering,
    dbscan_clustering,
    hdbscan_clustering,
    agglomerative_clustering,
    bert_topic_extraction
)


class ClusteringZoom:
    kmeans = kmeans_clustering
    dbscan = dbscan_clustering
    hdbscan = hdbscan_clustering
    agglomerative = agglomerative_clustering
    bert_topic_extraction = bert_topic_extraction
