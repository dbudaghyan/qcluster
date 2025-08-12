# The QCluster Pipeline

The QCluster pipeline is a series of steps that take a dataset of customer questions and group them into meaningful clusters. This document provides a high-level overview of the pipeline's stages and explains how you can customize its behavior.

## Pipeline Stages

The main pipeline is defined in `qcluster/pipeline.py` and consists of the following stages:

1.  **Loading Samples:** The pipeline starts by loading the customer questions from a CSV file into a `SampleCollection` object. The path to the CSV file is hardcoded in `qcluster/pipeline.py`.

2.  **Processing Samples:** The loaded samples are grouped by their original categories. For each category, the pipeline generates embeddings for the questions and uses a large language model (LLM) to create a description of the category. This step is used to create a baseline for the evaluation.

3.  **Creating and Clustering Instructions:** To avoid data leakage during the clustering process, the pipeline creates an `InstructionCollection` from the `SampleCollection`. This collection contains only the raw questions, without any of the original category information. The pipeline then generates embeddings for these instructions and applies a clustering algorithm to group them into clusters.

4.  **Describing Clusters:** Once the instructions have been clustered, the pipeline uses an LLM to generate a descriptive title and a summary for each cluster. This helps to understand the common theme of the questions within each cluster.

5.  **Matching Clusters to Categories:** To evaluate the quality of the clustering, the pipeline matches each generated cluster to the original categories from the dataset. It does this by comparing the embeddings of the instruction clusters with the embeddings of the sample categories and finding the best match.

6.  **Evaluation and Reporting:** Finally, the pipeline evaluates the clustering results by comparing the predicted cluster for each question with its original category. It generates a confusion matrix, calculates various statistics (e.g., precision, recall, F1-score), and saves all the results to a unique directory in the `EVALUATION_RESULTS_DIR`. It also uses an LLM to generate a qualitative report on the results.

## Customizing the Pipeline

The QCluster pipeline is designed to be customizable. You can easily swap out the core components of the pipeline by modifying the `qcluster/pipeline.py` file. The main components are:

-   `feature_extractor`: This function is responsible for generating embeddings for the questions. By default, it uses a sentence transformer model and UMAP for dimensionality reduction. You can replace this with any function that takes a list of texts and returns a `torch.Tensor` of embeddings.

-   `clustering_function`: This function performs the clustering. The default is `kmeans_clustering`. You can replace it with other algorithms from `qcluster/algorithms/clustering.py` or implement your own. The function should take a `torch.Tensor` of embeddings and return a list of cluster labels.

-   `describer`: This function generates the titles and descriptions for the clusters. It uses an LLM and a prompt template. You can customize the prompt by changing the `DESCRIPTION_PROMPT_TEMPLATE` environment variable or by providing a different function altogether.

-   `similarity_function`: This function is used to match the instruction clusters to the sample categories. The default implementation finds the top N most similar embeddings. You can customize its behavior or replace it with a different similarity metric.

All the signatures for these functions are defined in `qcluster/custom_types.py`. By adhering to these signatures, you can easily extend and experiment with the QCluster pipeline.
