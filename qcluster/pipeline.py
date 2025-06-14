import functools
import os
import time
from os import PathLike
from pathlib import Path

import torch
from loguru import logger
from tqdm import tqdm

from qcluster import ROOT_DIR

# Clustering algorithms
from qcluster.algorithms.clustering import (
    kmeans_clustering,
    # dbscan_clustering,
    # hdbscan_clustering,
    # agglomerative_clustering
    # bert_topic_extraction
    # spectral_clustering
)
from qcluster.llm.describer import get_description

# Feature extractors
from qcluster.algorithms.feature_extractors import (
    create_embeddings,
    # pca_reduction,
    umap_reduction,
    # pacmap_reduction
)
from qcluster.algorithms.similarity import get_top_n_similar_embeddings
from qcluster.custom_types import (
    CategoryType,
    IdToCategoryResultType,
    category_to_idx,
    ClusterType,
)
from qcluster.datamodels.instruction import InstructionCollection
from qcluster.datamodels.sample import SampleCollection
from qcluster.evaluation import (
    evaluate_results,
    cluster_to_class_similarity_measures,
    store_results,
)
from qcluster.preload import MODEL

N_CATEGORIES = len(SampleCollection.all_category_classes())

clustering_function = functools.partial(
    # hdbscan_clustering,
    kmeans_clustering,
    # min_cluster_size=1000,
    n_clusters=N_CATEGORIES,
)


describer = functools.partial(
    get_description,
    template_name=os.environ['DESCRIPTION_PROMPT_TEMPLATE'],
    # template_name='description_prompt_from_instructions'
)

similarity_function = functools.partial(
    get_top_n_similar_embeddings,
    use_mmr=False,
    # mmr_lambda=0.3,
    # mmr_top_n=20
)


def feature_extractor(texts: list[str]) -> torch.Tensor:
    """
    Creates embeddings for the given texts and reduces their dimensionality using PCA.
    """
    embeddings = create_embeddings(texts, model=MODEL)
    embeddings = umap_reduction(embeddings, n_components=28)
    return embeddings


CSV_PATH: PathLike = (
    ROOT_DIR.parent
    / "data"
    / "Bitext_Sample_Customer_Support_Training_Dataset_27K_responses-v11.csv"
)


def load_samples(path: PathLike) -> SampleCollection:
    """Loads samples from a CSV file."""
    logger.info(f"Loading samples from {path}...")
    samples = SampleCollection.from_csv(path)
    logger.info(f"Loaded {len(samples)} samples.")
    return samples


def process_samples(samples: SampleCollection) -> dict[CategoryType, SampleCollection]:
    """Groups samples by category and updates their embeddings and descriptions."""
    logger.info("Grouping samples by category and updating embeddings...")
    samples_by_category = samples.group_by_category()
    logger.info(f"Grouped samples into {len(samples_by_category)} categories.")

    logger.info("Describing samples in each category...")
    for category, sample_collection in tqdm(samples_by_category.items()):
        sample_collection.update_embeddings(feature_extractor)
        sample_collection.describe(describer)
    logger.info("Embeddings updated and samples described.")
    return samples_by_category


def create_instructions(samples: SampleCollection) -> InstructionCollection:
    """Creates and processes an InstructionCollection from a SampleCollection."""
    logger.info("Creating instruction collection from samples...")
    instructions = InstructionCollection.from_samples(samples)
    logger.info(
        f"Created an instruction collection with" f" {len(instructions)} instructions."
    )

    logger.info("Updating instruction embeddings and clustering...")
    (
        instructions.update_embeddings(feature_extractor).update_clusters(
            clustering_function=clustering_function, use_raw_instructions=False
        )
    )
    logger.info("Instruction embeddings updated and clusters created.")
    return instructions


def create_clusters(
    instructions: InstructionCollection,
) -> dict[ClusterType, InstructionCollection]:
    """Groups instructions into clusters and describes them."""
    logger.info("Grouping instructions by cluster...")
    instructions_by_cluster = instructions.group_by_cluster()
    logger.info(f"Grouped instructions into {len(instructions_by_cluster)} clusters.")
    logger.info("Describing instructions in each cluster...")
    for cluster, instruction_collection in tqdm(instructions_by_cluster.items()):
        instruction_collection.describe(describer)
    logger.info("Instructions described.")
    return instructions_by_cluster


def match_clusters(
    instructions_by_cluster: dict[ClusterType, InstructionCollection],
    samples_by_category: dict[CategoryType, SampleCollection],
    all_samples: SampleCollection,
) -> IdToCategoryResultType:
    """Matches instruction clusters to sample categories."""
    logger.info("Finding top similar sample categories for each instruction cluster...")
    id_to_category_pairs: IdToCategoryResultType = {}
    for cluster, instruction_collection in tqdm(instructions_by_cluster.items()):
        predicted_category = instruction_collection.get_cluster_category(
            sample_collections=list(samples_by_category.values()),
            similarity_function=similarity_function,
        )
        logger.info(
            f"Cluster N {instruction_collection.cluster} title:"
            f" `{instruction_collection.title}`"
            f" top similar sample category: {predicted_category}"
        )
        logger.info(f"Mapping cluster {cluster} to category {predicted_category}")
        for sample in instruction_collection:
            id_to_category_pairs[sample.id] = (
                all_samples.get_sample_by_id(sample.id).category,
                predicted_category,
            )
    logger.info("Matching completed.")
    logger.info(f"Total pairs: {len(id_to_category_pairs)}")
    return id_to_category_pairs


def main():
    """
    Main function to run the clustering pipeline.
    """
    samples = load_samples(CSV_PATH)
    samples: SampleCollection = samples[:4000]; logger.error("WARNING: Using only limited number of  samples for testing purposes."*10)
    logger.info(f"Using {len(samples)} samples for processing.")
    output_path = Path(os.environ["EVALUATION_RESULTS_DIR"])
    samples_by_category = process_samples(samples)
    instructions = create_instructions(samples)
    instructions_by_cluster = create_clusters(instructions)
    id_to_category_pairs = match_clusters(
        instructions_by_cluster, samples_by_category, samples
    )
    logger.info("Evaluating results...")
    cm = evaluate_results(id_to_category_pairs)
    predicted_clusters_dict: dict[int, int] = {i.id: i.cluster for i in instructions}
    actual_categories_dict: dict[int, int] = {
        cat.id: category_to_idx(cat.category) for cat in samples
    }
    assert None not in predicted_clusters_dict.values()
    assert None not in actual_categories_dict.values()
    predicted_cluster_list = []
    actual_category_list = []
    for id_, (actual_category, predicted_category) in id_to_category_pairs.items():
        predicted_cluster_list.append(predicted_category)
        actual_category_list.append(actual_category)
    cluster_to_class_scores = cluster_to_class_similarity_measures(
        predicted_cluster_list, actual_category_list
    )
    logger.info("Evaluation results:")
    for measure, score in cluster_to_class_scores.items():
        print(f"{measure.capitalize()}: {score:.4f}")
    cm.print_matrix(sparse=True)
    cm.stat(summary=True)
    # create a unique storage path based on the current timestamp and git commit
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    try:
        git_commit = os.popen("git rev-parse --short HEAD").read().strip()
    except Exception as e:
        logger.warning(f"Failed to get git commit: {e}")
        git_commit = "unknown"
    unique_folder_name = f"{timestamp}-{git_commit}"
    unique_folder_path = output_path / unique_folder_name
    store_results(
        cm=cm,
        cluster_to_class_scores=cluster_to_class_scores,
        storage_path=unique_folder_path,
        instructions_by_cluster=instructions_by_cluster,
    )


if __name__ == "__main__":
    start_time = time.time()
    main()
    logger.info(f"Execution time: {time.time() - start_time:.2f} seconds")
