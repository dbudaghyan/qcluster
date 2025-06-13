# Preloading env vars, seeds and models
from qcluster import preload  # noqa: F401
from qcluster.preload import MODEL
import functools
import time
from os import PathLike

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
)
from qcluster.custom_types import CategoryType, IdToCategoryResultType
from qcluster.datamodels.instruction import InstructionCollection
from qcluster.datamodels.sample import SampleCollection
from qcluster.algorithms.describer import get_description
from qcluster.evaluation import evaluate_results
# Feature extractors
from qcluster.algorithms.feature_extractors import (
    create_embeddings,
    pca_reduction,
    # umap_reduction,
)
from qcluster.algorithms.similarity import (
    get_top_n_similar_embeddings
)




def feature_extractor(texts: list[str]) -> torch.Tensor:
    """
    Creates embeddings for the given texts and reduces their dimensionality using PCA.
    """
    embeddings = create_embeddings(texts, model=MODEL)
    embeddings = pca_reduction(embeddings, n_components=20)
    return embeddings


clustering_function = functools.partial(
    kmeans_clustering,
    n_clusters=len(SampleCollection.all_category_classes()))
# clustering_function = functools.partial(
#     bert_topic_extraction,
#     n_topics=len(SampleCollection.all_category_classes()),
# )


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
        sample_collection.describe(get_description)
    logger.info("Embeddings updated and samples described.")
    return samples_by_category


def create_instructions(samples: SampleCollection) -> InstructionCollection:
    """Creates and processes an InstructionCollection from a SampleCollection."""
    logger.info("Creating instruction collection from samples...")
    instructions = InstructionCollection.from_samples(samples)
    logger.info(f"Created an instruction collection with"
                f" {len(instructions)} instructions.")

    logger.info("Updating instruction embeddings and clustering...")
    (instructions
     .update_embeddings(feature_extractor)
     .update_clusters(clustering_function=clustering_function,
                      use_raw_instructions=False)
     )
    logger.info("Instruction embeddings updated and clusters created.")
    return instructions


def create_and_match_clusters(
        instructions: InstructionCollection,
        samples_by_category: dict[CategoryType, SampleCollection],
        all_samples: SampleCollection
) -> IdToCategoryResultType:
    """Describes instructions and matches them to sample categories."""
    logger.info("Grouping instructions by cluster...")
    instructions_by_cluster = instructions.group_by_cluster()
    logger.info(f"Grouped instructions into {len(instructions_by_cluster)} clusters.")

    logger.info("Describing instructions in each cluster...")
    for cluster, instruction_collection in tqdm(instructions_by_cluster.items()):
        instruction_collection.describe(get_description)
    logger.info("Instructions described.")

    logger.info("Finding top similar sample categories for each instruction cluster...")
    id_to_category_pairs: IdToCategoryResultType = {}
    for cluster, instruction_collection in tqdm(instructions_by_cluster.items()):
        predicted_category = instruction_collection.get_cluster_category(
            sample_collections=list(samples_by_category.values()),
            similarity_function=get_top_n_similar_embeddings,
        )
        logger.info(f"Cluster N {instruction_collection.cluster} title:"
                    f" `{instruction_collection.title}`"
                    f" top similar sample category: {predicted_category}")
        logger.info(f"Mapping cluster {cluster} to category {predicted_category}")
        for sample in instruction_collection:
            id_to_category_pairs[sample.id] = (
                all_samples.get_sample_by_id(sample.id).category,
                predicted_category
            )
    logger.info("Matching completed.")
    logger.info(f"Total pairs: {len(id_to_category_pairs)}")
    return id_to_category_pairs


def main():
    """
    Main function to run the clustering pipeline.
    """
    samples = load_samples(CSV_PATH)
    # samples: SampleCollection = samples[:1000]
    logger.info(f"Using {len(samples)} samples for processing.")

    samples_by_category = process_samples(samples)
    instructions = create_instructions(samples)
    id_to_category_pairs = create_and_match_clusters(
        instructions, samples_by_category, samples
    )

    logger.info("Evaluating results...")
    cm = evaluate_results(id_to_category_pairs)
    logger.info("Evaluation results:")
    cm.print_matrix(sparse=True)
    cm.stat(summary=True)


if __name__ == '__main__':
    start_time = time.time()
    main()
    logger.info(f"Execution time: {time.time() - start_time:.2f} seconds")
