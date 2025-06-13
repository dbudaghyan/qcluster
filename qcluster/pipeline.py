from dotenv import load_dotenv
# Make sure to load environment variables first,
# as some imports may initialize models using them.
load_dotenv()

import functools
from os import PathLike

import torch
from loguru import logger
from tqdm import tqdm

from qcluster.custom_types import CategoryType, IdToCategoryResultType
from qcluster import ROOT_DIR
from qcluster.datamodels.instruction import InstructionCollection
from qcluster.datamodels.sample import SampleCollection
from qcluster.describer import get_description

# Clustering algorithms
# noinspection PyUnresolvedReferences
from qcluster.clustering import (
    kmeans_clustering,
    dbscan_clustering,
    hdbscan_clustering,
    agglomerative_clustering
)
# Feature extractors
# noinspection PyUnresolvedReferences
from qcluster.feature_extractors import (
    create_embeddings,
    pca_reduction,
    umap_reduction,
)


from qcluster.similarity import (
    get_top_n_similar_embeddings
)
from qcluster.evaluation import evaluate_results
from qcluster.models import MODEL


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

CSV_PATH: PathLike = (
        ROOT_DIR.parent
        / "data"
        / "Bitext_Sample_Customer_Support_Training_Dataset_27K_responses-v11.csv"
)


if __name__ == '__main__':
    import time
    start_time = time.time()
    logger.info(f"Loading samples from {CSV_PATH}...")
    samples: SampleCollection = SampleCollection.from_csv(CSV_PATH)
    logger.info(f"Loaded {len(samples)} samples.")
    # samples: SampleCollection = samples[:1000]
    logger.info(f"Using {len(samples)} samples for processing.")

    logger.info("Grouping samples by category and updating embeddings...")
    samples_by_category: dict[CategoryType,
        SampleCollection] = samples.group_by_category()
    logger.info(f"Grouped samples into {len(samples_by_category)} categories.")

    logger.info("Describing samples in each category...")
    for predicted_category, sample_collection in tqdm(samples_by_category.items()):
        sample_collection.update_embeddings(feature_extractor)
        sample_collection.describe(get_description)
    logger.info("Embeddings updated and samples described.")

    logger.info("Creating instruction collection from samples...")
    instructions: InstructionCollection = InstructionCollection.from_samples(samples)
    logger.info(f"Created an instruction collection with"
                f" {len(instructions)} instructions.")

    logger.info("Updating instruction embeddings and clustering...")
    (instructions
        .update_embeddings(feature_extractor)  # inplace operation
        .update_clusters(clustering_function)  # inplace operation
     )
    logger.info("Instruction embeddings updated and clusters created.")

    logger.info("Grouping instructions by cluster...")
    instructions_by_cluster: dict[int, InstructionCollection] = (
        instructions.group_by_cluster()
    )
    logger.info(f"Grouped instructions into {len(instructions_by_cluster)} clusters.")

    logger.info("Describing instructions in each cluster...")
    for cluster, instruction_collection in tqdm(instructions_by_cluster.items()):
        instruction_collection.describe(get_description)
    logger.info("Instructions described.")

    logger.info("Finding top similar sample categories for each instruction cluster...")
    id_to_category_pairs: IdToCategoryResultType = {}
    for cluster, instruction_collection in tqdm(instructions_by_cluster.items()):
        predicted_category: CategoryType = instruction_collection.get_cluster_category(
            sample_collections=list(samples_by_category.values()),
            similarity_function=get_top_n_similar_embeddings,
        )
        logger.info(f"Cluster N {instruction_collection.cluster}"
              f" title: `{instruction_collection.title}`"
              f" top similar sample category: {predicted_category}")
        logger.info(f"Mapping cluster {cluster} to category {predicted_category}")
        for sample in instruction_collection:
            id_to_category_pairs[sample.id] = (
                samples.get_sample_by_id(sample.id).category,
                predicted_category
            )
    logger.info("Matching completed.")
    logger.info(f"Total pairs: {len(id_to_category_pairs)}")
    logger.info("Evaluating results...")
    cm = evaluate_results(id_to_category_pairs)
    logger.info("Evaluation results:")
    cm.print_matrix(sparse=True )
    cm.stat(summary=True)
    cm.save_obj(
        ROOT_DIR / "evaluation_results" / "confusion_matrix.json"
    )
    logger.info(f"Execution time: {time.time() - start_time:.2f} seconds")
