import functools
from os import PathLike

from loguru import logger

from qcluster import ROOT_DIR
from qcluster.clustering import (
    kmeans_clustering,
    # dbscan_clustering,
    # hdbscan_clustering,
    # agglomerative_clustering
)
from qcluster.consts import EmbeddingFunctionType, CategoryType, IdToCategoryResultType
from qcluster.datamodels import SampleCollection, InstructionCollection
from qcluster.describer import get_description
# from qcluster.dissimilarity import select_mmr
from qcluster.feature_extractors import create_embeddings
from dotenv import load_dotenv

from qcluster.models import MODEL
from qcluster.similarity import get_top_n_similar_embeddings
from qcluster.evaluation import evaluate_results


load_dotenv()

# Define the feature extractor, which can be any function that takes
# a list of strings and returns a collection of embeddings.
feature_extractor: EmbeddingFunctionType = functools.partial(
    create_embeddings,
    model=MODEL
)

clustering_function = functools.partial(
    kmeans_clustering,
    n_clusters=len(SampleCollection.all_category_classes()))

CSV_PATH: PathLike = (
        ROOT_DIR.parent
        / "data"
        / "Bitext_Sample_Customer_Support_Training_Dataset_27K_responses-v11.csv"
)


if __name__ == '__main__':

    # Load the samples from the CSV file.
    logger.info(f"Loading samples from {CSV_PATH}...")
    import time
    start_time = time.time()
    samples: SampleCollection = SampleCollection.from_csv(CSV_PATH)

    # Limit the number of samples for demonstration purposes.
    logger.info(f"Loaded {len(samples)} samples.")
    # samples: SampleCollection = samples[:1000]
    logger.info(f"Using {len(samples)} samples for processing.")


    # Update the embeddings for each sample using the feature extractor.
    logger.info("Grouping samples by category and updating embeddings...")
    samples_by_category: dict[CategoryType,
        SampleCollection] = samples.group_by_category()
    logger.info(f"Grouped samples into {len(samples_by_category)} categories.")
    # Describe each sample collection in the categories.
    logger.info("Describing samples in each category...")
    for predicted_category, sample_collection in samples_by_category.items():
        sample_collection.update_embeddings(feature_extractor)
        sample_collection.describe(get_description)
    logger.info("Embeddings updated and samples described.")

    # Create an instruction collection from the samples
    instructions: InstructionCollection = InstructionCollection.from_samples(samples)

    # Add embeddings and clusters to the instructions.
    (instructions
        .update_embeddings(feature_extractor)  # inplace operation
        .update_clusters(kmeans_clustering)  # inplace operation
     )

    # Group instructions by cluster.
    logger.info("Grouping instructions by cluster...")
    instructions_by_cluster: dict[int, InstructionCollection] = (
        instructions.group_by_cluster()
    )
    logger.info(f"Grouped instructions into {len(instructions_by_cluster)} clusters.")
    # Describe each instruction collection in the clusters.
    logger.info("Describing instructions in each cluster...")
    for cluster, instruction_collection in instructions_by_cluster.items():
        instruction_collection.describe(get_description)
    logger.info("Instructions described.")

    # Get the top 2 dissimilar instructions of cluster 2 using MMR.
    logger.info("Finding top 2 dissimilar instructions from cluster 2...")
    # dissimilar_instructions: InstructionCollection = (
    #     instructions
    #         # get all instructions from cluster 2
    #         .get_cluster(cluster=2)
    #         # select the top 2 dissimilar instructions
    #         .get_top_dissimilar_instructions(select_mmr, top_n=2)
    #  )
    # logger.info(f"Top 2 dissimilar instructions from cluster 2:"
    #             f" {dissimilar_instructions}")
    # logger.info(dissimilar_instructions)

    # Describe the dissimilar instructions using a custom description function.
    # descriptions = dissimilar_instructions.describe(
    #     description_function=get_description)
    # logger.info(descriptions)

    # Find the top 1 most similar sample_collection to each cluster.
    logger.info("Finding top similar sample categories for each instruction cluster...")
    id_to_category_pairs: IdToCategoryResultType = {}
    for cluster, instruction_collection in instructions_by_cluster.items():
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


    #
    cm = evaluate_results(id_to_category_pairs)
    logger.info("Evaluation results:")
    logger.info("Confusion matrix:")
    cm.print_matrix()
    logger.info("Summary statistics:")
    cm.stat(summary=True)
    cm.save_obj(
        ROOT_DIR / "evaluation_results" / "confusion_matrix.json"
    )
    logger.info(f"Execution time: {time.time() - start_time:.2f} seconds")
