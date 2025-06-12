import functools
from os import PathLike

from sentence_transformers import SentenceTransformer

from qcluster import ROOT_DIR
from qcluster.clustering import kmeans_clustering
from qcluster.consts import EmbeddingFunctionType
from qcluster.datamodels import SampleCollection, InstructionCollection
from qcluster.describer import get_description
from qcluster.dissimilarity import select_mmr
from qcluster.feature_extractors import create_embeddings
from dotenv import load_dotenv


load_dotenv()

# Define the feature extractor, which can be any function that takes
# a list of strings and returns a collection of embeddings.
feature_extractor: EmbeddingFunctionType = functools.partial(
    create_embeddings,
    model=SentenceTransformer('all-MiniLM-L6-v2')
)

CSV_PATH: PathLike = (ROOT_DIR.parent
            / "data"
            / "Bitext_Sample_Customer_Support_Training_Dataset_27K_responses-v11.csv")


if __name__ == '__main__':

    # Load the samples from the CSV file.
    samples: SampleCollection = SampleCollection.from_csv(CSV_PATH)
    # Limit the number of samples for demonstration purposes.
    samples: SampleCollection = samples[:20]
    # Create an instruction collection from the samples
    instructions: InstructionCollection = InstructionCollection.from_samples(samples)
    # Add embeddings and clusters to the instructions.
    (instructions
        .update_embeddings(feature_extractor)  # inplace operation
        .update_clusters(kmeans_clustering)  # inplace operation
     )
    # Get the top 2 dissimilar instructions of cluster 2 using MMR.
    dissimilar_instructions: InstructionCollection = (
        instructions
            # get all instructions from cluster 2
            .get_cluster(cluster=2)
            # select the top 2 dissimilar instructions
            .get_top_dissimilar_instructions(select_mmr, top_n=2)
     )
    print(dissimilar_instructions)
    descriptions = dissimilar_instructions.describe(
        description_function=get_description)
    print(descriptions)
