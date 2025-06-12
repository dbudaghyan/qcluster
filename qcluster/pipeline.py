import functools
from os import PathLike

from sentence_transformers import SentenceTransformer

from qcluster import ROOT_DIR
from qcluster.clustering import kmeans_clustering
from qcluster.consts import EmbeddingFunctionType
from qcluster.datamodels import SampleCollection, InstructionCollection
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

    samples: SampleCollection = SampleCollection.from_csv(CSV_PATH)
    samples: SampleCollection = samples[:10]
    instructions: InstructionCollection = InstructionCollection.from_samples(samples)
    (
        instructions
        .update_embeddings(feature_extractor)
        .update_clusters(kmeans_clustering)
     )
    print(instructions)
