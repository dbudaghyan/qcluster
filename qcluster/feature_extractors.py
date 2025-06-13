from loguru import logger
from sentence_transformers import SentenceTransformer
import torch
from datamodels import SampleCollection
from qcluster import ROOT_DIR
from sklearn.decomposition import PCA


def create_embeddings(texts: list[str],
                      model: SentenceTransformer) -> torch.Tensor:
    """
    Creates embeddings for the given texts using the specified model.

    Args:
        texts: A list of strings to be encoded.
        model: A SentenceTransformer model used to encode the texts.
    Returns:
        A torch.Tensor containing the embeddings for the texts.
    """
    assert isinstance(texts[0], str)
    embeddings = model.encode(texts, convert_to_tensor=True)
    if not embeddings.is_cpu:
        embeddings = embeddings.cpu()
    return embeddings

def pca_reduction(embeddings: torch.Tensor, n_components: int = 2) -> torch.Tensor:
    """
    Reduces the dimensionality of the embeddings using PCA.

    Args:
        embeddings: A torch.Tensor containing the embeddings to be reduced.
        n_components: The number of components to keep after reduction.
    Returns:
        A torch.Tensor containing the reduced embeddings.
    """
    pca = PCA(n_components=n_components)
    reduced_embeddings = pca.fit_transform(embeddings.cpu().numpy())
    return torch.tensor(reduced_embeddings)


if __name__ == '__main__':
    from qcluster.models import MODEL

    csv_file_path = (
            ROOT_DIR.parent
            / "data"
            / "Bitext_Sample_Customer_Support_Training_Dataset_27K_responses-v11.csv"
    )
    samples_ = SampleCollection.from_csv(csv_file_path)[:3]
    instruction_embeddings = create_embeddings(
        [i.instruction for i in samples_], MODEL)
    instruction_embeddings = pca_reduction(instruction_embeddings,
                                           n_components=20)
    logger.info("Instruction Embeddings:")
    logger.info(instruction_embeddings)
    logger.info(f"Shape of embeddings: {instruction_embeddings.shape}")
