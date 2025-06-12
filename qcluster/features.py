from loguru import logger
from sentence_transformers import SentenceTransformer
import torch
from datamodels import SampleCollection
from qcluster import ROOT_DIR

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
    return embeddings


if __name__ == '__main__':
    csv_file_path = (
        ROOT_DIR.parent
        / "data"
        / "Bitext_Sample_Customer_Support_Training_Dataset_27K_responses-v11.csv"
    )
    MODEL = SentenceTransformer('all-MiniLM-L6-v2')
    samples_ = SampleCollection.from_csv(csv_file_path)[:3]
    instruction_embeddings = create_embeddings(
        [i.instruction for i in samples_], MODEL)
    logger.info("Instruction Embeddings:")
    logger.info(instruction_embeddings)
    logger.info(f"Shape of embeddings: {instruction_embeddings.shape}")
