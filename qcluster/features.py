from sentence_transformers import SentenceTransformer
import torch
from datamodels import Samples, Instructions
from qcluster import ROOT_DIR


def create_instruction_embeddings(samples: Samples) -> torch.Tensor:
    """
    Creates embeddings for the categories of the given samples.

    Args:
        samples: A Samples object containing the list of samples.

    Returns:
        A torch.Tensor containing the embeddings for the categories.
    """
    model = SentenceTransformer('all-MiniLM-L6-v2')
    instructions = (
        Instructions
        .from_samples(samples)
        .to_list_of_strings()
    )
    assert isinstance(instructions[0], str)
    embeddings = model.encode(instructions, convert_to_tensor=True)
    return embeddings


if __name__ == '__main__':
    csv_file_path = (
        ROOT_DIR.parent
        / "data"
        / "Bitext_Sample_Customer_Support_Training_Dataset_27K_responses-v11.csv"
    )
    samples = Samples.from_csv(csv_file_path)[:10]
    # Generate embeddings
    instruction_embeddings = create_instruction_embeddings(samples)

    # Print the embeddings
    print("Instruction Embeddings:")
    print(instruction_embeddings)
    print(f"Shape of embeddings: {instruction_embeddings.shape}")
