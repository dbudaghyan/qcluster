from sentence_transformers import SentenceTransformer

from qcluster import ROOT_DIR
from qcluster.datamodels import SampleCollection, InstructionCollection
from qcluster.features import create_embeddings

def feature_engineer(texts: list[str]):
    return create_embeddings(
        texts=texts,
        model=SentenceTransformer('all-MiniLM-L6-v2')
    )
csv_file_path = (
        ROOT_DIR.parent
        / "data"
        / "Bitext_Sample_Customer_Support_Training_Dataset_27K_responses-v11.csv"
    )

if __name__ == '__main__':
    samples = SampleCollection.from_csv(csv_file_path)
    samples = samples[:5]
    instructions = InstructionCollection.from_samples(samples)
    instructions.update_embeddings(feature_engineer)
    print(instructions[:2])
