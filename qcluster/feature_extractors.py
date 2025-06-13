import os
from typing import Optional

from loguru import logger
from sentence_transformers import SentenceTransformer
import torch
from umap import UMAP

from qcluster.datamodels.sample import SampleCollection
from qcluster import ROOT_DIR
from sklearn.decomposition import PCA
from bertopic import BERTopic


def bert_topic_extraction(
    texts: list[str],
    model: SentenceTransformer = None,
    n_topics: Optional[int] = 5,
):
    embeddings = create_embeddings(texts, model)
    if not model:
        model = SentenceTransformer(os.environ['SENTENCE_TRANSFORMERS_MODEL'])
    topic_model = BERTopic(
        embedding_model=model,
        nr_topics=n_topics,
        verbose=True,
    )
    topics, probs = topic_model.fit_transform(texts)
    raise NotImplementedError

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

def pca_reduction(embeddings: torch.Tensor, n_components: int) -> torch.Tensor:
    """
    Reduces the dimensionality of the embeddings using PCA.

    Args:
        embeddings: A torch.Tensor containing the embeddings to be reduced.
        n_components: The number of components to keep after reduction.
    Returns:
        A torch.Tensor containing the reduced embeddings.
    """
    pca = PCA(n_components=n_components, random_state=42)
    reduced_embeddings = pca.fit_transform(embeddings.cpu().numpy())
    return torch.tensor(reduced_embeddings)

def umap_reduction(embeddings: torch.Tensor,
                   n_components: int,
                   n_neighbors=15,
                   metric="euclidean",
                   metric_kwds=None,
                   output_metric="euclidean",
                   output_metric_kwds=None,
                   n_epochs=None,
                   learning_rate=1.0,
                   init="spectral",
                   min_dist=0.1,
                   spread=1.0,
                   low_memory=True,
                   n_jobs=-1,
                   set_op_mix_ratio=1.0,
                   local_connectivity=1.0,
                   repulsion_strength=1.0,
                   negative_sample_rate=5,
                   transform_queue_size=4.0,
                   a=None,
                   b=None,
                   angular_rp_forest=False,
                   target_n_neighbors=-1,
                   target_metric="categorical",
                   target_metric_kwds=None,
                   target_weight=0.5,
                   transform_seed=42,
                   transform_mode="embedding",
                   force_approximation_algorithm=False,
                   verbose=False,
                   tqdm_kwds=None,
                   unique=False,
                   densmap=False,
                   dens_lambda=2.0,
                   dens_frac=0.3,
                   dens_var_shift=0.1,
                   output_dens=False,
                   disconnection_distance=None,
                   precomputed_knn=(None, None, None),
                   ) -> torch.Tensor:
    umap = UMAP(
        n_components=n_components,
        random_state=42,
        n_neighbors=n_neighbors,
        metric=metric,
        metric_kwds=metric_kwds,
        output_metric=output_metric,
        output_metric_kwds=output_metric_kwds,
        n_epochs=n_epochs,
        learning_rate=learning_rate,
        init=init,
        min_dist=min_dist,
        spread=spread,
        low_memory=low_memory,
        n_jobs=n_jobs,
        set_op_mix_ratio=set_op_mix_ratio,
        local_connectivity=local_connectivity,
        repulsion_strength=repulsion_strength,
        negative_sample_rate=negative_sample_rate,
        transform_queue_size=transform_queue_size,
        a=a,
        b=b,
        angular_rp_forest=angular_rp_forest,
        target_n_neighbors=target_n_neighbors,
        target_metric=target_metric,
        target_metric_kwds=target_metric_kwds,
        target_weight=target_weight,
        transform_seed=transform_seed,
        transform_mode=transform_mode,
        force_approximation_algorithm=force_approximation_algorithm,
        verbose=verbose,
        tqdm_kwds=tqdm_kwds,
        unique=unique,
        densmap=densmap,
        dens_lambda=dens_lambda,
        dens_frac=dens_frac,
        dens_var_shift=dens_var_shift,
        output_dens=output_dens,
        disconnection_distance=disconnection_distance,
        precomputed_knn=precomputed_knn,
    )
    reduced_embeddings = umap.fit_transform(embeddings.cpu().numpy())
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
