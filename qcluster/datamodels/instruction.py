from typing import Union, Optional

from loguru import logger
from pydantic import BaseModel

from qcluster.custom_types import (
    EmbeddingFunctionType,
    EmbeddingType,
    ClusterType,
    ClusteringFunctionType,
    DissimilarityFunctionType,
    DescriptionFunctionType,
    SimilarityFunctionType,
    CategoryType,
)
from qcluster.datamodels.output import ClusterOutput


class Instruction(BaseModel):
    id: int
    text: str
    embedding: Optional[EmbeddingType] = None
    cluster: Optional[ClusterType] = None

    @property
    def embedding_shape(self) -> Optional[tuple[int, ...]]:
        """
        Get the shape of the embedding if it exists.

        Returns:
            Optional[tuple[int, ...]]: The shape of the embedding or None if not set.
        """
        return self.embedding.shape if hasattr(self.embedding, "shape") else None

    @classmethod
    def from_sample(cls, sample: "Sample") -> "Instruction":
        """
        Create an Instruction object from a Sample object.

        Args:
            sample (Sample): A Sample object.

        Returns:
            Instruction: An Instruction object with the instruction from the Sample.
        """
        return cls(id=sample.id, text=sample.instruction)

    def update_embedding(self, embedding_function: EmbeddingFunctionType):
        """
        Update the embedding of the instruction using the provided embedding function.

        Args:
            embedding_function (EmbeddingFunction): A function that takes
             a list of strings and updates the embedding.
        """
        self.embedding = embedding_function([self.text])

    def __repr__(self) -> str:
        """
        Get a string representation of the instruction.

        Returns:
            str: The text of the instruction.
        """
        shape = "NA" if self.embedding is None else self.embedding_shape
        return (
            f"Instruction(id={self.id}, text='{self.text}',"
            f" embedding_shape={shape}, cluster={self.cluster})"
        )

    def __str__(self) -> str:
        return self.__repr__()


class InstructionCollection(BaseModel):
    instructions: list[Instruction]
    description: Optional[str] = None
    title: Optional[str] = None

    @property
    def count(self):
        return len(self.instructions)

    @property
    def clusters(self) -> list[Optional[ClusterType]]:
        """
        Get the clusters of the instructions.

        Returns:
            list[Optional[ClusterType]]: A list of clusters for each instruction.
        """
        return [instruction.cluster for instruction in self.instructions]

    @property
    def embeddings(self) -> list[Optional[EmbeddingType]]:
        """
        Get the embeddings of the instructions.

        Returns:
            list[Optional[EmbeddingType]]: A list of embeddings for each instruction.
        """
        return [instruction.embedding for instruction in self.instructions]

    @classmethod
    def from_samples(cls, samples: "SampleCollection") -> "InstructionCollection":
        """
        Create an InstructionCollection object from a list of Sample objects.

        Args:
            samples (list[Sample]): A list of SampleCollection objects.

        Returns:
            InstructionCollection: An InstructionCollection object
             containing the instructions from the SampleCollection.
        """
        instructions = [Instruction.from_sample(sample=sample) for sample in samples]
        return cls(instructions=instructions)

    @property
    def cluster(self):
        if self.is_a_cluster():
            return self.instructions[0].cluster
        else:
            raise ValueError(
                "Instructions do not belong to a single cluster."
                " Please use `group_by_cluster` to group them first."
            )

    def update_embeddings(
        self, embedding_function: EmbeddingFunctionType
    ) -> "InstructionCollection":
        """
        Add embeddings to the instructions using the provided embedding function.

        Args:
            embedding_function (EmbeddingFunction): A function that takes
             a list of strings and updates the embeddings.
        """
        instructions_text = [instruction.text for instruction in self.instructions]
        embeddings = embedding_function(instructions_text)
        for instruction, embedding in zip(self.instructions, embeddings):
            instruction.embedding = embedding
        return self

    def update_clusters(
        self,
        clustering_function: ClusteringFunctionType,
        use_raw_instructions: bool = True,
    ) -> "InstructionCollection":
        """
        Add clusters to the instructions using the provided clustering function.

        Args:
            clustering_function (ClusteringFunction): A function that takes
             a list of embeddings and updates cluster labels.
            use_raw_instructions (bool): If True, uses the raw instruction text
        """
        if use_raw_instructions:
            clusters = clustering_function(
                [instruction.text for instruction in self.instructions]
            )
        else:
            if self.has_empty_embeddings():
                raise ValueError("One or more embeddings are empty. Cannot cluster.")
            elif len(self.embeddings) != len(self.instructions):
                raise ValueError(
                    "Embeddings and instructions must have the same length."
                )
            clusters = clustering_function(self.embeddings)

        for instruction, cluster in zip(self.instructions, clusters):
            instruction.cluster = cluster
        return self

    def get_top_dissimilar_instructions(
        self,
        dissimilarity_function: DissimilarityFunctionType,
        top_n: int = 5,
        raise_if_too_few: bool = False,
    ) -> "InstructionCollection":
        """Get the top N dissimilar instructions based on a dissimilarity function.
        Args:
            dissimilarity_function (DissimilarityFunction): A function that takes
             a list of strings, and an integer (top_n),
             and returns a list of tuples containing the index
             and the corresponding string.
            top_n (int): The number of top dissimilar instructions to return.
            raise_if_too_few (bool): If True, raises an error if there are not enough
             instructions to return the top N dissimilar instructions.
        Returns:
            InstructionCollection: A new InstructionCollection object containing
             the top N dissimilar instructions.
        """
        if not self.instructions:
            raise ValueError("No instructions available for dissimilarity calculation.")
        if top_n <= 0:
            raise ValueError("top_n must be a positive integer.")
        if top_n > len(self.instructions):
            if raise_if_too_few:
                raise ValueError(
                    f"top_n ({top_n}) cannot be greater than the number of instructions"
                    f" ({len(self)})."
                )
            else:
                logger.warning(
                    f"top_n ({top_n}) is greater than the number of instructions"
                    f" ({len(self)}). Returning all instructions."
                )
                top_n = len(self.instructions)
        instruction_strings = [instruction.text for instruction in self]
        dissimilarities = dissimilarity_function(instruction_strings, top_n)
        top_indices = [index for index, _ in dissimilarities]
        top_instructions = [self[index] for index in top_indices]
        return InstructionCollection(instructions=top_instructions)

    def __repr__(self) -> str:
        indented_instructions = "\n".join(
            f"    {instruction!r}" for instruction in self
        )
        return (
            f"{self.__class__.__name__}(\n"
            f"  num_instructions={len(self)},\n"
            f"  instructions=[\n"
            f"{indented_instructions}\n"
            f"  ]\n"
            f")"
        )

    def __str__(self) -> str:
        return self.__repr__()

    def __len__(self) -> int:
        """
        Get the number of instructions.

        Returns:
            int: The number of instructions.
        """
        return len(self.instructions)

    def __iter__(self):
        """
        Iterate over the instructions.

        Returns:
            iterator: An iterator over the Instruction objects.
        """
        return iter(self.instructions)

    def __getitem__(self, index: int) -> Union[Instruction, "InstructionCollection"]:
        """
        Get an instruction or a slice of instructions by index.

        Args:
            index (int): Index of the instruction or slice.

        Returns:
            Instruction or InstructionCollection: A single Instruction object
             or an InstructionCollection object if a slice is requested.
        """
        if isinstance(index, slice):
            return InstructionCollection(instructions=self.instructions[index])
        return self.instructions[index]

    def to_list_of_strings(self) -> list[str]:
        """
        Convert the instructions to a list of strings.

        Returns:
            list[str]: A list of instruction strings.
        """
        return [instruction.text for instruction in self]

    def get_cluster(self, cluster: ClusterType) -> "InstructionCollection":
        """
        Collect instructions by a specific cluster.

        Args:
            cluster (ClusterType): The cluster to filter instructions by.

        Returns:
            InstructionCollection: A new InstructionCollection object containing
             only the instructions that belong to the specified cluster.
        """
        filtered_instructions = [
            instruction for instruction in self if instruction.cluster == cluster
        ]
        return InstructionCollection(instructions=filtered_instructions)

    def group_by_cluster(self) -> dict[ClusterType, "InstructionCollection"]:
        """
        Group instructions by their clusters.

        Returns:
            dict[ClusterType, InstructionCollection]: A dictionary where keys are
             cluster labels and values are InstructionCollection objects containing
             the instructions for each cluster.
        """
        grouped_instructions = {}
        for instruction in self.instructions:
            if instruction.cluster not in grouped_instructions:
                grouped_instructions[instruction.cluster] = []
            grouped_instructions[instruction.cluster].append(instruction)
        return {
            cluster: InstructionCollection(instructions=instructions)
            for cluster, instructions in grouped_instructions.items()
        }

    def is_a_cluster(self) -> bool:
        unique_clusters = set(instruction.cluster for instruction in self.instructions)
        if len(unique_clusters) == 1 and None not in unique_clusters:
            return True
        return False

    def describe(self, description_function: DescriptionFunctionType) -> ClusterOutput:
        """
        Get a description of the instruction collection.

        Args:
            description_function (DescriptionFunctionType): A function that takes
             a string and returns a ClusterDescription object.

        Returns:
            ClusterDescription: A description of the instruction collection.
        """
        if not self.instructions:
            raise ValueError("No instructions available for description.")
        document = "\n".join(self.to_list_of_strings())
        if not self.is_a_cluster():
            raise ValueError(
                "Instructions do not belong to a single cluster."
                " Please use `group_by_cluster` to group them first."
            )
        description_output = description_function(document)
        self.description = description_output.description
        self.title = description_output.title
        return ClusterOutput(
            cluster_id=self.instructions[0].cluster,
            name=description_output.title,
            description=description_output.description,
            count=len(self.instructions),
        )

    def description_embedding(
        self, embedding_function: EmbeddingFunctionType
    ) -> EmbeddingType:
        """
        Get the embedding of the description.

        Args:
            embedding_function (EmbeddingFunction): A function that takes
             a list of strings and returns a list of embeddings.

        Returns:
            EmbeddingType: The embedding of the description.
        """
        if not self.description:
            raise ValueError("No description available for embedding.")
        return embedding_function([self.description])[0]

    def find_top_similar_sample_collections(
        self,
        sample_collections: list["SampleCollection"],
        similarity_function: SimilarityFunctionType,
        top_n: int = 5,
        use_centroid: bool = False,
    ) -> list["SampleCollection"]:
        """
        Find the most similar samples from a SampleCollection based on the instructions.

        Args:
            sample_collections (list[SampleCollection]):
                The collection of samples to compare
             against.
            similarity_function (SimilarityFunctionType): A function that takes a list
             of strings
             and an integer (top_n) and returns a list of tuples containing the index
             and the corresponding string.
            top_n (int): The number of top similar samples to return.
            use_centroid (bool): If True, uses the centroid of the cluster,

        Returns:
            SampleCollection: A new SampleCollection object containing the most similar
             samples.
        """
        if use_centroid:
            cluster_emb = self.centroid
        else:
            if not self.description:
                raise ValueError("No description available for similarity calculation.")
            cluster_emb = self.description
        if not sample_collections:
            raise ValueError(
                "No sample collections available for similarity calculation."
            )
        for sample_collection in sample_collections:
            if use_centroid:
                break
            if not sample_collection.description:
                raise ValueError(
                    "All sample collections must have a description for similarity"
                    " calculation."
                    f"The following collection is missing a description:"
                    f" {sample_collection[:3]}..."
                )
        if use_centroid:
            category_embs = [sc.centroid for sc in sample_collections]
        else:
            category_embs = [sc.description for sc in sample_collections]
        similarities = similarity_function(cluster_emb, category_embs, top_n)
        top_indices = [index for index, _ in similarities]
        top_samples = [sample_collections[index] for index in top_indices]
        return top_samples

    def get_cluster_category(
        self,
        sample_collections: list["SampleCollection"],
        similarity_function: SimilarityFunctionType,
    ) -> CategoryType:
        """
        Get the most common category of samples in the cluster.

        Args:
            sample_collections (list[SampleCollection]):
                The collection of samples to compare against.
            similarity_function (SimilarityFunctionType): A function that takes a list
             of strings
             and an integer (top_n) and returns a list of tuples containing the index
             and the corresponding string.

        Returns:
            CategoryType: The most common category in the cluster.
        """
        if not self.is_a_cluster():
            raise ValueError(
                "Instructions do not belong to a single cluster."
                " Please use `group_by_cluster` to group them first."
            )
        similar_sample_collections = self.find_top_similar_sample_collections(
            sample_collections, similarity_function, top_n=1
        )
        if not similar_sample_collections:
            raise ValueError("No similar samples found in the collection.")
        similar_sample_collection = similar_sample_collections[0]
        category = similar_sample_collection.category
        if category is None:
            return "UNKNOWN"
        else:
            return category

    @property
    def centroid(self) -> Optional[EmbeddingType]:
        # Check that there are no empty embeddings
        if self.has_empty_embeddings():
            raise ValueError("One or more embeddings are empty. Cannot compute centroid.")
        return sum(self.embeddings) / len(self.embeddings)


    def has_empty_embeddings(self) -> bool:
        """
        Check if any instruction has an empty embedding.

        Returns:
            bool: True if any instruction has an empty embedding, False otherwise.
        """
        return any(embedding is None or len(embedding) == 0
                   for embedding in self.embeddings)

if __name__ == "__main__":
    # Used for pycharm type checking to work properly while avoiding circular imports
    from qcluster.datamodels.sample import Sample, SampleCollection
