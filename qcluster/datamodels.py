from os import PathLike
from typing import Union, Optional, get_args, Any

import pandas as pd
from loguru import logger
from pydantic import BaseModel

from qcluster.consts import (
    EmbeddingFunctionType,
    EmbeddingType,
    ClusterType,
    ClusteringFunctionType, DissimilarityFunctionType, DescriptionFunctionType,
    SimilarityFunctionType, CategoryType, FlagType, IntentType
)


class ClusterOutput(BaseModel):
    """
    Represents the cluster output for a sample instruction.
    """
    cluster_id: int
    name: str
    description: str
    count: int

    def __repr__(self) -> str:
        """
        Get a string representation of the cluster output.

        Returns:
            str: A formatted string representing the cluster output.
        """
        return (f"{self.__class__.__name__}(id={self.cluster_id}, "
                f"name='{self.name}', description='{self.description}', "
                f"count={self.count})")

    def __str__(self) -> str:
        return self.__repr__()

class ClusterOutputCollection(BaseModel):
    """
    Represents a collection of cluster outputs.
    """
    outputs: list[ClusterOutput]

    def __len__(self) -> int:
        """
        Get the number of cluster outputs.

        Returns:
            int: The number of cluster outputs.
        """
        return len(self.outputs)

    def __repr__(self) -> str:
        """
        Get a string representation of the cluster output collection.

        Returns:
            str: A formatted string representing the cluster output collection.
        """
        indented_outputs = '\n'.join(f'    {output!r}' for output in self.outputs)
        return (f"{self.__class__.__name__}(\n"
                f"  num_outputs={len(self)},\n"
                f"  outputs=[\n"
                f"{indented_outputs[:20]}\n....."
                f"  ]\n"
                f")")

    def __str__(self) -> str:
        return self.__repr__()


class Sample(BaseModel):
    id: int
    flags: str
    instruction: str
    category: CategoryType
    intent: str
    response: str
    embedding: Optional[EmbeddingType] = None

    @staticmethod
    def all_category_classes() -> set[CategoryType]:
        return set(get_args(CategoryType))

    @staticmethod
    def all_flag_classes() -> set[FlagType]:
        return set(get_args(FlagType))

    @staticmethod
    def all_intent_classes() -> set[IntentType]:
        return set(get_args(IntentType))

    @property
    def embedding_shape(self) -> Optional[tuple[int, ...]]:
        """
        Get the shape of the embedding if it exists.

        Returns:
            Optional[tuple[int, ...]]: The shape of the embedding or None if not set.
        """
        return self.embedding.shape if hasattr(self.embedding, 'shape') else None

    def update_embedding(self, embedding_function: EmbeddingFunctionType):
        """
        Update the embedding of the sample using the provided embedding function.

        Args:
            embedding_function (EmbeddingFunction): A function that takes
             a list of strings and updates the embedding.
        """
        self.embedding = embedding_function([self.instruction])

    def __repr__(self) -> str:
        """
        Get a string representation of the sample.

        Returns:
            str: A formatted string representing the sample.
        """
        shape = "NA" if self.embedding is None else self.embedding_shape
        return (f"Sample(id={self.id}, flags='{self.flags}',"
                f" instruction='{self.instruction}',"
                f" category='{self.category}', intent='{self.intent}',"
                f" response='{self.response[:40]}...',"
                f" embedding_shape={shape})")

    def __str__(self) -> str:
        return self.__repr__()

class SampleCollection(BaseModel):
    samples: list[Sample]
    description: Optional[str] = None
    title: Optional[str] = None

    @staticmethod
    def all_category_classes() -> set[CategoryType]:
        return Sample.all_category_classes()

    @staticmethod
    def all_flag_classes() -> set[FlagType]:
        return Sample.all_flag_classes()

    @staticmethod
    def all_intent_classes() -> set[IntentType]:
        return Sample.all_intent_classes()

    def is_a_category(self) -> bool:
        return len(set(sample.category for sample in self.samples)) == 1

    def is_intent(self) -> bool:
        return len(set(sample.intent for sample in self.samples)) == 1

    @property
    def category(self) -> Optional[CategoryType]:
        """
        Get the category of the samples if they all belong to the same category.

        Returns:
            Optional[CategoryType]: The category of the samples or None if not applicable.
        """
        if not self.is_a_category():
            raise ValueError("Samples do not belong to a single category."
                             " Please use `group_by_category` to group them first.")
        return self.samples[0].category


    @property
    def intent(self):
        if not self.is_intent():
            raise ValueError("Samples do not belong to a single intent."
                             " Please use `group_by_intent` to group them first.")
        return self.samples[0].intent


    @property
    def embeddings(self) -> list[Optional[EmbeddingType]]:
        """
        Get the embeddings of the samples.

        Returns:
            list[Optional[EmbeddingType]]: A list of embeddings for each sample.
        """
        return [sample.embedding for sample in self.samples]

    def get_sample_by_id(self, sample_id: int) -> Optional[Sample]:
        """
        Get a sample by its ID.

        Args:
            sample_id (int): The ID of the sample to retrieve.

        Returns:
            Optional[Sample]: The Sample object with the specified ID
             or None if not found.
        """
        for sample in self.samples:
            if sample.id == sample_id:
                return sample
        return None

    def update_embeddings(
            self, embedding_function: EmbeddingFunctionType) -> 'SampleCollection':
        """
        Add embeddings to the samples using the provided embedding function.

        Args:
            embedding_function (EmbeddingFunction): A function that takes
             a list of strings and updates the embeddings.
        """
        instructions = [sample.instruction for sample in self.samples]
        embeddings = embedding_function(instructions)
        for sample, embedding in zip(self.samples, embeddings):
            sample.embedding = embedding
        return self

    def __repr__(self) -> str:
        """
        Get a string representation of the sample collection.

        Returns:
            str: A formatted string representing the sample collection.
        """
        indented_samples = '\n'.join(f'    {sample!r}' for sample in self)
        return (f"{self.__class__.__name__}(\n"
                f"  num_samples={len(self)},\n"
                f"  samples=[\n"
                f"{indented_samples}\n"
                f"  ]\n"
                f")")

    def __str__(self) -> str:
        return self.__repr__()

    def __len__(self) -> int:
        """
        Get the number of samples.

        Returns:
            int: The number of samples.
        """
        return len(self.samples)

    def __iter__(self):
        """
        Iterate over the samples.

        Returns:
            iterator: An iterator over the Sample objects.
        """
        return iter(self.samples)

    def __getitem__(self, index: int) -> Union[Sample, 'SampleCollection']:
        """
        Get a sample or a slice of samples by index.

        Args:
            index (int): Index of the sample or slice.

        Returns:
            Sample or SampleCollection: A single Sample object
             or a SampleCollection object if a slice is requested.
        """
        if isinstance(index, slice):
            return SampleCollection(samples=self.samples[index])
        return self.samples[index]

    def _group_by(
            self, attr: str) -> dict[
                    Any, 'SampleCollection']:
        """
        Generic method to group samples by a specified attribute.

        Args:
            attr (str): The attribute to group by ('category' or 'intent').

        Returns:
            dict[Union[CategoryType, IntentType], SampleCollection]: A dictionary
             where keys are the attribute values and values are SampleCollection
             objects containing the samples for each attribute value.
        """
        grouped_samples = {}
        for sample in self.samples:
            key = getattr(sample, attr)
            if key not in grouped_samples:
                grouped_samples[key] = []
            grouped_samples[key].append(sample)
        return {key: SampleCollection(samples=samples)
                for key, samples in grouped_samples.items()}

    def group_by_category(self) -> dict[CategoryType, 'SampleCollection']:
        """
        Group samples by their categories.

        Returns:
            dict[CategoryType, SampleCollection]: A dictionary where keys are
             category labels and values are SampleCollection objects containing
             the samples for each category.
        """
        return self._group_by('category')

    def group_by_intent(self) -> dict[IntentType, 'SampleCollection']:
        """
        Group samples by their intents.

        Returns:
            dict[IntentType, SampleCollection]: A dictionary where keys are
             intent labels and values are SampleCollection objects containing
             the samples for each intent.
        """
        return self._group_by('intent')

    def filter_by_category(self, category: CategoryType) -> 'SampleCollection':
        """
        Collect samples by a specific category.

        Args:
            category (CategoryType): The category to filter samples by.

        Returns:
            SampleCollection: A new SampleCollection object containing
             only the samples that belong to the specified category.
        """
        filtered_samples = [sample for sample in self
                            if sample.predicted_category == category]
        return SampleCollection(samples=filtered_samples)

    def describe(
            self, description_function: DescriptionFunctionType) -> ClusterOutput:
        """
        Get a description of the sample collection.

        Args:
            description_function (DescriptionFunctionType): A function that takes
             a string and returns a ClusterDescription object.

        Returns:
            ClusterDescription: A description of the sample collection.
        """
        if not self.samples:
            raise ValueError("No samples available for description.")
        document = "\n".join(sample.instruction for sample in self.samples)
        description_output = description_function(document)
        self.description = description_output.description
        self.title = description_output.title
        return ClusterOutput(
            cluster_id=0,  # Assuming a single cluster for the entire collection
            name=description_output.title,
            description=description_output.description,
            count=len(self.samples)
        )

    @classmethod
    def from_csv(cls, csv_file: PathLike) -> 'SampleCollection':
        """
        Create a `SampleCollection` object from a CSV file.

        Args:
            csv_file (str): Path to the CSV file.

        Returns:
            SampleCollection: A SampleCollection object containing the data from the CSV.
        """
        df = pd.read_csv(csv_file, dtype=str)
        df.index.name = "id"
        df.reset_index(inplace=True)
        # Shuffle the DataFrame to avoid any order dependencies
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)
        samples = [Sample(**row) for _, row in df.iterrows()]
        return cls(samples=samples)

    def number_of_samples(self) -> int:
        """
        Get the number of samples.

        Returns:
            int: The number of samples.
        """
        return len(self.samples)

    def description_embedding(self,
                              embedding_function: EmbeddingFunctionType) -> EmbeddingType:
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
        return self.embedding.shape if hasattr(self.embedding, 'shape') else None

    @classmethod
    def from_sample(cls, sample: Sample) -> 'Instruction':
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
        return (f"Instruction(id={self.id}, text='{self.text}',"
                f" embedding_shape={shape}, cluster={self.cluster})")

    def __str__(self) -> str:
        return self.__repr__()

class InstructionCollection(BaseModel):
    instructions: list[Instruction]
    description: Optional[str] = None
    title: Optional[str] = None

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
    def from_samples(cls, samples: SampleCollection) -> 'InstructionCollection':
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
            raise ValueError("Instructions do not belong to a single cluster."
                             " Please use `group_by_cluster` to group them first.")

    def update_embeddings(
            self, embedding_function: EmbeddingFunctionType) -> 'InstructionCollection':
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
            self, clustering_function: ClusteringFunctionType) -> 'InstructionCollection':
        """
        Add clusters to the instructions using the provided clustering function.

        Args:
            clustering_function (ClusteringFunction): A function that takes
             a list of embeddings and updates cluster labels.
        """
        if not self.embeddings:
            raise ValueError("Embeddings must be updated before clustering.")
        elif len(self.embeddings) != len(self.instructions):
            raise ValueError("Embeddings and instructions must have the same length.")
        clusters = clustering_function(self.embeddings)

        for instruction, cluster in zip(self.instructions, clusters):
            instruction.cluster = cluster
        return self

    def get_top_dissimilar_instructions(
            self, dissimilarity_function: DissimilarityFunctionType,
            top_n: int = 5, raise_if_too_few: bool = False
    ) -> 'InstructionCollection':
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
                    f" ({len(self)}).")
            else:
                logger.warning(
                    f"top_n ({top_n}) is greater than the number of instructions"
                    f" ({len(self)}). Returning all instructions.")
                top_n = len(self.instructions)
        instruction_strings = [instruction.text for instruction in self]
        dissimilarities = dissimilarity_function(instruction_strings, top_n)
        top_indices = [index for index, _ in dissimilarities]
        top_instructions = [self[index] for index in top_indices]
        return InstructionCollection(instructions=top_instructions)


    def __repr__(self) -> str:
        indented_instructions = '\n'.join(f'    {instruction!r}' for instruction in self)
        return (f"{self.__class__.__name__}(\n"
                f"  num_instructions={len(self)},\n"
                f"  instructions=[\n"
                f"{indented_instructions}\n"
                f"  ]\n"
                f")")

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

    def __getitem__(self, index: int) -> Union[Instruction, 'InstructionCollection']:
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

    def get_cluster(self, cluster: ClusterType) -> 'InstructionCollection':
        """
        Collect instructions by a specific cluster.

        Args:
            cluster (ClusterType): The cluster to filter instructions by.

        Returns:
            InstructionCollection: A new InstructionCollection object containing
             only the instructions that belong to the specified cluster.
        """
        filtered_instructions = [
            instruction for instruction in self if instruction.cluster == cluster]
        return InstructionCollection(instructions=filtered_instructions)

    def group_by_cluster(self) -> dict[ClusterType, 'InstructionCollection']:
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
        return {cluster: InstructionCollection(instructions=instructions)
                for cluster, instructions in grouped_instructions.items()}

    def is_a_cluster(self) -> bool:
        if len(set(instruction.cluster for instruction in self.instructions)) == 1:
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
            raise ValueError("Instructions do not belong to a single cluster."
                             " Please use `group_by_cluster` to group them first.")
        description_output = description_function(document)
        self.description = description_output.description
        self.title = description_output.title
        return ClusterOutput(
            cluster_id=self.instructions[0].cluster,
            name=description_output.title,
            description=description_output.description,
            count=len(self.instructions)
        )

    def description_embedding(self,
                              embedding_function: EmbeddingFunctionType) -> EmbeddingType:
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
            sample_collections: list[SampleCollection],
            similarity_function: SimilarityFunctionType,
            top_n: int = 5
    ) -> list[SampleCollection]:
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

        Returns:
            SampleCollection: A new SampleCollection object containing the most similar
             samples.
        """
        if not self.description:
            raise ValueError("No description available for similarity calculation.")
        if not sample_collections:
            raise ValueError(
                "No sample collections available for similarity calculation.")
        for sample_collection in sample_collections:
            if not sample_collection.description:
                raise ValueError(
                    "All sample collections must have a description for similarity"
                    " calculation."
                    f"The following collection is missing a description:"
                    f" {sample_collection[:3]}..."
                )
        cluster_description = self.description
        sample_descriptions = [sc.description for sc in sample_collections]
        similarities = similarity_function(
            cluster_description, sample_descriptions, top_n)
        top_indices = [index for index, _ in similarities]
        top_samples = [
            sample_collections[index] for index in top_indices
        ]
        return top_samples

    def get_cluster_category(self,
            sample_collections: list[SampleCollection],
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
            raise ValueError("Instructions do not belong to a single cluster."
                             " Please use `group_by_cluster` to group them first.")
        similar_sample_collections = self.find_top_similar_sample_collections(
            sample_collections, similarity_function, top_n=1)
        if not similar_sample_collections:
            raise ValueError("No similar samples found in the collection.")
        similar_sample_collection = similar_sample_collections[0]
        category = similar_sample_collection.category
        if category is None:
            return 'UNKNOWN'
        else:
            return category
