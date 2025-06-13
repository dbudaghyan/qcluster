from os import PathLike
from typing import Optional, get_args, Union, Any

import pandas as pd
from pydantic import BaseModel

from qcluster.custom_types import (
    CategoryType,
    EmbeddingType,
    FlagType,
    IntentType,
    EmbeddingFunctionType,
    DescriptionFunctionType
)
from qcluster.datamodels.output import ClusterOutput


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

    @property
    def centroid(self) -> Optional[EmbeddingType]:
        return sum(self.embeddings) / len(self.embeddings)

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
