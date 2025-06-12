from typing import Literal, Union, Optional, Any

import pandas as pd
from pydantic import BaseModel, Field

from qcluster.consts import EmbeddingFunction

CategoryType = Literal[
    'ACCOUNT',
    'CANCEL',
    'CONTACT',
    'DELIVERY',
    'FEEDBACK',
    'INVOICE',
    'ORDER',
    'PAYMENT',
    'REFUND',
    'SHIPPING',
    'SUBSCRIPTION'
]

FlagType = Literal['N', 'B', 'P', 'Z', 'V', 'E', 'M', 'S', 'C', 'W', 'Q', 'L', 'K', 'I']

IntentType = Literal[
    'delivery_options', 'create_account', 'change_shipping_address',
    'place_order', 'contact_human_agent', 'complaint',
    'newsletter_subscription', 'delete_account',
    'registration_problems', 'check_cancellation_fee', 'switch_account',
    'change_order', 'check_payment_methods', 'track_order',
    'check_refund_policy', 'track_refund', 'check_invoice',
    'payment_issue', 'contact_customer_service', 'cancel_order',
    'delivery_period', 'set_up_shipping_address', 'get_invoice',
    'recover_password', 'edit_account', 'get_refund', 'review'
]


class Sample(BaseModel):
    id: int
    flags: str
    instruction: str
    category: CategoryType
    intent: str
    response: str

class Instruction(BaseModel):
    id: int
    text: str
    embedding: Optional[Any] = None

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

    def update_embedding(self, embedding_function: EmbeddingFunction):
        """
        Update the embedding of the instruction using the provided embedding function.

        Args:
            embedding_function (EmbeddingFunction): A function that takes
             a list of strings and returns an embedding.
        """
        self.embedding = embedding_function([self.text])


class SampleCollection(BaseModel):
    samples: list[Sample]

    @classmethod
    def from_csv(cls, csv_file: str) -> 'SampleCollection':
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
        samples = [Sample(**row) for _, row in df.iterrows()]
        return cls(samples=samples)

    def __len__(self) -> int:
        """
        Get the number of samples.

        Returns:
            int: The number of samples.
        """
        return len(self.samples)

    def number_of_samples(self) -> int:
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

class InstructionCollection(BaseModel):
    instructions: list[Instruction]
    embeddings: list[Optional[Any]] = Field(default_factory=list)

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

    def update_embeddings(self, embedding_function: EmbeddingFunction):
        """
        Add embeddings to the instructions using the provided embedding function.

        Args:
            embedding_function (EmbeddingFunction): A function that takes
             a list of strings and returns an embedding.
        """
        instructions_text = [instruction.text for instruction in self.instructions]
        embeddings = embedding_function(instructions_text)

        for instruction, embedding in zip(self.instructions, embeddings):
            instruction.embedding = embedding
            self.embeddings.append(embedding)

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


class ClusterOutput(BaseModel):
    """
    Represents the cluster output for a sample instruction.
    """
    id: int
    name: str
    description: str
    count: int

class ClusterOutputs(BaseModel):
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
