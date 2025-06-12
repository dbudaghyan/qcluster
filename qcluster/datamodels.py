from typing import Literal, Union

import pandas as pd
from pydantic import BaseModel

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
    instruction: str

    @classmethod
    def from_sample(cls, sample: Sample) -> 'Instruction':
        """
        Create an Instruction object from a Sample object.

        Args:
            sample (Sample): A Sample object.

        Returns:
            Instruction: An Instruction object with the instruction from the Sample.
        """
        return cls(id=sample.id, instruction=sample.instruction)


class Samples(BaseModel):
    samples: list[Sample]

    @classmethod
    def from_csv(cls, csv_file: str) -> 'Samples':
        """
        Create a `Samples` object from a CSV file.

        Args:
            csv_file (str): Path to the CSV file.

        Returns:
            Samples: A Samples object containing the data from the CSV file.
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

    def __getitem__(self, index: int) -> Union[Sample, 'Samples']:
        """
        Get a sample or a slice of samples by index.

        Args:
            index (int): Index of the sample or slice.

        Returns:
            Sample or Samples: A single Sample object or a Samples object if a slice is requested.
        """
        if isinstance(index, slice):
            return Samples(samples=self.samples[index])
        return self.samples[index]

class Instructions(BaseModel):
    instructions: list[Instruction]

    @classmethod
    def from_samples(cls, samples: Samples) -> 'Instructions':
        """
        Create an Instructions object from a list of Sample objects.

        Args:
            samples (list[Sample]): A list of Sample objects.

        Returns:
            Instructions: An Instructions object containing the instructions
             from the Samples.
        """
        instructions = [Instruction.from_sample(sample) for sample in samples.samples]
        return cls(instructions=instructions)

    def to_list_of_strings(self) -> list[str]:
        """
        Convert the instructions to a list of strings.

        Returns:
            list[str]: A list of instruction strings.
        """
        return [instruction.instruction for instruction in self.instructions]


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
