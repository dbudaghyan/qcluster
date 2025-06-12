from typing import Literal

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
# TODO
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
    # id: int
    flags: str
    instruction: str
    category: CategoryType
    intent: str
    response: str


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
        samples = [Sample(**row) for _, row in df.iterrows()]
        return cls(samples=samples)

    def __len__(self) -> int:
        """
        Get the number of samples.

        Returns:
            int: The number of samples.
        """
        return len(self.samples)
