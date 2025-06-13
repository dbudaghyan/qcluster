from typing import Callable, Any, Union, Literal, get_args, Optional

from pydantic import BaseModel


class ClusterDescription(BaseModel):
  """ Used only by the LLM"""
  title: str
  description: str


EmbeddingType = Any
ClusterType = Any
EmbeddingFunctionType = Callable[[list[str]], EmbeddingType]
ClusteringFunctionType = Callable[[list[EmbeddingType]], list[ClusterType]]


DissimilarityFunctionType = Callable[[list[Union[str, EmbeddingType]], int],
                                        list[tuple[int, Union[str, EmbeddingType]]]]
SimilarityFunctionType = Callable[[Union[str, EmbeddingType],
                                   list[Union[str, EmbeddingType]],
                                   int],
                                        list[tuple[int, float]]]
DescriptionFunctionType = Callable[[str], ClusterDescription]

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
    'SUBSCRIPTION',
    'UNKNOWN',
]
FlagType = Literal['N', 'B', 'P', 'Z', 'V', 'E', 'M', 'S', 'C', 'W', 'Q', 'L', 'K', 'I',
                    '?']
IntentType = Literal[
    'delivery_options', 'create_account', 'change_shipping_address',
    'place_order', 'contact_human_agent', 'complaint',
    'newsletter_subscription', 'delete_account',
    'registration_problems', 'check_cancellation_fee', 'switch_account',
    'change_order', 'check_payment_methods', 'track_order',
    'check_refund_policy', 'track_refund', 'check_invoice',
    'payment_issue', 'contact_customer_service', 'cancel_order',
    'delivery_period', 'set_up_shipping_address', 'get_invoice',
    'recover_password', 'edit_account', 'get_refund', 'review',
    'UNKNOWN'
]

ALL_CATEGORIES = get_args(CategoryType)
ALL_FLAGS = get_args(FlagType)
ALL_INTENTS = get_args(IntentType)

ActualType = Any
PredictedType = Any

PredictionPairType = tuple[ActualType, PredictedType]

IdToCategoryResultType = dict[int, PredictionPairType]

def category_to_idx(category: CategoryType) -> Optional[int]:
    """Convert a category to its index."""
    return ALL_CATEGORIES.index(category) if category in ALL_CATEGORIES else None

def flag_to_idx(flag: FlagType) -> Optional[int]:
    """Convert a flag to its index."""
    return ALL_FLAGS.index(flag) if flag in ALL_FLAGS else None

def intent_to_idx(intent: IntentType) -> Optional[int]:
    """Convert an intent to its index."""
    return ALL_INTENTS.index(intent) if intent in ALL_INTENTS else None
