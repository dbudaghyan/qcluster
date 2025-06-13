from typing import Callable, Any, Union, Literal

from qcluster.describer import ClusterDescription

EmbeddingType = Any
ClusterType = Any
EmbeddingFunctionType = Callable[[list[str]], list[EmbeddingType]]
ClusteringFunctionType = Callable[[list[EmbeddingType]], list[ClusterType]]

# Dissimilarity and similarity functions takes a list of strings or embeddings,
# and an integer (top_n), as input,
# returns a list of tuples containing the index and the distance or similarity score.
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

ActualType = Any
PredictedType = Any

PredictionPairType = tuple[ActualType, PredictedType]

IdToCategoryResultType = dict[int, PredictionPairType]
