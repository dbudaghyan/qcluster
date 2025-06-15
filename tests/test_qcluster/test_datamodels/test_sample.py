import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from qcluster.custom_types import (ALL_CATEGORIES, ALL_FLAGS, ALL_INTENTS,
                                   ClusterDescription)
from qcluster.datamodels.sample import Sample, SampleCollection

# --- Tests for Sample Class ---


def test_sample_creation():
    """Tests the creation of a Sample object."""
    sample = Sample(
        id=1,
        flags="N",
        instruction="Test instruction",
        category="ACCOUNT",
        intent="create_account",
        response="Test response",
    )
    assert sample.id == 1
    assert sample.flags == "N"
    assert sample.instruction == "Test instruction"
    assert sample.category == "ACCOUNT"
    assert sample.intent == "create_account"
    assert sample.response == "Test response"
    assert sample.embedding is None


def test_sample_all_classes():
    """Tests the static methods that return all possible classes."""
    assert Sample.all_category_classes() == set(ALL_CATEGORIES)
    assert Sample.all_flag_classes() == set(ALL_FLAGS)
    assert Sample.all_intent_classes() == set(ALL_INTENTS)


def test_sample_embedding_shape():
    """Tests the embedding_shape property."""
    sample = Sample(
        id=1,
        flags="N",
        instruction="Test instruction",
        category="ACCOUNT",
        intent="create_account",
        response="Test response",
    )
    assert sample.embedding_shape is None

    sample.embedding = np.array([1, 2, 3])
    assert sample.embedding_shape == (3,)

    sample.embedding = np.array([[1, 2], [3, 4]])
    assert sample.embedding_shape == (2, 2)


def test_update_embedding():
    """Tests updating the embedding of a sample."""
    sample = Sample(
        id=1,
        flags="N",
        instruction="Test instruction",
        category="ACCOUNT",
        intent="create_account",
        response="Test response",
    )
    mock_embedding_function = lambda x: np.array([0.1, 0.2, 0.3])
    sample.update_embedding(mock_embedding_function)
    assert isinstance(sample.embedding, np.ndarray)
    np.testing.assert_array_equal(sample.embedding, np.array([0.1, 0.2, 0.3]))


def test_sample_repr_and_str():
    """Tests the string representation of a Sample."""
    sample = Sample(
        id=1,
        flags="N",
        instruction="Test instruction",
        category="ACCOUNT",
        intent="create_account",
        response="This is a very long response that should be truncated.",
    )
    repr_str = repr(sample)
    assert "Sample(id=1" in repr_str
    assert "flags='N'" in repr_str
    assert "instruction='Test instruction'" in repr_str
    assert "category='ACCOUNT'" in repr_str
    assert "intent='create_account'" in repr_str
    assert "response='This is a very long response that should...'," in repr_str
    assert "embedding_shape=NA" in repr_str
    assert str(sample) == repr(sample)

    sample.embedding = np.zeros((5,))
    repr_str_with_embedding = repr(sample)
    assert "embedding_shape=(5,)" in repr_str_with_embedding


# --- Tests for SampleCollection Class ---


def create_sample_list():
    """Helper function to create a list of samples for testing."""
    return [
        Sample(
            id=1,
            flags="N",
            instruction="instruction 1",
            category="ACCOUNT",
            intent="create_account",
            response="response 1",
        ),
        Sample(
            id=2,
            flags="P",
            instruction="instruction 2",
            category="PAYMENT",
            intent="payment_issue",
            response="response 2",
        ),
        Sample(
            id=3,
            flags="N",
            instruction="instruction 3",
            category="ACCOUNT",
            intent="delete_account",
            response="response 3",
        ),
    ]


def test_sample_collection_creation():
    """Tests the creation of a SampleCollection object."""
    samples = create_sample_list()
    collection = SampleCollection(samples=samples)
    assert len(collection) == 3
    assert collection.samples == samples


def test_sample_collection_all_classes():
    """Tests the static methods from the collection."""
    assert SampleCollection.all_category_classes() == set(ALL_CATEGORIES)
    assert SampleCollection.all_flag_classes() == set(ALL_FLAGS)
    assert SampleCollection.all_intent_classes() == set(ALL_INTENTS)


def test_is_a_category():
    """Tests the is_a_category method."""
    collection = SampleCollection(samples=create_sample_list())
    assert not collection.is_a_category()

    single_category_samples = [s for s in collection if s.category == "ACCOUNT"]
    single_category_collection = SampleCollection(samples=single_category_samples)
    assert single_category_collection.is_a_category()


def test_is_intent():
    """Tests the is_intent method."""
    collection = SampleCollection(samples=create_sample_list())
    assert not collection.is_intent()

    single_intent_samples = [s for s in collection if s.intent == "create_account"]
    single_intent_collection = SampleCollection(samples=single_intent_samples)
    assert single_intent_collection.is_intent()


def test_category_property():
    """Tests the category property."""
    collection = SampleCollection(samples=create_sample_list())
    try:
        _ = collection.category
        assert False, "ValueError was not raised for heterogeneous collection"
    except ValueError as e:
        assert "Samples do not belong to a single category" in str(e)

    single_category_samples = [s for s in collection if s.category == "ACCOUNT"]
    single_category_collection = SampleCollection(samples=single_category_samples)
    assert single_category_collection.category == "ACCOUNT"


def test_intent_property():
    """Tests the intent property."""
    collection = SampleCollection(samples=create_sample_list())
    try:
        _ = collection.intent
        assert False, "ValueError was not raised for heterogeneous collection"
    except ValueError as e:
        assert "Samples do not belong to a single intent" in str(e)

    single_intent_samples = [s for s in collection if s.intent == "create_account"]
    single_intent_collection = SampleCollection(samples=single_intent_samples)
    assert single_intent_collection.intent == "create_account"


def test_get_sample_by_id():
    """Tests retrieving a sample by its ID."""
    collection = SampleCollection(samples=create_sample_list())
    sample = collection.get_sample_by_id(2)
    assert sample is not None
    assert sample.id == 2
    assert sample.category == "PAYMENT"
    assert collection.get_sample_by_id(99) is None


def test_update_embeddings_collection():
    """Tests updating embeddings for the whole collection."""
    collection = SampleCollection(samples=create_sample_list())
    mock_embedding_function = lambda instructions: [
        np.array([0.1, 0.2]),
        np.array([0.3, 0.4]),
        np.array([0.5, 0.6]),
    ]
    collection.update_embeddings(mock_embedding_function)
    assert collection.samples[0].embedding is not None
    assert collection.samples[1].embedding is not None
    assert collection.samples[2].embedding is not None
    np.testing.assert_array_equal(collection.embeddings[0], np.array([0.1, 0.2]))
    np.testing.assert_array_equal(collection.embeddings[2], np.array([0.5, 0.6]))


def test_centroid():
    """Tests the centroid calculation."""
    samples = create_sample_list()[:2]
    samples[0].embedding = np.array([1.0, 2.0])
    samples[1].embedding = np.array([3.0, 4.0])
    collection = SampleCollection(samples=samples)
    centroid = collection.centroid
    assert centroid is not None
    np.testing.assert_array_equal(centroid, np.array([2.0, 3.0]))

    # Test with no embeddings, which should raise a TypeError
    collection_no_emb = SampleCollection(samples=create_sample_list())
    try:
        _ = collection_no_emb.centroid
        assert False, "TypeError was not raised for centroid with no embeddings"
    except TypeError:
        pass  # Expected


def test_collection_slicing_and_getitem():
    """Tests getting a slice or a single item from the collection."""
    collection = SampleCollection(samples=create_sample_list())

    # Test slicing
    sliced_collection = collection[0:2]
    assert isinstance(sliced_collection, SampleCollection)
    assert len(sliced_collection) == 2
    assert sliced_collection[0].id == 1
    assert sliced_collection[1].id == 2

    # Test getting a single item
    item = collection[1]
    assert isinstance(item, Sample)
    assert item.id == 2


def test_group_by_category():
    """Tests grouping samples by category."""
    collection = SampleCollection(samples=create_sample_list())
    grouped = collection.group_by_category()
    assert isinstance(grouped, dict)
    assert "ACCOUNT" in grouped
    assert "PAYMENT" in grouped
    assert len(grouped["ACCOUNT"]) == 2
    assert len(grouped["PAYMENT"]) == 1
    assert grouped["ACCOUNT"][0].category == "ACCOUNT"
    assert grouped["PAYMENT"][0].category == "PAYMENT"


def test_group_by_intent():
    """Tests grouping samples by intent."""
    collection = SampleCollection(samples=create_sample_list())
    grouped = collection.group_by_intent()
    assert isinstance(grouped, dict)
    assert "create_account" in grouped
    assert "payment_issue" in grouped
    assert "delete_account" in grouped
    assert len(grouped["create_account"]) == 1
    assert len(grouped["payment_issue"]) == 1
    assert len(grouped["delete_account"]) == 1


def test_filter_by_category():
    """Tests filtering samples by category."""
    collection = SampleCollection(samples=create_sample_list())

    # Filter by a category that exists
    account_collection = collection.filter_by_category("ACCOUNT")
    assert isinstance(account_collection, SampleCollection)
    assert len(account_collection) == 2
    assert all(s.category == "ACCOUNT" for s in account_collection)

    # Filter by a category that doesn't exist in the sample list
    shipping_collection = collection.filter_by_category("SHIPPING")
    assert isinstance(shipping_collection, SampleCollection)
    assert len(shipping_collection) == 0


def test_describe():
    """Tests the describe method."""
    collection = SampleCollection(samples=create_sample_list())

    # noinspection PyUnusedLocal
    def mock_description_function(doc: str):
        return ClusterDescription(title="Test Title", description="Test Description")

    output = collection.describe(mock_description_function)
    assert collection.title == "Test Title"
    assert collection.description == "Test Description"
    assert output.name == "Test Title"
    assert output.description == "Test Description"
    assert output.count == len(collection)

    empty_collection = SampleCollection(samples=[])
    try:
        empty_collection.describe(mock_description_function)
        assert False, "ValueError was not raised for empty collection"
    except ValueError as e:
        assert "No samples available for description" in str(e)


def test_description_embedding():
    """Tests getting the embedding of the description."""
    collection = SampleCollection(samples=create_sample_list())
    mock_embedding_function = lambda x: [np.array([0.5, 0.6, 0.7])]

    try:
        collection.description_embedding(mock_embedding_function)
        assert False, "ValueError was not raised for missing description"
    except ValueError as e:
        assert "No description available for embedding" in str(e)

    collection.description = "This is a test description."
    embedding = collection.description_embedding(mock_embedding_function)
    np.testing.assert_array_equal(embedding, np.array([0.5, 0.6, 0.7]))


def test_from_csv():
    """Tests creating a SampleCollection from a CSV file."""
    with tempfile.TemporaryDirectory() as tmpdir:
        csv_file = Path(tmpdir) / "test_data.csv"
        data = {
            "flags": ["N", "P"],
            "instruction": ["Instruction A", "Instruction B"],
            "category": ["ACCOUNT", "PAYMENT"],
            "intent": ["create_account", "payment_issue"],
            "response": ["Response A", "Response B"],
        }
        df = pd.DataFrame(data)
        df.to_csv(csv_file, index=False)

        collection = SampleCollection.from_csv(csv_file)
        assert isinstance(collection, SampleCollection)
        assert len(collection) == 2

        # The method shuffles the data, so we check by content
        instructions = {s.instruction for s in collection}
        assert instructions == {"Instruction A", "Instruction B"}

        sample_a = next(s for s in collection if s.instruction == "Instruction A")
        assert sample_a.category == "ACCOUNT"
        assert sample_a.id is not None  # check id is assigned


def test_number_of_samples():
    """Tests the number_of_samples method."""
    collection = SampleCollection(samples=create_sample_list())
    assert collection.number_of_samples() == 3

    empty_collection = SampleCollection(samples=[])
    assert empty_collection.number_of_samples() == 0


def test_sample_collection_repr_and_str():
    """Tests the string representation of a SampleCollection."""
    collection = SampleCollection(samples=create_sample_list()[:2])
    repr_str = repr(collection)

    assert "SampleCollection(" in repr_str
    assert "num_samples=2" in repr_str
    assert "Sample(id=1" in repr_str
    assert "Sample(id=2" in repr_str
    assert str(collection) == repr_str

    empty_collection = SampleCollection(samples=[])
    empty_repr_str = repr(empty_collection)
    assert "num_samples=0" in empty_repr_str
    assert "samples=[\n\n" in empty_repr_str
    assert str(empty_collection) == empty_repr_str
