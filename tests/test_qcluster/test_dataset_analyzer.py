from qcluster.dataset_analyzer import (create_category_intent_hierarchy,
                                       create_category_intent_tuples,
                                       identify_unique_categories,
                                       identify_unique_flags,
                                       identify_unique_intents)


# Mock objects to simulate Sample and SampleCollection for testing
class Sample:
    def __init__(self, predicted_category, intent, flags=None):
        self.predicted_category = predicted_category
        self.intent = intent
        self.flags = flags if flags is not None else []


class SampleCollection:
    def __init__(self, samples):
        self._samples = samples

    def __iter__(self):
        return iter(self._samples)


# Tests for identify_unique_categories
def test_identify_unique_categories_empty():
    samples = SampleCollection([])
    assert identify_unique_categories(samples) == set()


def test_identify_unique_categories_single_sample():
    samples = SampleCollection([Sample("cat1", "intent1")])
    assert identify_unique_categories(samples) == {"cat1"}


def test_identify_unique_categories_multiple_unique():
    samples = SampleCollection(
        [
            Sample("cat1", "intent1"),
            Sample("cat2", "intent2"),
        ]
    )
    assert identify_unique_categories(samples) == {"cat1", "cat2"}


def test_identify_unique_categories_with_duplicates():
    samples = SampleCollection(
        [
            Sample("cat1", "intent1"),
            Sample("cat2", "intent2"),
            Sample("cat1", "intent3"),
        ]
    )
    assert identify_unique_categories(samples) == {"cat1", "cat2"}


# Tests for identify_unique_intents
def test_identify_unique_intents_empty():
    samples = SampleCollection([])
    assert identify_unique_intents(samples) == set()


def test_identify_unique_intents_single_sample():
    samples = SampleCollection([Sample("cat1", "intent1")])
    assert identify_unique_intents(samples) == {"intent1"}


def test_identify_unique_intents_multiple_unique():
    samples = SampleCollection(
        [
            Sample("cat1", "intent1"),
            Sample("cat2", "intent2"),
        ]
    )
    assert identify_unique_intents(samples) == {"intent1", "intent2"}


def test_identify_unique_intents_with_duplicates():
    samples = SampleCollection(
        [
            Sample("cat1", "intent1"),
            Sample("cat2", "intent2"),
            Sample("cat3", "intent1"),
        ]
    )
    assert identify_unique_intents(samples) == {"intent1", "intent2"}


# Tests for create_category_intent_tuples
def test_create_category_intent_tuples_empty():
    samples = SampleCollection([])
    assert create_category_intent_tuples(samples) == set()


def test_create_category_intent_tuples_single_sample():
    samples = SampleCollection([Sample("cat1", "intent1")])
    assert create_category_intent_tuples(samples) == {("cat1", "intent1")}


def test_create_category_intent_tuples_multiple_samples():
    samples = SampleCollection(
        [
            Sample("cat1", "intent1"),
            Sample("cat2", "intent2"),
        ]
    )
    assert create_category_intent_tuples(samples) == {
        ("cat1", "intent1"),
        ("cat2", "intent2"),
    }


def test_create_category_intent_tuples_with_duplicates():
    samples = SampleCollection(
        [
            Sample("cat1", "intent1"),
            Sample("cat2", "intent2"),
            Sample("cat1", "intent1"),  # Duplicate pair
            Sample("cat1", "intent3"),
        ]
    )
    expected = {("cat1", "intent1"), ("cat2", "intent2"), ("cat1", "intent3")}
    assert create_category_intent_tuples(samples) == expected


# Tests for create_category_intent_hierarchy
def test_create_category_intent_hierarchy_empty():
    assert create_category_intent_hierarchy(set()) == {}


def test_create_category_intent_hierarchy_single_tuple():
    tuples = {("cat1", "intent1")}
    assert create_category_intent_hierarchy(tuples) == {"cat1": {"intent1"}}


def test_create_category_intent_hierarchy_multiple_tuples():
    tuples = {
        ("cat1", "intent1"),
        ("cat2", "intent2"),
        ("cat1", "intent3"),
    }
    expected = {
        "cat1": {"intent1", "intent3"},
        "cat2": {"intent2"},
    }
    assert create_category_intent_hierarchy(tuples) == expected


def test_create_category_intent_hierarchy_complex():
    tuples = {
        ("cat1", "intentA"),
        ("cat1", "intentB"),
        ("cat2", "intentA"),
        ("cat3", "intentC"),
    }
    expected = {
        "cat1": {"intentA", "intentB"},
        "cat2": {"intentA"},
        "cat3": {"intentC"},
    }
    assert create_category_intent_hierarchy(tuples) == expected


# Tests for identify_unique_flags
def test_identify_unique_flags_empty():
    samples = SampleCollection([])
    assert identify_unique_flags(samples) == set()


def test_identify_unique_flags_no_flags():
    samples = SampleCollection(
        [
            Sample("cat1", "intent1", flags=[]),
            Sample("cat2", "intent2", flags=[]),
        ]
    )
    assert identify_unique_flags(samples) == set()


def test_identify_unique_flags_single_sample_single_flag():
    samples = SampleCollection([Sample("cat1", "intent1", flags=["flagA"])])
    assert identify_unique_flags(samples) == {"flagA"}


def test_identify_unique_flags_single_sample_multiple_flags():
    samples = SampleCollection([Sample("cat1", "intent1", flags=["flagA", "flagB"])])
    assert identify_unique_flags(samples) == {"flagA", "flagB"}


def test_identify_unique_flags_multiple_samples_with_overlap():
    samples = SampleCollection(
        [
            Sample("cat1", "intent1", flags=["flagA", "flagB"]),
            Sample("cat2", "intent2", flags=["flagB", "flagC"]),
            Sample("cat3", "intent3", flags=[]),
        ]
    )
    assert identify_unique_flags(samples) == {"flagA", "flagB", "flagC"}
