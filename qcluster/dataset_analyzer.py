def identify_unique_categories(samples: "SampleCollection") -> set[str]:
    """
    Identify unique categories from the SampleCollection object.

    Args:
        samples (SampleCollection): A SampleCollection object.

    Returns:
        set: A set of unique categories.
    """
    return {sample.predicted_category for sample in samples}


def identify_unique_intents(samples: "SampleCollection") -> set[str]:
    """
    Identify unique intents from the SampleCollection object.

    Args:
        samples (SampleCollection): A SampleCollection object.

    Returns:
        set: A set of unique intents.
    """
    return {sample.intent for sample in samples}


def create_category_intent_tuples(samples: "SampleCollection") -> set[tuple[str, str]]:
    """
    Create a set of tuples containing unique (category, intent) pairs
     from the SampleCollection object.

    Args:
        samples (SampleCollection): A SampleCollection object.

    Returns:
        set: A set of tuples with unique (category, intent) pairs.
    """
    return {(sample.predicted_category, sample.intent) for sample in samples}


def create_category_intent_hierarchy(
    category_intent_tuples: set[tuple[str, str]],
) -> dict[str, set[str]]:
    """
    Create a hierarchy of categories and their associated intents.
    Args:
        category_intent_tuples (set[tuple[str, str]]): A set of tuples
         where each tuple contains a category and intent.
    Returns:
        dict: A dictionary where keys are categories and values are sets of intents
         associated with those categories.
    """
    hierarchy = {}
    for category, intent in category_intent_tuples:
        if category not in hierarchy:
            hierarchy[category] = set()
        hierarchy[category].add(intent)
    return hierarchy


def identify_unique_flags(samples: "SampleCollection") -> set[str]:
    """
    Identify unique flags from the SampleCollection object.

    Args:
        samples (SampleCollection): A SampleCollection object.

    Returns:
        set: A set of unique flags.
    """
    return {flag for sample in samples for flag in sample.flags}


if __name__ == "__main__":
    from qcluster.datamodels.sample import SampleCollection
    from qcluster import ROOT_DIR

    from loguru import logger

    csv_file_ = (
        ROOT_DIR.parent
        / "data"
        / "Bitext_Sample_Customer_Support_Training_Dataset_27K_responses-v11.csv"
    )
    samples_ = SampleCollection.from_csv(csv_file_)
    unique_categories_ = identify_unique_categories(samples_)
    logger.info(f"Unique categories: {unique_categories_}")
    unique_intents_ = identify_unique_intents(samples_)
    logger.info(f"Unique intents: {unique_intents_}")
    unique_flags_ = identify_unique_flags(samples_)
    logger.info(f"Unique flags: {unique_flags_}")
    category_intent_tuples_ = create_category_intent_tuples(samples_)
    category_intent_hierarchy_ = create_category_intent_hierarchy(
        category_intent_tuples_
    )
    logger.info(f"Category-Intent Hierarchy: {category_intent_hierarchy_}")
