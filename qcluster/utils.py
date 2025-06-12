import pandas as pd

from qcluster import ROOT_DIR
from qcluster.datamodels import Samples, Sample


def csv_to_samples(csv_file: str) -> Samples:
    """
    Convert a CSV file to a Samples object.

    Args:
        csv_file (str): Path to the CSV file.

    Returns:
        Samples: A Samples object containing the data from the CSV file.
    """
    df = pd.read_csv(csv_file, dtype=str)
    df.index.name = "id"
    samples = [Sample(**row) for _, row in df.iterrows()]
    return Samples(samples=samples)

def identify_unique_categories(samples: Samples) -> set[str]:
    """
    Identify unique categories from the Samples object.

    Args:
        samples (Samples): A Samples object.

    Returns:
        set: A set of unique categories.
    """
    return {sample.category for sample in samples.samples}


def identify_unique_intents(samples: Samples) -> set[str]:
    """
    Identify unique intents from the Samples object.

    Args:
        samples (Samples): A Samples object.

    Returns:
        set: A set of unique intents.
    """
    return {sample.intent for sample in samples.samples}


def identify_unique_flags(samples: Samples) -> set[str]:
    """
    Identify unique flags from the Samples object.

    Args:
        samples (Samples): A Samples object.

    Returns:
        set: A set of unique flags.
    """
    return {flag for sample in samples.samples for flag in sample.flags}


if __name__ == '__main__':
    from loguru import logger

    csv_file_ = (
            ROOT_DIR.parent
            / "data"
            / "Bitext_Sample_Customer_Support_Training_Dataset_27K_responses-v11.csv"
    )
    samples_ = Samples.from_csv(csv_file_)
    unique_categories_ = identify_unique_categories(samples_)
    logger.info(f"Unique categories: {unique_categories_}")
    unique_intents_ = identify_unique_intents(samples_)
    logger.info(f"Unique intents: {unique_intents_}")
    unique_flags_ = identify_unique_flags(samples_)
    logger.info(f"Unique flags: {unique_flags_}")
