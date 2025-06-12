import pandas as pd

from qcluster import ROOT_DIR
from qcluster.datamodels import (
    SampleCollection,
    Sample
)


def test_samples_from_csv():
    csv_file_path = (
        ROOT_DIR.parent
        / "data"
        / "Bitext_Sample_Customer_Support_Training_Dataset_27K_responses-v11.csv"
    )
    rows = pd.read_csv(csv_file_path, dtype=str).shape[0]
    samples = SampleCollection.from_csv(csv_file_path)

    assert isinstance(samples, SampleCollection)
    assert isinstance(samples[:10], SampleCollection)
    assert isinstance(samples[0], Sample)
    n_samples = samples.number_of_samples()
    assert rows == n_samples
