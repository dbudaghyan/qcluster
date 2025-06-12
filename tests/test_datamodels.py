import pandas as pd

from qcluster import ROOT_DIR
from qcluster.datamodels import (
    Samples,
    Sample
)


def test_samples_from_csv():
    csv_file_path = (
        ROOT_DIR.parent
        / "data"
        / "Bitext_Sample_Customer_Support_Training_Dataset_27K_responses-v11.csv"
    )
    rows = pd.read_csv(csv_file_path, dtype=str).shape[0]
    samples = Samples.from_csv(csv_file_path)
    assert isinstance(samples, Samples)
    assert len(samples.samples) > 0
    assert isinstance(samples.samples[0], Sample)
    n_samples = len(samples.samples)
    assert rows == n_samples
