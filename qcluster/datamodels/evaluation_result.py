import os
from functools import cached_property
from os import PathLike
from pathlib import Path
from typing import Optional

from pycm import ConfusionMatrix
from pydantic import BaseModel

from qcluster.custom_types import ClusteringReport
from qcluster.llm.reporter import create_report
from qcluster.datamodels.filesystem import File, slugify, CSVFile, PYCMObject


class EvaluationResult(BaseModel):
    name: str
    path: PathLike
    cluster_to_class_scores_str: CSVFile
    git_diff: File
    entrypoint: File
    results_html: File
    stats_matrix: File
    stats_csv: CSVFile
    stats_pycm: PYCMObject
    clusters: File
    final_report: Optional[File]

    @staticmethod
    def from_folder_path(path: PathLike) -> "EvaluationResult":
        """
        Reads the report files from a folder and returns a Report instance.

        Args:
            path (PathLike): The path to the folder containing the report files.

        Returns:
            EvaluationResult: An instance of Report with all fields populated.
        """
        path = Path(path)
        return EvaluationResult(
            name=path.name,
            path=path,
            cluster_to_class_scores_str=CSVFile.from_path(
                path / "cluster_to_class_scores.csv"
            ),
            git_diff=File.from_path(path / "git_diff.txt"),
            entrypoint=File.from_path(path / "entrypoint.py"),
            results_html=File.from_path(path / "results.html"),
            stats_matrix=CSVFile.from_path(path / "stats_matrix.csv"),
            stats_csv=CSVFile.from_path(path / "stats.csv"),
            stats_pycm=PYCMObject.from_path(path / "pycm.obj"),
            clusters= File.from_path(path / "clusters.json"),
            final_report=(
                File.from_path(path / "final_report.md")
                if (path / "final_report.md").exists()
                else None
            ),
        )

    @cached_property
    def cm(self) -> ConfusionMatrix:
        """
        Returns the confusion matrix object from the stats_pycm file.

        Returns:
            PYCMObject: The confusion matrix object.
        """
        return self.stats_pycm.pycm

    @property
    def num_samples(self) -> int:
        return self.stats_matrix.df.shape[0]

    @property
    def labels(self):
        return self.stats_csv.df.columns.tolist()

    @property
    def clustering_summary_html(self):
        return self.results_html.content

    @property
    def additional_metrics(self):
        return str(self.cluster_to_class_scores_str.df.to_dict(orient="records"))

    @property
    def cluster_json(self):
        return self.clusters.content

    def to_template_args(self) -> dict:
        """
        Converts the EvaluationResult instance to a dictionary suitable for templating.

        Returns:
            dict: A dictionary with all fields of the EvaluationResult instance.
        """
        return {
            "num_samples": self.num_samples,
            "labels": self.labels,
            "clustering_summary_html": self.clustering_summary_html,
            "additional_metrics": self.additional_metrics,
            "cluster_json": self.cluster_json,
        }

    def add_final_report(self) -> ClusteringReport:
        """
        Generates a final report and adds it to the EvaluationResult instance.

        Returns:
            ClusteringReport: The generated final report.
        """
        report: ClusteringReport = create_report(
            template_name=os.environ["EVALUATION_REPORT_PROMPT_TEMPLATE"],
            evaluation_result=self,
        )
        self.final_report = File(
            name="final_report.md", path=Path(self.path) / "final_report.md"
        )
        with open(self.final_report.path, "w") as f:
            f.write(report.report)
        name = slugify(self.title)
        return ClusteringReport(title=name, report=report.content)
