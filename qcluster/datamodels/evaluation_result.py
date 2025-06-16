import os
from functools import cached_property
from os import PathLike
from pathlib import Path
from typing import Optional

from loguru import logger
from pycm import ConfusionMatrix
from pydantic import BaseModel

from qcluster.custom_types import ClusteringReport
from qcluster.datamodels.filesystem import CSVFile, File, PYCMObject, slugify
from qcluster.llm.reporter import create_report


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
    summary_statistics: File
    final_report: Optional[File]

    @staticmethod
    def final_report_filename() -> str:
        return "final_report.md"

    @classmethod
    def from_folder_path(cls, path: PathLike) -> "EvaluationResult":
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
            stats_pycm=PYCMObject.from_path(path / "pycm.zip"),
            clusters=File.from_path(path / "clusters.json"),
            summary_statistics=File.from_path(path / "stats.pycm"),
            final_report=(
                File.from_path(path / cls.final_report_filename())
                if (path / cls.final_report_filename()).exists()
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
            "summary_statistics": self.summary_statistics.content,
        }

    def add_final_report(self) -> ClusteringReport:
        """
        Generates a final report and adds it to the EvaluationResult instance.

        Returns:
            ClusteringReport: The generated final report.
        """
        logger.info("Generating final report...")
        report: str = create_report(
            template_name=os.environ["EVALUATION_REPORT_PROMPT_TEMPLATE"],
            evaluation_result=self,
        )
        self.final_report = File(
            name=self.final_report_filename(),
            path=Path(self.path) / self.final_report_filename(),
        )
        with open(self.final_report.path, "w") as f:
            f.write(report)
        name = slugify(f"final_report_{self.name}")
        logger.info(f"Final report saved as {self.final_report.path}")
        return ClusteringReport(title=name, report=report)
