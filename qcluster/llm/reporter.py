import os


# from qcluster.custom_types import ClusteringReport
from qcluster.llm.ollama import query_llm


def create_report(
    template_name: str,
    evaluation_result: "EvaluationResult",
) -> str:
    model = os.environ["OLLAMA_REPORTING_MODEL"]
    llm_output = query_llm(
        template_name=template_name,
        data=evaluation_result.to_template_args(),
        model=model,
        # output_model=ClusteringReport,
        # max_tokens=25000,
    )
    # assert isinstance(
    #     llm_output, ClusteringReport
    # ), f"Expected ClusteringReport, got {type(llm_output)}"
    return llm_output


if __name__ == "__main__":
    # Import for pycharm type checking
    from qcluster.datamodels.evaluation_result import EvaluationResult
