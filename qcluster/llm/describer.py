import os
from typing import Optional


from qcluster.custom_types import ClusterDescription
from qcluster.llm.ollama import query_llm


def get_description(
    document: str, template_name: str, limit: Optional[int] = 5000
) -> ClusterDescription:
    model = os.environ["OLLAMA_MODEL"]
    if limit:
        document = document[:limit]
    llm_output = query_llm(
        template_name=template_name,
        data={"document": document},
        model=model,
        output_model=ClusterDescription,
    )
    assert isinstance(llm_output, ClusterDescription)
    return llm_output
