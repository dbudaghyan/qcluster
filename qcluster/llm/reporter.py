import os
from typing import Optional

from ollama import Client

from qcluster.custom_types import ClusterDescription
from qcluster.templates.templates import read_prompt_template


def get_report(
    document: str, template_name: str, limit: Optional[int] = 5000
) -> ClusterDescription:
    template = read_prompt_template(template_name)
    model = os.environ["OLLAMA_MODEL"]
    client = Client(host=os.environ["OLLAMA_HOST"])
    if limit:
        document = document[:limit]
    prompt = template.render(document=document)
    response = client.chat(
        messages=[{"role": "user", "content": prompt}],
        model=model,
        format=ClusterDescription.model_json_schema(),
        options={"temperature": 0.0, "max_tokens": 2048},
    )

    return ClusterDescription.model_validate_json(response.message.content)
