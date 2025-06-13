import os
from typing import Optional

from ollama import Client

from qcluster.custom_types import ClusterDescription
from qcluster.preload import LLM_TEMPLATE


def get_description(document: str, limit: Optional[int] = None) -> ClusterDescription:
  model = os.environ['OLLAMA_MODEL']
  client = Client(host=os.environ['OLLAMA_HOST'])
  if limit:
    document = document[:limit]
  prompt = LLM_TEMPLATE.render(document=document)
  response = client.chat(
    messages=[{'role': 'user', 'content': prompt}],
    model=model,
    format=ClusterDescription.model_json_schema(),
    options={'temperature': 0.0},
  )

  return ClusterDescription.model_validate_json(response.message.content)
