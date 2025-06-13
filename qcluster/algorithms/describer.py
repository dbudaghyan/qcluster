import os

from ollama import Client

from qcluster.custom_types import ClusterDescription


def get_description(document: str, limit: int = 2000) -> ClusterDescription:
  model = os.environ['OLLAMA_MODEL']
  client = Client(host=os.environ['OLLAMA_HOST'])
  if limit:
    document = document[:limit]
  response = client.chat(
    messages=[
      {
        'role': 'user',
        'content':
          'Provide a description'
          f' and a short title for the following document:\n\n{document}',
      }
    ],
    model=model,
    format=ClusterDescription.model_json_schema(),
    options={'temperature': 0.0},
  )

  return ClusterDescription.model_validate_json(response.message.content)
