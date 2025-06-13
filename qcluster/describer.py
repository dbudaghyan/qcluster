import os

import ollama
from pydantic import BaseModel


class ClusterDescription(BaseModel):
  """ Used only by the LLM"""
  title: str
  description: str

def get_description(document: str):
  response = ollama.chat(
    messages=[
      {
        'role': 'user',
        'content':
          'Provide a short description'
          f' and a short title for the following document:\n\n{document}',
      }
    ],
    model=os.environ['OLLAMA_MODEL'],
    format=ClusterDescription.model_json_schema(),
  )

  return ClusterDescription.model_validate_json(response.message.content)
