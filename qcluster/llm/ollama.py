import os
from typing import Optional, Any, Union, Type

from loguru import logger
from ollama import Client
from pydantic import BaseModel

from qcluster.templates.templates import read_prompt_template


def query_llm(
    template_name: str,
    data: dict[str, Any],
    model: str,
    max_tokens: int = 2048,
    output_model: Optional[Type[BaseModel]] = None,
) -> Union[BaseModel, str]:
    if issubclass(output_model, BaseModel):
        output_schema = output_model.model_json_schema()
        has_output_model = True
    elif isinstance(output_model, str) or output_model is None:
        output_schema = None
        has_output_model = False
    else:
        logger.warning(
            f"Unsupported output_model type: `{type(output_model)}`. "
            f"Expected `pydantic.BaseModel` or `str`,"
            f" got `{type(output_model)}`."
        )
        output_schema = None
        has_output_model = False
    template = read_prompt_template(template_name)
    client = Client(host=os.environ["OLLAMA_HOST"])
    prompt = template.render(**data)
    response = client.chat(
        messages=[{"role": "user", "content": prompt}],
        model=model,
        format=output_schema,
        options={"temperature": 0.0, "max_tokens": max_tokens},
    )
    if has_output_model:
        return output_model.model_validate_json(response.message.content)
    else:
        return response.message.content
