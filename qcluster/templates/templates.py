import os
from pathlib import Path

import jinja2

TEMPLATE_PATH = os.path.join(
    Path(__file__).parent,
    "templates",
)


def read_prompt_template(template_name: str) -> jinja2.Template:
    """
    Reads a Jinja2 template from the `templates` directory.
    """
    loader = jinja2.FileSystemLoader(os.path.dirname(TEMPLATE_PATH))
    env = jinja2.Environment(loader=loader, autoescape=True)
    return env.get_template(f"{template_name}.j2")
