# Configuration

QCluster is configured using environment variables. You can set these variables by creating a `.env` file in the root of the project. To get started, you can copy the example file:

```sh
cp .env.example .env
```

Then, you can modify the `.env` file to customize the pipeline's behavior.

## Available Variables

Here is a list of all the available environment variables and their meanings:

### `TOKENIZERS_PARALLELISM`

-   **Description:** This variable controls whether the tokenizers from the `transformers` library should run in parallel. Setting it to `false` can help avoid potential issues on some systems.
-   **Default:** `false`

### `OLLAMA_MODEL`

-   **Description:** The name of the Ollama model to use for generating cluster descriptions. This model should be a large language model that is capable of understanding and summarizing text.
-   **Default:** `qwen2.5:3b`

### `OLLAMA_REPORTING_MODEL`

-   **Description:** The name of the Ollama model to use for generating the final evaluation report. You can use a more powerful model for this task to get more insightful reports.
-   **Default:** `qwen2.5:3b`

### `OLLAMA_HOST`

-   **Description:** The URL of the Ollama server. If you are running Ollama on a different machine or port, you will need to change this variable.
-   **Default:** `http://localhost:11434`

### `SENTENCE_TRANSFORMERS_MODEL`

-   **Description:** The name of the sentence transformer model to use for creating embeddings. These embeddings are used to calculate the semantic similarity between questions. A list of available models can be found on the [Hugging Face Hub](https://huggingface.co/models?library=sentence-transformers).
-   **Default:** `all-mpnet-base-v2`

### `EVALUATION_RESULTS_DIR`

-   **Description:** The directory where the evaluation results will be saved. Each run of the pipeline will create a new subdirectory in this location.
-   **Default:** `../temp/evaluation_results`

### `EVALUATION_REPORT_PROMPT_TEMPLATE`

-   **Description:** The name of the Jinja2 template file to use for generating the final evaluation report. The template should be located in the `qcluster/templates` directory.
-   **Default:** `generate_report_prompt`

### `DESCRIPTION_PROMPT_TEMPLATE`

-   **Description:** The name of the Jinja2 template file to use for generating cluster descriptions. The template should be located in the `qcluster/templates` directory.
-   **Default:** `description_prompt_simple`
