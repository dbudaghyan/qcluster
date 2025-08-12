# Getting Started with QCluster

This guide will walk you through the process of setting up your local environment and running the QCluster pipeline for the first time. By the end of this guide, you will have a working installation of QCluster and will have generated your first set of clustering results.

## Prerequisites

Before you begin, ensure you have the following software installed on your system:

- **Python (3.12+):** QCluster is built on Python 3.12. You can download the latest version from the [official Python website](https://www.python.org/downloads/).
- **uv:** We recommend using `uv` for package management. It's a fast and reliable package installer and resolver. You can find installation instructions in the [official `uv` documentation](https://github.com/astral-sh/uv).
- **Ollama:** QCluster uses [Ollama](https://ollama.com/) to run large language models locally. You will need to install Ollama and pull the required models.

This project was tested on `macOS` with `Apple Silicon`, but it should work on other systems as well.

## Installation and Setup

Follow these steps to get your local copy of QCluster up and running:

1.  **Clone the Repository:**

    First, clone the QCluster repository to your local machine using `git`:

    ```sh
    git clone https://github.com/dbudaghyan/qcluster.git
    cd qcluster
    ```

2.  **Set Up Environment Variables:**

    QCluster uses environment variables to configure the pipeline. You can create a `.env` file by copying the provided example:

    ```sh
    cp .env.example .env
    ```

    You can modify the `.env` file to change the default settings. For more information on the available options, see the [Configuration](configuration.md) guide.

3.  **Install and Set Up Ollama:**

    If you don't have Ollama installed, you can do so using Homebrew (on macOS) or by downloading it from the [official website](https://ollama.com/download).

    Using Homebrew:

    ```sh
    brew install ollama
    ```

    Once Ollama is installed, you need to pull the language model used by QCluster. By default, QCluster uses `qwen2.5:3b`.

    ```sh
    ollama pull qwen2.5:3b
    ```

    If you have configured other models in your `.env` file, make sure to pull them as well.

    After pulling the models, you need to start the Ollama server. In a new terminal window, run:

    ```sh
    ollama serve
    ```

    Keep this terminal window open while you are using QCluster.

4.  **Install Python Dependencies:**

    Now, you can install the required Python packages using `uv`:

    ```sh
    uv sync
    ```

    This command will create a virtual environment and install all the necessary dependencies.

## Running the Pipeline

You can run the clustering pipeline in two ways: as a Python script or using a Jupyter Notebook.

### Option 1: As a Python Script

The most straightforward way to run the pipeline is to execute the `qcluster.pipeline` module:

```bash
uv run qcluster.pipeline
```

This will run the entire pipeline, from loading the data to generating the evaluation report.

### Option 2: Using a Jupyter Notebook

For a more interactive experience, you can use the provided Jupyter Notebook. First, add the project root to your Python path:

```bash
export PYTHONPATH=$(pwd)
```

Then, launch Jupyter Lab:

```bash
uv run --with jupyter jupyter-lab
```

This will open Jupyter Lab in your browser. Navigate to `notebooks/pipeline.ipynb` and run the cells in the notebook.

## What's Next?

Now that you have successfully run the QCluster pipeline, you can explore the following topics to learn more:

- [**Configuration**](configuration.md): Learn how to customize the pipeline's behavior using environment variables.
- [**The Pipeline**](pipeline.md): Get a high-level overview of the clustering pipeline and learn how to customize it.
- [**Evaluation Results**](evaluation.md): Understand the structure of the evaluation results and how to interpret them.
- [**API Reference**](api_reference.md): Dive into the technical details of the `qcluster` library.
