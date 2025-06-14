<div align="center">

  <h1 align="center">QCluster</h1>
  <p align="center">
    A Python library for clustering customer questions using large language models.
    <br />
    <a href="#about-the-project"><strong>Explore the docs »</strong></a>
    <br />
    <br />
    <a href="https://github.com/dbudaghyan/qcluster/issues">Report Bug</a>
    ·
    <a href="https://github.com/dbudaghyan/qcluster/issues">Request Feature</a>
  </p>

  <!-- Project Shields -->
  <p align="center">
    <a href="https://github.com/dbudaghyan/qcluster/stargazers"><img src="https://img.shields.io/github/stars/dbudaghyan/qcluster.svg?style=for-the-badge" alt="Stars"></a>
    <a href="https://github.com/dbudaghyan/qcluster/network/members"><img src="https://img.shields.io/github/forks/dbudaghyan/qcluster.svg?style=for-the-badge" alt="Forks"></a>
    <a href="https://github.com/dbudaghyan/qcluster/issues"><img src="https://img.shields.io/github/issues/dbudaghyan/qcluster.svg?style=for-the-badge" alt="Issues"></a>
    <a href="https://github.com/dbudaghyan/qcluster/blob/main/LICENSE"><img src="https://img.shields.io/github/license/dbudaghyan/qcluster.svg?style=for-the-badge" alt="License"></a>
    <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/Python-3.12+-blue?style=for-the-badge&logo=python" alt="Python Version"></a>
  </p>
</div>

<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#contributing">Contributing</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
  </ol>
</details>

<!-- ABOUT THE PROJECT -->
## About The Project

QCluster is a powerful Python library designed to help you make sense of large volumes of customer feedback. By leveraging the power of Large Language Models (LLMs), QCluster can automatically group similar customer questions, allowing you to identify trends, pain points, and frequently asked questions with ease.

This project provides a complete pipeline for:
*   Extracting customer questions from your data sources.
*   Generating embeddings for each question.
*   Clustering the questions based on their semantic similarity.
*   Evaluating the quality of the clustering results.
*   Generating insightful reports.

<!-- GETTING STARTED -->
## Getting Started

Follow these simple steps to get your local copy of QCluster up and running.

### Prerequisites

This project was tested on `macOS` with `Apple Silicon`, but it should work on other systems as well.

*   **Python** `3.12+`
*   **uv**: A fast Python package installer and resolver.
*   **ollama**: Run large language models locally.
    *   You will also need the `qwen2.5:3b` model, but you can configure other models as well.

### Installation

1.  **Clone the repo**
    ```sh
    git clone https://github.com/dbudaghyan/qcluster.git
    cd qcluster
    ```
2.  **Set up the environment variables**
    ```sh
    cp .env.example .env
    ```
    *You can modify the `.env` file to change the default settings.*
3.  **Install `ollama`**
    *   Using `Homebrew` (on macOS):
        ```sh
        brew install ollama
        ```
    *   Or download the binary directly from the [official website](https://ollama.com/download).
4.  **Pull the LLM model**
    ```sh
    ollama pull qwen2.5:3b
    ```
    *If you have defined other models in your `.env` file, make sure to pull them as well.*
5.  **Start the `ollama` server**
    ```sh
    ollama serve
    ```
6.  **Install the Python dependencies**
    ```sh
    uv sync
    ```

<!-- USAGE -->
## Usage

You can run the clustering pipeline either as a simple Python script or through a Jupyter Notebook for a more interactive experience.

### Run the `qcluster.pipeline`
```bash
uv run qcluster.pipeline
```

### Jupyter Notebook
Add the project root to the Python path:
```bash
export PYTHONPATH=$(pwd)
```

Then run Jupyter Lab:
```bash
uv run --with jupyter jupyter-lab
```

The notebook is located at `notebooks/pipeline.ipynb`.
The reports will be saved in the `EVALUATION_RESULTS_DIR` defined in the `.env` file.

### TL;DR
```bash
cd customer_questions_clustering
cp .env.example .env
# Modify the .env file if needed
brew install ollama
ollama pull qwen2.5:3b
# pull other models if needed (if defined in the .env file)
ollama serve
uv sync
uv run qcluster.pipeline
# or
# export PYTHONPATH=$(pwd)
# uv run --with jupyter jupyter-lab
# and open notebooks/pipeline.ipynb and run the cells
```
The reports will be saved in the `EVALUATION_RESULTS_DIR` defined in the `.env` file.
