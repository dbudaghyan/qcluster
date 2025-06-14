# QCluster
_A Python library for clustering customer questions._

## Requirements
- `Python 3.12+`
- `uv`
- `ollama` and the model `qwen2.5:3b`

## Setup
All the steps below where tested on a `macOS` machine, with `Apple Silicon` (M3) chip.
The LLM model 
#### Install `ollama` using `Homebrew`:
```bash
brew install ollama
```
#### Or download the binary directly:
You can download the latest release from [here](https://ollama.com/download)

#### Pull the `qwen2.5:3b` model:
```bash
ollama pull qwen2.5:3b
```

#### Make sure the `ollama` server is running:
```bash
ollama serve
```

#### Install the requirements using `uv`:
```bash
uv sync
```

#### Set up the environment
Option 1: Set the environment variables directly:
```bash
export TOKENIZERS_PARALLELISM=false
export OLLAMA_MODEL=qwen2.5:3b
export OLLAMA_HOST=http://localhost:11434
export SENTENCE_TRANSFORMERS_MODEL=all-mpnet-base-v2
export EVALUATION_RESULTS_DIR=../evaluation_results
```

Option 2: Create a `.env` file in the project root with the following content:
```ini
TOKENIZERS_PARALLELISM=false
OLLAMA_MODEL=qwen2.5:3b
OLLAMA_HOST=http://localhost:11434
SENTENCE_TRANSFORMERS_MODEL=all-mpnet-base-v2
EVALUATION_RESULTS_DIR=../evaluation_results
```

## Usage
### Python
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
