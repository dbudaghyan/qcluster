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
cp .env.example .env
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
The reports will be saved in the `EVALUATION_RESULTS_DIR` defined in the `.env` file.

### TLDR
```bash
cp .env.example .env
brew install ollama
ollama pull qwen2.5:3b
ollama serve
uv sync
uv run qcluster.pipeline
```
