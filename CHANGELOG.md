# Changelog

## [0.2]

### [0.2.3]
- Add extensive test coverage (80%+ coverage)
- Add a CI / CD pipeline with GitHub Actions
- Code cleanup using `black` and `isort`
- Update `README.md` with badges and project status

### [0.2.2]
- Make a streamlined version of `notebooks/pipeline.ipynb`
- Update setup instructions, add reproducibility details
- Refine environment configuration workflow.

### [0.2.1]
- Implement an end-to-end pipeline for streamlined workflow.
- Introduce `EvaluationResult` data model and enhance storage/reporting workflows.
- Add `flake8` for linting and reformat code with `black`.
- Improve logging and enhance the report template with Jinja rendering.
- Add `Objective.md` documentation and update `README.md` for easier setup.

### [0.2.0]
- Rename sample_analyzer to dataset_analyzer
- Add adjusted_rand_score to evaluation metrics
- Implement result storage functionality
- Add jupyter notebook
- Update the `README.md`
- Save `git diff` and the running script/notebook with the results

## [01]

### [0.1.9]
- F1 macro 65%
- Set ollama temperature to 0.0
- Add preload module to make sure deterministic output and preloaded env vars and models
- Modularize pipeline

### [0.1.8]
- F1 Macro: 55%
- Refactor pipeline and data models
- Modularize clustering method
- Simplify cluster category retrieval
- Update TODO with an experimentation module task


### [0.1.7]
- Add and integrate cluster-to-category matching
- Add evaluation module

### [0.1.6]
- Add cluster describer, using LLMs to generate descriptions and titles
- Integrate with `InstructionCollection`

### [0.1.5]
- Add dissimilarity module and integrate with `InstructionCollection`

### [0.1.4]
- Add clustering module
- Add clustering to the pipeline
- Update data models and dependencies

### [0.1.3]
- Add instruction embedding updates
- Enhance InstructionCollection methods
- Add the main pipeline script

### [0.1.2]
- Refactor data models to improve consistency
- Update feature engineering
- Update `tests/test_datamodels.py`

### [0.1.1]
- Add feature engineering module
- Update dependencies
- Improve data models

### [0.1.0]
- Initialize the project structure
- Add dataset definitions