<div align="center">

  <h1 align="center">QCluster</h1>
  <p align="center">
    A Python library for clustering customer questions using large language models.
    <br />
    <a href="docs/index.md"><strong>Explore the documentation »</strong></a>
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
    <a href="https://github.com/dbudaghyan/qcluster/graphs/contributors"><img src="https://img.shields.io/github/contributors/dbudaghyan/qcluster.svg?style=for-the-badge" alt="Contributors"></a>
    <a href="https://github.com/dbudaghyan/qcluster/blob/master/LICENSE"><img src="https://img.shields.io/github/license/dbudaghyan/qcluster?label=license&style=for-the-badge" alt="License"></a>
    <br />
    <a href="https://github.com/dbudaghyan/qcluster/actions/workflows/flake8.yml"><img src="https://img.shields.io/github/actions/workflow/status/dbudaghyan/qcluster/flake8.yml?style=for-the-badge&logo=python" alt="Flake8"></a>
    <a href="https://www.codefactor.io/repository/github/dbudaghyan/qcluster"><img src="https://img.shields.io/codefactor/grade/github/dbudaghyan/qcluster?style=for-the-badge" alt="CodeFactor"></a>
    <a href="https://coveralls.io/github/dbudaghyan/qcluster?branch=master"><img src="https://img.shields.io/coveralls/github/dbudaghyan/qcluster/master.svg?style=for-the-badge" alt="Coverage Status"></a>
    <a href="https://github.com/dbudaghyan/qcluster/commits/master"><img src="https://img.shields.io/github/last-commit/dbudaghyan/qcluster.svg?style=for-the-badge" alt="Last Commit"></a>
    <a href="https://github.com/dbudaghyan/qcluster"><img src="https://img.shields.io/github/repo-size/dbudaghyan/qcluster.svg?style=for-the-badge" alt="Repo Size"></a>
    <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/Python-3.12+-blue?style=for-the-badge&logo=python" alt="Python Version"></a>
    <br />
    <a href="https://github.com/dbudaghyan/qcluster/graphs/commit-activity"><img src="https://img.shields.io/github/commit-activity/m/dbudaghyan/qcluster?style=for-the-badge" alt="Commit Activity"></a>
    <a href="https://github.com/dbudaghyan/qcluster"><img src="https://img.shields.io/github/languages/top/dbudaghyan/qcluster?style=for-the-badge" alt="Top Language"></a>
    <a href="http://makeapullrequest.com"><img src="https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=for-the-badge" alt="PRs Welcome"></a>
  </p>
</div>

<!-- ABOUT THE PROJECT -->
<h2 id="about-the-project">🎯 About The Project</h2>

QCluster is a powerful Python library designed to help you make sense of large volumes of
customer feedback. By leveraging the power of Large Language Models (LLMs), QCluster can
automatically group similar customer questions, allowing you to identify trends, pain
points, and frequently asked questions with ease.

This project provides a complete pipeline for:

* Extracting customer questions from your data sources.
* Generating embeddings for each question.
* Clustering the questions based on their semantic similarity.
* Evaluating the quality of the clustering results.
* Generating insightful reports.

<!-- GETTING STARTED -->
<h2 id="getting-started">🚀 Getting Started</h2>

For detailed instructions on how to install, configure, and run QCluster, please see the [**full documentation**](docs/index.md).

The basic steps are:
1. Clone the repository.
2. Set up the environment variables.
3. Install the dependencies using `uv sync`.
4. Run the pipeline using `uv run qcluster.pipeline`.

<!-- LICENSE -->
<h2 id="license">📄 License</h2>

Distributed under the GPL-2.0 License. See [LICENSE](LICENSE) for more information.
