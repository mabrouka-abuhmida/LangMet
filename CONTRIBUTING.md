# Contributing to LangMet

Thanks for your interest in improving LangMet! This guide covers the basics for
local development.

## Development setup

```bash
git clone https://github.com/mabrouka-abuhmida/LangMet.git
cd LangMet
python -m pip install -e ".[dev,sqlalchemy]"
```

Optional extras:

- `".[embeddings]"` — enables `EmbeddingScorer` (sentence-transformers)
- `".[fastapi]"` — runs the two-minute demo

## Running checks

```bash
ruff check .        # lint
pytest              # test suite
python -m build     # build sdist + wheel
twine check dist/*  # validate distribution metadata
```

Please make sure `ruff check .` and `pytest` both pass before opening a pull
request.

## Architecture

LangMet follows a hexagonal (ports-and-adapters) design:

- `langmet/models.py` — immutable event dataclasses (the contracts)
- `langmet/analytics.py` — pure, dependency-free computation functions
- `langmet/cost.py` — cost / token-budget analytics
- `langmet/scoring.py` — pluggable RAGAS scorers (`RagaScorer` protocol)
- `langmet/alerts.py` — threshold-based alerting over computed metrics
- `langmet/ports.py` — the `MetricsRepository` protocol
- `langmet/service.py` — orchestration facade
- `langmet/adapters/` — infrastructure adapters (in-memory, SQLAlchemy)

Keep the core dependency-free: new third-party dependencies belong in an
optional extra, imported lazily so the base install stays lightweight.

## Pull requests

1. Branch from `main`.
2. Add tests for new behaviour.
3. Update `CHANGELOG.md` under an `Unreleased`/next-version heading.
4. Keep changes focused and the diff small.

## Releases

Releases are published to PyPI by the `Publish to PyPI` GitHub Actions workflow
when a GitHub Release is published (or via manual `workflow_dispatch`). Bump the
version in both `pyproject.toml` and `langmet/__init__.py` first.
