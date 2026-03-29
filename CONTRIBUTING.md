# Contributing

Thank you for your interest in contributing to the Adversarial Market MARL framework.

## Development setup

```bash
git clone https://github.com/yourusername/adversarial-market-marl.git
cd adversarial-market-marl
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install -e .
pre-commit install
```

## Running tests

```bash
# All tests
pytest tests/ -v

# Unit tests only (fast)
pytest tests/unit/ -v

# Integration tests
pytest tests/integration/ -v

# With coverage
pytest tests/ --cov=adversarial_market --cov-report=html
open htmlcov/index.html
```

## Code style

This project uses [Black](https://black.readthedocs.io/) for formatting, [isort](https://pycqa.github.io/isort/) for import ordering, and [flake8](https://flake8.pycqa.org/) for linting. All are enforced via pre-commit hooks.

```bash
# Format
black adversarial_market tests scripts
isort adversarial_market tests scripts

# Lint
flake8 adversarial_market tests scripts --max-line-length=100

# Type check
mypy adversarial_market
```

## Project structure conventions

- **All heavy computation** lives in `adversarial_market/networks/`. Agent wrapper classes in `agents/` are thin.
- **Configuration** is always passed as a dict, never accessed as globals.
- **No hardcoded hyperparameters** in source files — everything goes in `configs/`.
- **Tests** must not depend on GPU. CPU-only is the CI target.
- **Docstrings** on all public classes and methods. NumPy style preferred.

## Submitting changes

1. Fork the repository and create a feature branch: `git checkout -b feat/my-feature`
2. Make your changes with tests
3. Ensure all tests pass and pre-commit hooks are clean
4. Push and open a pull request using the provided template

## Reporting bugs

Use the GitHub issue tracker with the **bug report** template. Include:
- Python version, OS, PyTorch version
- Config file used (or relevant YAML snippet)
- Full traceback
- Minimal reproduction script if possible

## Proposing new features

Open a GitHub issue with the **feature request** template before writing code, particularly for:
- New agent architectures
- Alternative MI estimators
- New environment dynamics
- Training algorithm changes

## Questions

Open a GitHub Discussion for questions about the theory, implementation, or experimental design.
