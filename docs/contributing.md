# Contributing

See [CONTRIBUTING.md](https://github.com/yourusername/adversarial-market-marl/blob/main/CONTRIBUTING.md) in the repository root for full details.

## Quick reference

```bash
# Setup
git clone https://github.com/yourusername/adversarial-market-marl.git
cd adversarial-market-marl
pip install -r requirements.txt && pip install -e .
pre-commit install

# Run all checks before pushing
black --target-version py310 --line-length 100 adversarial_market tests scripts
isort --profile black --line-length 100 adversarial_market tests scripts
flake8 adversarial_market scripts --max-line-length=100 --ignore=E203,W503
mypy adversarial_market --ignore-missing-imports --no-strict-optional
pytest tests/ -v
```

## Pull request checklist

- [ ] All 222 tests pass
- [ ] Black, isort, and flake8 pass
- [ ] mypy passes
- [ ] New functionality has tests
- [ ] Docstrings added for public methods
