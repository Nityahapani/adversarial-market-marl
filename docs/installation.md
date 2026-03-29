# Installation

## Prerequisites

| Requirement | Minimum | Notes |
|-------------|---------|-------|
| Python | 3.10 | 3.11 also tested in CI |
| PyTorch | 2.1 | CPU-only works; GPU strongly recommended |
| RAM | 16 GB | 32 GB recommended for `n_envs=8` |
| CUDA (optional) | 11.8 | For GPU-accelerated training |

## Standard install

```bash
git clone https://github.com/Nityahapani/adversarial-market-marl.git
cd adversarial-market-marl

python -m venv venv
source venv/bin/activate        # Linux / macOS
# venv\Scripts\activate         # Windows

pip install -r requirements.txt
pip install -e .

# Verify
python -c "import adversarial_market; print('OK —', adversarial_market.__version__)"
```

## GPU setup

```bash
# CUDA 11.8
pip install torch==2.1.0+cu118 --index-url https://download.pytorch.org/whl/cu118

# CUDA 12.1
pip install torch==2.1.0+cu121 --index-url https://download.pytorch.org/whl/cu121
```

## Pre-commit hooks (contributors)

```bash
pip install pre-commit
pre-commit install
```

Hooks run automatically on `git commit`: black, isort, flake8, merge-conflict detection.
