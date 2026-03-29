from setuptools import setup, find_packages

setup(
    name="adversarial-market-marl",
    version="0.1.0",
    packages=find_packages(exclude=["tests*", "scripts*", "docs*"]),
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.1.0",
        "numpy>=1.24.0",
        "gymnasium>=0.29.0",
        "pyyaml>=6.0",
        "tensorboard>=2.14.0",
        "tqdm>=4.66.0",
        "rich>=13.0.0",
        "einops>=0.7.0",
        "scipy>=1.11.0",
        "pandas>=2.0.0",
        "matplotlib>=3.7.0",
    ],
    entry_points={
        "console_scripts": [
            "amm-train=scripts.train:main",
            "amm-eval=scripts.evaluate:main",
            "amm-sweep=scripts.sweep_lambda:main",
        ]
    },
)
