from pathlib import Path

from setuptools import find_namespace_packages, setup

# Load packages from requirements.txt
BASE_DIR = Path(__file__).parent
with open(Path(BASE_DIR, "requirements.txt")) as file:
    required_packages = [line.strip() for line in file.readlines()]

docs_packages = [
    "mkdocs",
    "mkdocstrings",
    "mkdocstrings-python",
    "mkdocs-material",
]

style_packages = [
    "black",
    "flake8",
    "isort",
]

test_packages = [
    "pytest",
    "pytest-cov",
    "great-expectations",
]

setup(
    name="tagifai",
    version="0.0.1",
    description="Classify machine learning projects.",
    author="Naveen Agrawal",
    author_email="navee.agrawal@gmail.com",
    url="https://github.com/naveeagrawal",
    python_requires=">=3.7",
    packages=find_namespace_packages(),
    install_requires=[required_packages],
    extras_require={
        "dev": docs_packages + style_packages + test_packages + ["pre-commit"],
        "docs": docs_packages,
        "test": test_packages,
    },
)
