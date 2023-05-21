SHELL = /bin/bash

# Set variable
MESSAGE := "hello world"

# Use variable
greeting:
	@echo ${MESSAGE}

# Makefile
.ONESHELL:
venv:
    python3 -m venv venv
    source venv/bin/activate && \
    python3 -m pip install --upgrade pip setuptools wheel && \
    python3 -m pip install -e ".[dev]" && \
    pre-commit install && \
    pre-commit autoupdate

.PHONY: help
help:
	@echo "Commands:"
	@echo "venv    : creates a virtual environment."
	@echo "style   : executes style formatting."
	@echo "clean   : cleans all unnecessary files."

# Test
.PHONY: test
test:
	pytest tests/model
	cd tests && great_expectations checkpoint run projects
	cd tests && great_expectations checkpoint run tags
	cd tests && great_expectations checkpoint run labeled_projects
