SHELL = /bin/bash

# Set variable
MESSAGE := "hello world"

# Use variable
greeting:
	@echo ${MESSAGE}

# Environment
createv:
	@source venv/bin/activate

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
