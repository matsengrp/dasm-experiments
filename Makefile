default:

githubciinstall:
	pip install netam@git+https://github.com/matsengrp/netam.git
	pip install -e .

install:
	pip install -e .

formatdocs:
	docformatter --in-place --black --recursive dnsmex

format:
	docformatter --in-place --black --recursive dnsmex tests || echo "Docformatter made changes"
	black dnsmex tests

format-commit-push:
	@make format || true
	@if [ -n "$$(git status --porcelain)" ]; then \
		git add -u && \
		git commit -m "make format" && \
		git push; \
	else \
		echo "No formatting changes to commit"; \
		git push; \
	fi

checkformat:
	# docformatter --check --black --recursive dnsmex
	black --check dnsmex tests

checktodo:
	(find . -name "*.py" -o -name "*.Snakemake" | grep -v "/\." | xargs grep -l "TODO") && echo "TODOs found" && exit 1 || echo "No TODOs found" && exit 0

test:
	pytest tests
	cd tests/simulation; ./test_simulation_cli.sh; cd ../..

lint:
	flake8 dnsmex --max-complexity=30 --ignore=E731,W503,E402,F541,E501,E203,E266 --statistics --exclude=__pycache__


runnotebooks:
	./run_notebooks.sh

# Remote training configuration - loaded from local-config.py
REMOTE_HOST = $(shell python -c "from dnsmex.local import get_remote_config; print(get_remote_config()['host'])")
REMOTE_DIR = $(shell python -c "from dnsmex.local import get_remote_config; print(get_remote_config()['dir'])")
REMOTE_VENV_PATH = $(shell python -c "from dnsmex.local import get_remote_config; print(get_remote_config()['venv'])")
LOCAL_BRANCH = $$(git rev-parse --abbrev-ref HEAD)

# Ensure we're on the same branch and sync code
remote-sync:
	@echo "Checking git status..."
	@if [ -n "$$(git status --porcelain)" ]; then \
		echo "Error: You have uncommitted changes. Please commit or stash them first."; \
		exit 1; \
	fi
	@echo "Current branch: $(LOCAL_BRANCH)"
	@echo "Pushing to origin..."
	git push origin $(LOCAL_BRANCH)
	@echo "Syncing with $(REMOTE_HOST)..."
	ssh -A $(REMOTE_HOST) "cd $(REMOTE_DIR) && git fetch origin && git checkout $(LOCAL_BRANCH) && git pull --ff-only origin $(LOCAL_BRANCH)"
	@echo "Remote sync complete!"

# Run command on remote server with live output
remote-run: remote-sync
	@if [ -z "$(CMD)" ]; then \
		echo "Error: Please specify CMD='your command'"; \
		echo "Example: make remote-run CMD='python -m dnsmex.dasm_zoo ...'"; \
		exit 1; \
	fi
	@echo "Running on $(REMOTE_HOST): $(CMD)"
	ssh -At $(REMOTE_HOST) "cd $(REMOTE_DIR) && source $(REMOTE_VENV_PATH) && $(CMD)"

# Copy remote command to clipboard for manual execution
remote-copy:
	@if [ -z "$(CMD)" ]; then \
		echo "Error: Please specify CMD='your command'"; \
		echo "Example: make remote-copy CMD='python -m dnsmex.dasm_zoo ...'"; \
		exit 1; \
	fi
	@echo "make remote-run CMD='$(CMD)'" | pbcopy
	@echo "Command copied to clipboard! Paste and run in your terminal to see live progress."

# Fetch results from remote server using rsync
remote-fetch:
	@if [ -z "$(DIR)" ]; then \
		echo "Error: Please specify DIR='path/to/sync'"; \
		echo "Example: make remote-fetch DIR='dasm-train/trained_models'"; \
		echo "Example: make remote-fetch DIR='crepe'"; \
		exit 1; \
	fi
	@echo "Syncing $(DIR) from $(REMOTE_HOST)..."
	rsync -av $(REMOTE_HOST):$(REMOTE_DIR)/$(DIR)/ $(DIR)/
	@echo "Results synced to $(DIR)/"

# Show remote command help
remote-help:
	@echo "Remote commands:"
	@echo "  make remote-sync       - Sync code to remote server"
	@echo "  make remote-run        - Run command on remote server"
	@echo "  make remote-copy       - Copy command to clipboard for manual run"
	@echo "  make remote-fetch      - Fetch results using rsync"
	@echo "  make remote-status     - Check remote git status"
	@echo ""
	@echo "Examples:"
	@echo "  make remote-run CMD='python -m dnsmex.dasm_zoo train_model dasm_77k tst joint 0'"
	@echo "  make remote-fetch DIR='dasm-train/trained_models'"

# Quick status check on remote
remote-status:
	@echo "Checking remote status..."
	ssh -A $(REMOTE_HOST) "cd $(REMOTE_DIR) && git status --short && git branch --show-current"

# Setup git hooks
setup-hooks:
	@echo "Setting up git hooks..."
	@cat > .git/hooks/pre-push << 'EOF'
	#!/bin/bash
	echo "Running make format..."
	if ! make format; then
	    echo "Error: make format failed. Please fix formatting issues and try again."
	    exit 1
	fi
	echo "Format check passed!"
	exit 0
	EOF
	@chmod +x .git/hooks/pre-push
	@echo "Pre-push hook installed! It will run 'make format' before each push."

.PHONY: default install test formatdocs format format-commit-push checkformat checktodo lint runnotebooks githubciinstall remote-sync remote-run remote-copy remote-fetch remote-help remote-status setup-hooks
