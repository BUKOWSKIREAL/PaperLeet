.PHONY: dev check data build ci

PYTHON ?= $(if $(wildcard .venv/bin/python),.venv/bin/python,python3)

dev:
	$(PYTHON) scripts/dev.py

check:
	$(PYTHON) -m paperleet.cli export
	$(PYTHON) -m pytest tests -q
	npm --prefix web run check

data:
	$(PYTHON) -m paperleet.cli vizdata all

build:
	$(PYTHON) -m paperleet.cli export
	npm --prefix web run build
	npm --prefix web run check:build

ci: check
	$(PYTHON) -m paperleet.cli ci --repeats 1 --budget 600
	$(MAKE) build PYTHON=$(PYTHON)
