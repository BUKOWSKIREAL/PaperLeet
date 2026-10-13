.PHONY: dev check data build

PYTHON := .venv/bin/python
PAPERLEET := .venv/bin/paperleet

dev:
	$(PYTHON) scripts/dev.py

check:
	$(PYTHON) -m pytest tests -q
	$(PAPERLEET) ci

data:
	$(PAPERLEET) vizdata all

build:
	$(PAPERLEET) export
	cd web && npm run build
