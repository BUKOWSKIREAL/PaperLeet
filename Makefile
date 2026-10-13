.PHONY: dev check data build deploy-build deploy-up deploy-down deploy-backup

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

deploy-build:
	sh deploy/manage.sh build

deploy-up:
	sh deploy/manage.sh up

deploy-down:
	sh deploy/manage.sh down

deploy-backup:
	sh deploy/manage.sh backup
