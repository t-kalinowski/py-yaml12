VENV ?= .venv
PY := $(VENV)/bin/python
PIP := $(VENV)/bin/pip

.PHONY: venv develop test docs docs-serve clean

venv:
	@test -d $(VENV) || python3 -m venv $(VENV)
	$(PIP) install -U pip
	$(PIP) install maturin pytest mkdocs

develop: venv
	$(VENV)/bin/maturin develop --locked

test: develop
	$(PY) -m pytest tests_py

docs: venv
	$(PY) -m mkdocs build

docs-serve: venv
	$(PY) -m mkdocs serve

clean:
	rm -rf $(VENV)
