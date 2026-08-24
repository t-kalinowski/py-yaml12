VENV ?= .venv
PY := $(VENV)/bin/python
PIP := $(VENV)/bin/pip

.PHONY: venv develop test docs docs-serve clean

venv:
	@test -d $(VENV) || python3 -m venv $(VENV)
	$(PIP) install -U pip
	$(PIP) install --group test

develop: venv
	$(VENV)/bin/maturin develop --locked

test: develop
	$(PY) -m pytest tests_py

docs: develop
	$(PIP) install --group docs
	QUARTO_PYTHON="$(CURDIR)/$(PY)" uvx great-docs build

docs-serve: develop
	$(PIP) install --group docs
	QUARTO_PYTHON="$(CURDIR)/$(PY)" uvx great-docs preview

clean:
	rm -rf $(VENV)
