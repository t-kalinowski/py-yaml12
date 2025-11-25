# yaml12

A YAML 1.2 parser/formatter for Python, implemented in Rust for speed
and correctness. Built on the excellent
[`saphyr`](https://github.com/saphyr-rs/saphyr) crate.

For almost every use case, `yaml12` lets you work with plain builtin Python types end to end: `dict`, `list`, `int`, `float`, `str`, and `None`.

- Parse YAML text or files with `parse_yaml()` and `read_yaml()`.
- Serialize Python values with `format_yaml()` or `write_yaml()`.
- 100% compliance with the [yaml-test-suite](https://github.com/yaml/yaml-test-suite).
- Advanced YAML features (document streams, tags, complex mapping keys) are supported and
  round-trip cleanly when needed; see the advanced guide if needed.

## Install (local dev)

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e . --no-build-isolation
```

Python 3.10+ and a Rust toolchain are required.

## Quick start

```python
from yaml12 import parse_yaml, format_yaml, Yaml

yaml_text = """
title: A modern YAML parser and emitter written in Rust
properties: [fast, correct, safe, simple]
features:
  tags: preserve
  streams: multi
"""

doc = parse_yaml(yaml_text)

assert doc == {
    "title": "A modern YAML parser and emitter written in Rust",
    "properties": ["fast", "correct", "safe", "simple"],
    "features": {"tags": "preserve", "streams": "multi"},
}

round_tripped = parse_yaml(format_yaml(doc))
assert round_tripped == doc

# Tagged values (advanced)
from yaml12 import Yaml

tagged = parse_yaml("!expr 1+1")
assert tagged == Yaml(value="1 + 1", tag="!expr")
```

## Docs

- Guides: `docs/usage.md` (basics) and `docs/tags.md` (advanced tags,
  anchors, streams).
- Reference: `docs/reference/`.
- API overview: `docs/api.md`.

To build or serve the docs locally:

```bash
.venv/bin/mkdocs build        # or: .venv/bin/mkdocs serve
```

## Tests

From the repo root:

```bash
cargo fmt
cargo check
cargo test
cargo build
cargo clippy
.venv/bin/pip install -e . --no-build-isolation
.venv/bin/python -m pytest tests_py
```
