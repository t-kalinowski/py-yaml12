# yaml12

A YAML 1.2 parser/formatter for Python, implemented in Rust for speed
and correctness. Built on the excellent
[`saphyr`](https://github.com/saphyr-rs/saphyr) crate.

For almost every use case, `yaml12` lets you work with plain builtin
Python types end to end: `dict`, `list`, `int`, `float`, `str`, and
`None`. JSON is a subset of YAML 1.2, so all valid JSON is also valid
YAML and parses the same way.

- Parse YAML text or files with `parse_yaml()` and `read_yaml()`.
- Serialize Python values with `format_yaml()` or `write_yaml()`.
- 100% compliance with the [yaml-test-suite](https://github.com/yaml/yaml-test-suite).
- Advanced YAML features (document streams, tags, complex mapping keys) are supported and
  round-trip cleanly when needed. `Yaml` is the wrapper type for tagged nodes and unhashable
  mapping keys.

## Installation

The package ships prebuilt wheels for Python 3.10+ on common platforms. Install from PyPI:

```bash
pip install yaml12
```

## Development install

You can install the development version of `yaml12` from github.
Clone the repository and install in editable mode:

```bash
git clone https://github.com/posit-dev/py-yaml12.git
cd py-yaml12
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e . --no-build-isolation
```

To install the latest main branch without cloning:

```bash
pip install git+https://github.com/posit-dev/py-yaml12.git
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

tagged = parse_yaml("!expr 1 + 1")
assert tagged == Yaml(value="1 + 1", tag="!expr")
```

## Reading and writing files

```python
from yaml12 import read_yaml, write_yaml

value_out = {"alpha": 1, "nested": [True, None]}

write_yaml(value_out, "my.yaml")
value_in = read_yaml("my.yaml")
assert value_in == value_out

# Multi-document streams
docs_out = [{"foo": 1}, {"bar": [2, None]}]
write_yaml(docs_out, "my-multi.yaml", multi=True)
docs_in = read_yaml("my-multi.yaml", multi=True)
assert docs_in == docs_out
```

## Tag handlers

Handlers let you opt into custom behavior for tagged nodes while
keeping the default parser strict and safe.

```python
from yaml12 import parse_yaml

yaml_text = """
- !upper [rust, python]
- !expr 6 * 7
"""

handlers = {
    "!expr": lambda value: eval(value),
    "!upper": lambda value: [x.upper() for x in value],
}

doc = parse_yaml(yaml_text, handlers=handlers)
assert doc == [["RUST", "PYTHON"], 42]
```

## Non-string mapping keys and tags

YAML mappings can use keys that themselves collections, or that carry
tags. Such keys cannot always be represented directly in a Python
`dict`, so `yaml12` wraps them in `Yaml` to make the key hashable.

```python
from yaml12 import Yaml, parse_yaml, format_yaml

obj = {
    "seq": [1, 2],
    "map": {"key": "value"},
    "tagged": Yaml("1 + 1", "!expr"),
    Yaml("foo", "!custom-key"): "bar",
}

yaml_text = format_yaml(obj)
round_tripped = parse_yaml(yaml_text)
assert round_tripped == obj
```

## Documentation

Online docs: https://posit-dev.github.io/py-yaml12/

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
