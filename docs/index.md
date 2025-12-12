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

## Installation

The project targets Python 3.10+ and builds via `maturin` (Rust
toolchain required). From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e . --no-build-isolation
```

## Quick start

```python
from yaml12 import parse_yaml, format_yaml

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

text = format_yaml(doc)
print(text)
# title: A modern YAML parser and emitter written in Rust
# properties:
#   - fast
#   - correct
#   - safe
#   - simple
# features:
#   tags: preserve
#   streams: multi
```

### Reading and writing files

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

### Tag handlers

Handlers let you opt into custom behaviour for tagged nodes while
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

### Formatting and round-tripping

```python
from yaml12 import format_yaml, parse_yaml, Yaml

obj = {
    "seq": [1, 2],
    "map": {"key": "value"},
    "tagged": Yaml(value="1 + 1", tag="!expr"),
}

yaml_text = format_yaml(obj)
print(yaml_text)
# seq:
#   - 1
#   - 2
# map:
#   key: value
# tagged: !expr 1 + 1

parsed = parse_yaml(yaml_text)
assert parsed == {
    "seq": [1, 2],
    "map": {"key": "value"},
    "tagged": Yaml("1 + 1", "!expr"),
}
```

### Tagged nodes and mapping keys (advanced)

Tags, custom handlers, and non-string mapping keys work without extra
setup when you need them. Nodes that can’t be represented as plain
Python types are wrapped in `Yaml` (a small frozen dataclass). You’ll
only see `Yaml` when:

- A tagged node has no matching handler; inspect `.value` and `.tag`.
- A mapping key is a collection or otherwise unhashable; wrapping in `Yaml()` makes
  it hashable.

```python
from yaml12 import Yaml, format_yaml, parse_yaml

mapping = {Yaml("tagged-key", "!k"): "v"}
assert parse_yaml(format_yaml(mapping)) == mapping
```

See the advanced guide for details.

## Documentation

- [YAML in 2 Minutes](usage.md) for a quick primer.
- [Tags, Anchors, and Advanced YAML](tags.md) for handlers, mapping keys,
  document streams, and advanced tags.
- [Reference](reference/index.md) for detailed signatures and examples.
- [Contributing](contributing.md) for building or serving the docs locally.
