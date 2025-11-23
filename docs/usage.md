# Usage

The Python surface of `yaml12` is intentionally small: four functions that parse and emit YAML plus a `Tagged` helper for non-core tags.

## Parse YAML text

Use `parse_yaml(text, multi=False, handlers=None)` when you already have YAML as a string. `text` accepts either a single string or a sequence of strings that will be joined with newlines. Empty input returns `None`.

```python
from yaml12 import parse_yaml

config = parse_yaml("title: yaml12\nitems:\n  - rust\n  - python\n")
assert config["items"] == ["rust", "python"]
```

- Set `multi=True` to parse a stream of documents into a list.
- Supply `handlers` (a `dict` mapping tag strings to callables) to coerce specific tags on the fly. See [Custom Tags](tags.md) for details.

## Read from files

`read_yaml(path, multi=False, handlers=None)` reads YAML from disk and parses it with the same semantics as `parse_yaml`.

```python
from yaml12 import read_yaml

settings = read_yaml("config.yml")
print(settings["debug"])
```

File I/O errors raise `IOError`. Parse problems surface as `ValueError`. Invalid handler definitions raise `TypeError`.

## Emit YAML text

`format_yaml(value, multi=False)` serializes a Python value (or list of documents when `multi=True`) into YAML text.

```python
from yaml12 import format_yaml

yaml_text = format_yaml({"env": "dev", "replicas": 2})
print(yaml_text)
# env: dev
# replicas: 2
```

When `multi=True`, the returned string ends with `...\n` to close the stream.

## Write YAML to disk or stdout

`write_yaml(value, path=None, multi=False)` writes YAML straight to a file when `path` is provided or to stdout otherwise.

```python
from yaml12 import write_yaml

write_yaml(["first", "second"], path="out.yml", multi=True)
# out.yml now contains:
# ---
# - first
# - second
# ...
```

`write_yaml` always closes single-document output with `...\n` to make downstream stream parsing unambiguous.
