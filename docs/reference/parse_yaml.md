# Parse YAML 1.2 text or files into Python objects.

`parse_yaml()` takes YAML text; `read_yaml()` reads from a path or a
streaming reader.

## Usage

```python
parse_yaml(text, multi=False, handlers=None)

read_yaml(path, multi=False, handlers=None)
```

## Arguments

- text: `str` or sequence of `str`; sequence items are joined with
  `"\n"`. When empty, returns `None` (or `[]` when `multi=True`).
- path: `str`, `os.PathLike`, or readable object yielding `str` or
  UTF-8 `bytes`.
- multi: When `True`, parse the whole stream and return a list of
  documents; when `False`, stop after the first document.
- handlers: Optional `dict[str, Callable]` keyed by YAML tag strings;
  matching handlers transform tagged values and keys. Exceptions from
  handlers propagate unchanged.

## Returns

When `multi=False`, the first document or `None` for empty input. When
`multi=True`, a list of all documents. Tagged nodes without a matching
handler (including informative core tags such as `!!timestamp` or
`!!binary`) become `Yaml` objects. Unhashable mapping keys are wrapped
in `Yaml` so they remain hashable.

## Examples

```python
from yaml12 import parse_yaml

parse_yaml("foo: [1, 2, 3]")
# {'foo': [1, 2, 3]}

stream = """
---
first: 1
---
second: 2
"""
parse_yaml(stream)          # returns {'first': 1}
parse_yaml(stream, multi=True)  # returns [{'first': 1}, {'second': 2}]

handlers = {"!upper": lambda value: str(value).upper()}
parse_yaml("!upper key: !upper value", handlers=handlers)
# {'KEY': 'VALUE'}

lines = [
    "---",
    "title: Front matter only",
    "params:",
    "  answer: 42",
    "---",
    "# Body that is not YAML",
]
parse_yaml(lines)
# {'title': 'Front matter only', 'params': {'answer': 42}}
```
