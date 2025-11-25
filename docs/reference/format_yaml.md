# Format or write Python objects as YAML 1.2.

`format_yaml()` returns YAML as a string. `write_yaml()` writes a YAML
stream to a file or writer. Both functions honor `Yaml` tags on values
and mapping keys; scalar core tags (`!!str`, `!!int`, `!!bool`,
`!!float`, `!!null`, `!!seq`, `!!map`) are dropped on emit because they
add no extra information.

## Usage

```python
format_yaml(value, multi=False)

write_yaml(value, path=None, multi=False)
```

## Arguments

- value: Any combination of Python scalars, lists, and dicts; tagged
  values or mapping keys can be wrapped in `Yaml`. When `multi=True`,
  `value` must be a sequence of documents.
- multi: When `True`, treat `value` as a list of YAML documents and
  encode a stream separated by `---` and terminated with `...`.
- path: Destination path (`str` or `os.PathLike`) or writable object
  with `.write()`. When `None`, write to stdout. Writers are tried with
  text first, then retried as bytes if text writes fail.

## Returns

`format_yaml()` returns a `str` containing YAML (ending with `...\n` for
multi-document output); single-document output omits the leading `---`
marker and trailing newline. `write_yaml()` returns `None`.

## Examples

```python
from pathlib import Path
from yaml12 import Yaml, format_yaml, write_yaml

format_yaml({"foo": 1, "bar": [True, None]})
# 'foo: 1\nbar:\n  - true\n  - ~'

docs = [{"foo": 1}, {"bar": [2, None]}]
format_yaml(docs, multi=True)
# '---\nfoo: 1\n---\nbar:\n  - 2\n  - ~\n...\n'

tagged = Yaml("1 + 1", "!expr")
format_yaml(tagged)
# '!expr 1 + 1'

path = Path("example.yaml")
write_yaml({"alpha": 1}, path)
assert path.read_text(encoding="utf-8") == f"---\n{format_yaml({'alpha': 1})}\n...\n"
```
