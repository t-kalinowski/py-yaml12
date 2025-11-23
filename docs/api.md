# API Reference

All functions live in the top-level `yaml12` module. Arguments mirror the Rust bindings and are intentionally strict so type errors surface early.

## parse_yaml(text, multi=False, handlers=None)

- `text`: `str` or sequence of `str`; joined with newlines.
- `multi`: when `True`, returns a list of documents; otherwise a single document or `None` for empty input.
- `handlers`: optional `dict[str, Callable]` mapping tag strings to callables; applies to tagged values and keys.
- Raises `ValueError` on YAML parse errors or invalid tags; `TypeError` on wrong argument types.

## read_yaml(path, multi=False, handlers=None)

- Reads and parses the YAML file at `path` with the same semantics as `parse_yaml`.
- Raises `IOError` if the file cannot be read; otherwise identical error behaviour to `parse_yaml`.

## format_yaml(value, multi=False)

- Serializes a Python value (or list of documents when `multi=True`) into a YAML string.
- When `multi=True`, the returned string ends with `...\n` to close the stream.
- Raises `TypeError` if `multi=True` and `value` is not a sequence, or when unsupported types are provided.

## write_yaml(value, path=None, multi=False)

- Serializes `value` like `format_yaml` and writes to `path` if provided, or stdout when `path` is `None`.
- Always terminates single-document output with `...\n` and multi-document output with `...\n` after the final document.
- Raises `IOError` on write failures or `TypeError` when inputs are invalid.

## Tagged

Frozen `dataclass` with fields:

- `value`: the parsed Python value.
- `tag`: the rendered tag string (for example `"!foo"` or `"tag:yaml.org,2002:timestamp"`).

`Tagged` is produced for non-core tags without handlers. You can also construct it manually when emitting YAML to preserve tags.
