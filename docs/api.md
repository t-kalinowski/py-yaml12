# API Reference

The `yaml12` module exposes parsing/formatting helpers plus two small dataclasses. Arguments mirror the Rust bindings and stay strict so type errors surface early.

## parse_yaml(text, multi=False, handlers=None)

- `text`: `str` or sequence of `str`; sequence items are joined with newlines.
- `multi`: when `True`, parses the whole stream into a list (empty input yields `[]`); otherwise returns only the first document or `None` for empty input.
- `handlers`: optional `dict[str, Callable]` mapping tag strings to callables; handlers apply to tagged values and keys and run on already-parsed Python values.
- Returns parsed Python values. Non-core tags (and informative core tags like `!!timestamp`, `!!binary`, `!!set`, `!!omap`, `!!pairs`, or non-specific `!`) become `Tagged` when no handler matches. Scalar core tags (`!!str`, `!!int`, `!!bool`, `!!float`, `!!null`, `!!seq`, `!!map`) are normalized to plain Python types.
- Unhashable mapping keys (sequences/mappings) are wrapped in `MappingKey` so they can be hashable.
- Raises `ValueError` on YAML parse errors or invalid tag strings; `TypeError` on wrong argument types; handler exceptions propagate unchanged.

## read_yaml(path, multi=False, handlers=None)

- Reads and parses YAML from `path` (string, `os.PathLike`, or readable object yielding `str`/`bytes`) with the same semantics as `parse_yaml`.
- Streaming readers are supported; chunk sizes are handled internally.
- Raises `IOError` if the file cannot be read; otherwise identical error behaviour to `parse_yaml`.

## format_yaml(value, multi=False)

- Serializes a Python value (or list of documents when `multi=True`) into a YAML string. `Tagged` values keep their tags unless they target core scalar tags; `MappingKey` preserves complex mapping keys.
- When `multi=False`, the returned string omits the leading document marker and trailing newline. When `multi=True`, the string starts with `---\n` and ends with `...\n`; an empty sequence emits `---\n...\n`.
- Raises `TypeError` if `multi=True` and `value` is not a sequence, or when unsupported types are provided.

## write_yaml(value, path=None, multi=False)

- Serializes `value` like `format_yaml` and writes to `path` (string, `os.PathLike`, or object with `.write()`) if provided, or stdout when `path` is `None`.
- Single-document output is always wrapped with `---\n` and terminated with `...\n`; multi-document streams include `...\n` after the final document.
- Writers are tried with text first and retried as bytes if needed.
- Raises `IOError` on write failures or `TypeError` when inputs are invalid.

## Tagged

Frozen `dataclass` with fields:

- `value`: the parsed Python value.
- `tag`: the rendered tag string (for example `"!foo"` or `"tag:yaml.org,2002:timestamp"`).

Produced for non-core tags (and informative core tags such as timestamps or binary) when no handler applies. You can construct it manually when emitting YAML to preserve tags; core scalar tags are stripped during serialization because they carry no extra information.

## MappingKey

Frozen `dataclass` that wraps unhashable mapping keys (sequences, mappings, or tagged values). Equality and hashing use a frozen structural view of `value`, and it proxies indexing/iteration/len to the wrapped object when applicable. Use it to emit or compare YAML mappings that rely on complex keys.

## _dbg_yaml(text)

Debug helper that pretty-prints parsed YAML nodes (including tags) without converting them to Python values. Accepts the same `text` forms as `parse_yaml`.
