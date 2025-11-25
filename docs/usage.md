# YAML in 2 Minutes: A Gentle Introduction for Python Users

```python
from yaml12 import parse_yaml
```

YAML is a human-friendly data serialization format. Think of it as
"JSON with comments and nicer multiline strings." `yaml12` follows the
modern YAML 1.2 spec (no surprising 1.1-era conversions).

YAML has three building blocks: **scalars** (single values),
**sequences** (ordered collections), and **mappings** (key/value pairs).
JSON is a subset of YAML 1.2, so all valid JSON is also valid YAML and
parses the same way.

## A first example

```yaml
title: A Modern YAML parser written in Rust
properties: [correct, safe, fast, simple]
score: 9.5
categories:
  - yaml
  - python
  - example
settings:
  note: >
    This is a folded block
    that turns line breaks
    into spaces.
  note_literal: |
    This is a literal block
    that keeps
    line breaks.
```

```python
first_example_text = """
title: A Modern YAML parser written in Rust
properties: [correct, safe, fast, simple]
score: 9.5
categories:
  - yaml
  - python
  - example
settings:
  note: >
    This is a folded block
    that turns line breaks
    into spaces.
  note_literal: |
    This is a literal block
    that keeps
    line breaks.
"""

doc = parse_yaml(first_example_text)

assert doc == {
    "title": "A Modern YAML parser written in Rust",
    "properties": ["correct", "safe", "fast", "simple"],
    "score": 9.5,
    "categories": ["yaml", "python", "example"],
    "settings": {
        "note": "This is a folded block that turns line breaks into spaces.\n",
        "note_literal": "This is a literal block\nthat keeps\nline breaks.\n",
    },
}
```

## Collections

There are two collection types: **sequences** and **mappings**.

### Sequences: YAML's ordered collections

Each item begins with `-` at the parent indent.

```yaml
- cat
- dog
```

parses to `["cat", "dog"]`.

JSON-style arrays work too:

```yaml
[cat, dog]
```

Anything belonging to one of the sequence entries is indented at least
one space past the dash:

```yaml
- name: cat
  toys: [string, box]
- name: dog
  toys: [ball, bone]
```

parses to:

```python
[
    {"name": "cat", "toys": ["string", "box"]},
    {"name": "dog", "toys": ["ball", "bone"]},
]
```

### Mappings: key/value pairs

Mappings are sets of `key: value` pairs at the same indent:

```yaml
foo: 1
bar: true
```

parses to `{"foo": 1, "bar": True}`.

A key at its indent owns anything indented more:

```yaml
settings:
  debug: true
  max_items: 3
```

parses to `{"settings": {"debug": True, "max_items": 3}}`.

JSON-style objects work too:

```yaml
{a: true}
```

-> `{"a": True}`

## Scalars

Everything that is not a collection is a scalar. Scalars can be block,
quoted, or plain.

### Block scalars

`|` starts a **literal** block that keeps newlines; `>` starts a
**folded** block that joins lines with spaces (except blank/indented
lines keep breaks). Block scalars always become strings.

```yaml
|
  hello
  world
```

-> `"hello\nworld\n"`

```yaml
>
  hello
  world
```

-> `"hello world\n"`

### Quoted scalars

Quoted scalars always become strings. Double quotes interpret escapes
(`\n`, `\t`, `\\`, `\"`). Single quotes are literal and do not interpret
escapes, except for `''` which is parsed as a single `'`.

```yaml
["line\nbreak", "quote: \"here\""]
```

-> `["line\nbreak", 'quote: "here"']`

```yaml
['line\nbreak', 'quote: ''here''']
```

-> `["line\\nbreak", "quote: 'here'"]`

### Plain (unquoted) scalars

Plain nodes can resolve to one of five types: string, int, float, bool,
or null.

- `true` / `false` -> `True` / `False`
- `null`, `~`, or empty -> `None`
- numbers: signed, decimal, scientific, hex (`0x`), octal (`0o`),
  `.inf`, `.nan` -> `int` or `float`
- everything else stays a string (`yes`, `no`, `on`, `off` and other
  aliases remain strings in YAML 1.2)

```yaml
[true, 123, 4.5e2, 0x10, .inf, yes]
```

-> `[True, 123, 450.0, 16, float("inf"), "yes"]`

## End-to-end example

```yaml
doc:
  pets:
    - cat
    - dog
  numbers: [1, 2.5, 0x10, .inf, null]
  integers: [1, 2, 3, 0x10, null]
  flags: {enabled: true, label: on}
  literal: |
    hello
    world
  folded: >
    hello
    world
  quoted:
    - "line\nbreak"
    - 'quote: ''here'''
  plain: [yes, no]
  mixed: [won't simplify, 123, true]
```

Python result:

```python
{
    "doc": {
        "pets": ["cat", "dog"],
        "numbers": [1, 2.5, 16, float("inf"), None],
        "integers": [1, 2, 3, 16, None],
        "flags": {"enabled": True, "label": "on"},
        "literal": "hello\nworld\n",
        "folded": "hello world\n",
        "quoted": ["line\nbreak", "quote: 'here'"],
        "plain": ["yes", "no"],
        "mixed": ["won't simplify", 123, True],
    }
}
```

## Quick notes

- Indentation defines structure for collections. Sibling elements share
  an indent; children are indented more. YAML 1.2 forbids tabs; use
  spaces.
- All JSON is valid YAML.
- Sequences stay Python lists; there is no vector "simplification."
- Block scalars (`|`, `>`) always produce strings.
- Booleans are only `true`/`false`; `null` maps to `None`.
- Numbers can be signed, scientific, hex (`0x`), octal (`0o`), `.inf`,
  and `.nan`.

These essentials cover most YAML you'll see. For tags, anchors, and
non-string mapping keys, see the advanced guide.
