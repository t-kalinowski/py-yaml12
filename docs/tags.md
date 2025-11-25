# YAML Tags, Anchors, and Advanced Features with yaml12

This guide picks up where the 2-minute intro leaves off. It shows what
YAML tags are and how to work with them in `yaml12` using handlers.
Along the way it covers complex mapping keys and document streams so you
can handle real-world YAML.

## Tags in YAML and how yaml12 handles them

Tags annotate any YAML node with extra meaning. They always start with
`!` in YAML syntax and appear before the node's value; they are not part
of the scalar text itself.

`yaml12` preserves tags as `Yaml` objects. The object carries `value` (a
regular Python type) and `tag` (a string). Parsing a tagged scalar:

```python
from yaml12 import Yaml, parse_yaml

color = parse_yaml("!color red")
assert isinstance(color, Yaml) and (color.tag, color.value) == ("!color", "red")
```

Custom tags bypass the usual scalar typing; the scalar is returned as a
string even if it looks like another type.

```python
assert parse_yaml("! true") == Yaml(value="true", tag="!")
assert parse_yaml("true") is True
```

### Using handlers to transform tagged nodes while parsing

`parse_yaml()` and `read_yaml()` accept `handlers`: a `dict` mapping tag
strings to callables. Handlers run on any matching tagged node. For
tagged scalars the handler receives the Python scalar; for tagged
sequences or mappings it receives a plain list/dict.

```python
from dataclasses import dataclass
from yaml12 import parse_yaml

@dataclass(frozen=True)
class Point:
    x: int
    y: int

def point_handler(value):
    return Point(x=value["x"], y=value["y"])

doc = parse_yaml(
    "vertex: !point {x: 1,  y: 2}",
    handlers={"!point": point_handler},
)

assert doc["vertex"] == Point(1, 2)
```

Handlers apply to both values and keys, including non-specific `!` tags
if you register `"!"`. If a handler raises an exception, it propagates
unchanged to make debugging easy.

Any tag without a matching handler stays as a `Yaml` object. Handlers
without matching tags are simply unused.

### Post-process tags yourself

You can also parse without handlers and walk the result yourself. For
example, processing `!expr` scalars manually:

```python
from yaml12 import Yaml, parse_yaml

def eval_yaml_expr_nodes(obj):
    if isinstance(obj, Yaml):
        if obj.tag == "!expr":
            return eval(str(obj.value))
        return Yaml(eval_yaml_expr_nodes(obj.value), obj.tag)
    if isinstance(obj, list):
        return [eval_yaml_expr_nodes(item) for item in obj]
    if isinstance(obj, dict):
        return {eval_yaml_expr_nodes(k): eval_yaml_expr_nodes(v) for k, v in obj.items()}
    return obj

raw = parse_yaml("!expr 1 + 1")
assert isinstance(raw, Yaml) and eval_yaml_expr_nodes(raw) == 2
```

## Mappings revisited: non-string keys and `Yaml`

YAML mapping keys do not have to be plain strings; any node can be a
key. For example, this is valid even though the key is a boolean:

```yaml
true: true
```

Parsed with `yaml12`, unhashable or tagged keys become `Yaml` so they
can live in a Python `dict` while preserving equality and hashing by
structure:

```python
parsed = parse_yaml("true: true")
key = next(iter(parsed))
assert isinstance(key, Yaml) and key.value is True and parsed[key] is True
```

Complex keys use the explicit mapping-key indicator `?`:

```yaml
? [a, b]
: tuple
? {x: 1, y: 2}
: map-key
```

Becomes:

```python
parsed = parse_yaml(...above yaml...)
keys = list(parsed)
assert all(isinstance(k, Yaml) for k in keys) and keys[0].value == ["a", "b"] and keys[1].value == {"x": 1, "y": 2}
```

### Tagged mapping keys

Handlers run on keys too, so a handler can turn tagged keys into friendly
Python keys before they are wrapped.

```python
handlers = {"!upper": lambda value: str(value).upper()}
result = parse_yaml("!upper key: value", handlers=handlers)
assert result == {"KEY": "value"}
```

If you anticipate tagged mapping keys that you want to process yourself,
walk the `Yaml` keys alongside the values and unwrap them as needed.

## Document streams and markers

Most YAML files contain a single document. YAML also supports document
streams: multiple documents separated by `---` and optionally closed by
`...`.

### Reading multiple documents

`parse_yaml()` and `read_yaml()` default to `multi=False`, returning only
the first document. When `multi=True`, all documents are returned as a
list.

```python
doc_stream = """
---
doc 1
---
doc 2
"""

parsed_first = parse_yaml(doc_stream)
parsed_all = parse_yaml(doc_stream, multi=True)
assert (parsed_first, parsed_all) == ("doc 1", ["doc 1", "doc 2"])
```

### Writing multiple documents

`write_yaml()` and `format_yaml()` default to a single document. With
`multi=True`, the value must be a sequence of documents and the output
uses `---` between documents and `...` after the final one. For single
documents, `write_yaml()` always wraps the body with `---` and a final
`...`, while `format_yaml()` returns just the body.

```python
from yaml12 import format_yaml, write_yaml

docs = ["first", "second"]
text = format_yaml(docs, multi=True)
assert text.startswith("---") and text.rstrip().endswith("...")
write_yaml(docs, path="out.yml", multi=True)
```

When `multi=False`, parsing stops after the first document, even if later
content is not valid YAML. That makes it easy to extract front matter
from files that mix YAML with other text (like Markdown).

```python
rmd_lines = [
    "---",
    "title: Front matter only",
    "params:",
    "  answer: 42",
    "---",
    "# Body that is not YAML",
]
frontmatter = parse_yaml(rmd_lines)
assert frontmatter == {"title": "Front matter only", "params": {"answer": 42}}
```

## Writing YAML with tags

Attach `Yaml` to values before calling `format_yaml()` or `write_yaml()`
to emit a tag.

```python
from yaml12 import Yaml, write_yaml

tagged = Yaml("1 + x", "!expr")
write_yaml(tagged)
# stdout:
# ---
# !expr 1 + x
# ...
```

Tagged collections or mapping keys work the same way:

```python
from yaml12 import Yaml, format_yaml, parse_yaml

mapping = {Yaml(["a", "b"], "!pair"): "v"}
encoded = format_yaml(mapping)
reparsed = parse_yaml(encoded)
key = next(iter(reparsed))
assert key.tag == "!pair" and key.value == ["a", "b"]
```

## Serializing custom Python objects

You can opt into rich types by tagging your own objects on emit and
supplying a handler on parse. Here is a round-trip for a dataclass:

```python
from dataclasses import dataclass, asdict
from yaml12 import Yaml, format_yaml, parse_yaml

@dataclass
class Server:
    name: str
    host: str
    port: int

def encode_server(server: Server) -> Yaml:
    return Yaml(asdict(server), "!server")

def decode_server(value):
    return Server(**value)

servers = [Server("api", "api.example.com", 8000), Server("db", "db.local", 5432)]
yaml_text = format_yaml([encode_server(s) for s in servers])

round_tripped = parse_yaml(yaml_text, handlers={"!server": decode_server})
assert round_tripped == servers
```

By keeping the on-disk representation a plain mapping plus tag, you get a
stable YAML format while still round-tripping your Python types losslessly.

## Anchors

Anchors (`&id`) name a node; aliases (`*id`) copy it. `yaml12` resolves
aliases before returning Python objects.

```python
from yaml12 import parse_yaml

parsed = parse_yaml("""
recycle-me: &anchor-name
  a: b
  c: d

recycled:
  - *anchor-name
  - *anchor-name
""")

first, second = parsed["recycled"]
assert first["a"] == "b" and second["c"] == "d"
```

## (Very) advanced tags

### Tag directives (`%TAG`)

YAML lets you declare tag handles at the top of a document. The syntax
is `%TAG !<name>! <handle>` and it applies to the rest of the document.

```python
text = """
%TAG !e! tag:example.com,2024:widgets/
---
item: !e!gizmo foo
"""
parsed = parse_yaml(text)
assert parsed["item"].tag == "tag:example.com,2024:widgets/gizmo"
```

You can also declare a global tag prefix, which expands a bare `!`:

```python
text = """
%TAG ! tag:example.com,2024:widgets/
---
item: !gizmo foo
"""
assert parse_yaml(text)["item"].tag == "tag:example.com,2024:widgets/gizmo"
```

### Tag URIs

To bypass handle resolution, use `!<...>` with a valid URI-like string:

```python
parsed = parse_yaml("""
%TAG ! tag:example.com,2024:widgets/
---
item: !<gizmo> foo
""")
assert parsed["item"].tag == "gizmo"
```

### Core schema tags

Tags beginning with `!!` resolve against the YAML core schema handle
(`tag:yaml.org,2002:`). Scalar core tags (`!!str`, `!!int`, `!!float`,
`!!bool`, `!!null`, `!!seq`, `!!map`) add no information when emitting
and are normalized to plain Python values. Informative core tags
(`!!timestamp`, `!!binary`, `!!set`, `!!omap`, `!!pairs`) stay tagged so
you can decide how to handle them.

```python
from yaml12 import Yaml, parse_yaml

yaml_text = """
- !!timestamp 2025-01-01
- !!timestamp 2025-01-01 21:59:43.10 -5
- !!binary UiBpcyBBd2Vzb21l
"""
parsed = parse_yaml(yaml_text)
assert all(isinstance(item, Yaml) for item in parsed) and (
    parsed[0].tag,
    parsed[2].tag,
) == ("tag:yaml.org,2002:timestamp", "tag:yaml.org,2002:binary")
```

Handlers can convert these to richer Python types:

```python
import base64
from datetime import datetime, timezone
from yaml12 import parse_yaml

def ts_handler(value):
    return datetime.fromisoformat(str(value).replace("Z", "+00:00")).replace(tzinfo=timezone.utc)

def binary_handler(value):
    return base64.b64decode(str(value))

converted = parse_yaml(
    yaml_text,
    handlers={
        "tag:yaml.org,2002:timestamp": ts_handler,
        "tag:yaml.org,2002:binary": binary_handler,
    },
)
assert isinstance(converted[0], datetime) and isinstance(converted[2], (bytes, bytearray))
```
