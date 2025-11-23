# Custom Tags

YAML 1.2 allows tagging values with custom semantics. `yaml12` preserves those tags by default and lets you plug in your own coercions.

## Tagged wrapper

Non-core tags that do not have a matching handler are wrapped in a frozen `Tagged` dataclass:

```python
from yaml12 import Tagged, parse_yaml

color = parse_yaml("!color red")
assert isinstance(color, Tagged)
assert color.value == "red"
assert color.tag == "!color"
```

`Tagged` works for both scalar and collection nodes, including keys in mappings. You can serialize tagged values by passing a `Tagged` instance back into `format_yaml` or `write_yaml`.

## Handler functions

Provide a `handlers` dictionary to `parse_yaml` or `read_yaml` to intercept specific tags. Keys are tag strings (for example `"!point"` or `"tag:example.com,2024:point"`). Values are callables that receive the already-parsed value and return whatever Python object you want to substitute.

```python
from dataclasses import dataclass
from yaml12 import parse_yaml

@dataclass(frozen=True)
class Point:
    x: int
    y: int

def point_handler(value):
    # value is a regular Python mapping because the inner YAML was parsed first
    return Point(x=value["x"], y=value["y"])

doc = parse_yaml("vertex: !point\n  x: 1\n  y: 2\n", handlers={"!point": point_handler})
assert doc["vertex"] == Point(1, 2)
```

Handlers apply to both values and keys. If a handler raises an exception, it propagates as-is to help debugging.
