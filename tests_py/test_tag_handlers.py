from __future__ import annotations

import datetime as dt
from pathlib import Path
from typing import Any, Callable, Mapping

import yaml12

TagHandler = Callable[[yaml12.Tagged], Any]


def apply_tag_handlers(obj: Any, handlers: Mapping[str, TagHandler]):
    """Walk a parsed YAML tree and invoke handlers for tagged nodes."""
    if isinstance(obj, yaml12.Tagged):
        inner = apply_tag_handlers(obj.value, handlers)
        tagged = yaml12.Tagged(inner, obj.tag)
        handler = handlers.get(obj.tag)
        return handler(tagged) if handler else tagged
    if isinstance(obj, list):
        return [apply_tag_handlers(item, handlers) for item in obj]
    if isinstance(obj, dict):
        return {
            apply_tag_handlers(k, handlers): apply_tag_handlers(v, handlers)
            for k, v in obj.items()
        }
    return obj


def test_tag_handlers_transform_values_to_domain_types():
    text = """\
workdir: !path /srv/app
schedule:
  - !seconds 1.5
  - wait
  - !seconds 0.25
started_at: !ts 2024-11-22T18:30:00Z
"""
    handlers = {
        "!path": lambda tagged: Path(tagged.value),
        "!seconds": lambda tagged: dt.timedelta(seconds=float(tagged.value)),
        "!ts": lambda tagged: dt.datetime.fromisoformat(
            str(tagged.value).replace("Z", "+00:00")
        ),
    }

    parsed = yaml12.parse_yaml(text)
    converted = apply_tag_handlers(parsed, handlers)

    assert converted["workdir"] == Path("/srv/app")
    assert converted["schedule"][0] == dt.timedelta(seconds=1.5)
    assert converted["schedule"][1] == "wait"
    assert converted["schedule"][2] == dt.timedelta(seconds=0.25)
    assert converted["started_at"] == dt.datetime(
        2024, 11, 22, 18, 30, tzinfo=dt.timezone.utc
    )


def test_tag_handlers_update_keys_and_nested_structures():
    env = {"HOME": "/home/example", "CONF": "/etc/app.conf"}
    text = """\
? !env HOME
: {cfg: !env CONF}
paths:
  !path /tmp/output: done
"""
    handlers = {
        "!env": lambda tagged: env[tagged.value],
        "!path": lambda tagged: Path(tagged.value),
    }

    parsed = yaml12.parse_yaml(text)
    converted = apply_tag_handlers(parsed, handlers)

    assert converted[env["HOME"]] == {"cfg": env["CONF"]}
    assert converted["paths"][Path("/tmp/output")] == "done"
