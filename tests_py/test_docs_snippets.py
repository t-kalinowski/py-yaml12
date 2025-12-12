from __future__ import annotations

import base64
import textwrap
from pathlib import Path
from typing import Callable

import pytest

import yaml12
from yaml12 import Yaml


def test_quick_start_round_trip():
    yaml_text = textwrap.dedent(
        """
        title: A modern YAML parser and emitter written in Rust
        properties: [fast, correct, safe, simple]
        features:
          tags: preserve
          streams: multi
        """
    )
    doc = yaml12.parse_yaml(yaml_text)
    assert doc == {
        "title": "A modern YAML parser and emitter written in Rust",
        "properties": ["fast", "correct", "safe", "simple"],
        "features": {"tags": "preserve", "streams": "multi"},
    }

    round_trip = yaml12.parse_yaml(yaml12.format_yaml(doc))
    assert round_trip == doc


def test_tagged_scalar_parse_and_compare():
    tagged = yaml12.parse_yaml("!expr 1 + 1")
    assert tagged == Yaml(value="1 + 1", tag="!expr")


def test_handlers_transform_values_and_keys(tmp_path: Path):
    yaml_text = textwrap.dedent(
        """
        items:
          - !upper [rust, python]
          - !expr 6 * 7
        """
    )

    handlers: dict[str, Callable[[object], object]] = {
        "!expr": lambda value: eval(value),
        "!upper": lambda value: [x.upper() for x in value],
    }

    parsed = yaml12.parse_yaml(yaml_text, handlers=handlers)
    assert parsed["items"][0] == ["RUST", "PYTHON"]
    assert parsed["items"][1] == 42

    path = tmp_path / "handlers.yaml"
    yaml12.write_yaml(parsed, path)
    assert yaml12.read_yaml(path) == parsed


def test_format_preserves_tags_and_collections():
    obj = {
        "seq": [1, 2],
        "map": {"key": "value"},
        "tagged": Yaml(value="1 + 1", tag="!expr"),
    }

    encoded = yaml12.format_yaml(obj)
    reparsed = yaml12.parse_yaml(encoded)
    assert reparsed == {
        "seq": [1, 2],
        "map": {"key": "value"},
        "tagged": Yaml("1 + 1", "!expr"),
    }


def test_structured_example_matches_expected():
    yaml_text = textwrap.dedent(
        """
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
            - "line\\nbreak"
            - 'quote: ''here'''
          plain: [yes, no]
          mixed: [won't simplify, 123, true]
        """
    )

    parsed = yaml12.parse_yaml(yaml_text)
    assert parsed["doc"]["pets"] == ["cat", "dog"]
    assert parsed["doc"]["numbers"] == [1, 2.5, 16, float("inf"), None]
    assert parsed["doc"]["integers"] == [1, 2, 3, 16, None]
    assert parsed["doc"]["flags"] == {"enabled": True, "label": "on"}
    assert parsed["doc"]["literal"] == "hello\nworld\n"
    assert parsed["doc"]["folded"] == "hello world\n"
    assert parsed["doc"]["quoted"] == ["line\nbreak", "quote: 'here'"]
    assert parsed["doc"]["plain"] == ["yes", "no"]
    assert parsed["doc"]["mixed"] == ["won't simplify", 123, True]


def test_mapping_keys_and_tags_round_trip():
    mapping = {
        "tagged_value": Yaml(["a", "b"], "!pair"),
        Yaml("tagged-key", "!k"): "v",
    }

    encoded = yaml12.format_yaml(mapping)
    reparsed = yaml12.parse_yaml(encoded)

    value = reparsed["tagged_value"]
    assert (
        isinstance(value, Yaml) and value.tag == "!pair" and value.value == ["a", "b"]
    )

    key = next(k for k in reparsed if isinstance(k, Yaml))
    assert key.tag == "!k" and key.value == "tagged-key" and reparsed[key] == "v"


def test_anchor_resolution_and_streams():
    doc_stream = textwrap.dedent(
        """
        ---
        doc 1
        ---
        doc 2
        """
    )

    first = yaml12.parse_yaml(doc_stream)
    all_docs = yaml12.parse_yaml(doc_stream, multi=True)
    assert first == "doc 1"
    assert all_docs == ["doc 1", "doc 2"]

    anchors = textwrap.dedent(
        """
        recycle-me: &anchor-name
          a: b
          c: d

        recycled:
          - *anchor-name
          - *anchor-name
        """
    )
    parsed = yaml12.parse_yaml(anchors)
    assert parsed["recycled"][0]["a"] == "b"
    assert parsed["recycled"][1]["c"] == "d"


def test_tag_directives_and_uri_tags():
    text = textwrap.dedent(
        """
        %TAG !e! tag:example.com,2024:widgets/
        ---
        item: !e!gizmo foo
        """
    )
    parsed = yaml12.parse_yaml(text)
    assert parsed["item"].tag == "tag:example.com,2024:widgets/gizmo"

    text_global = textwrap.dedent(
        """
        %TAG ! tag:example.com,2024:widgets/
        ---
        item: !<gizmo> foo
        """
    )
    parsed = yaml12.parse_yaml(text_global)
    assert parsed["item"].tag == "gizmo"


def test_timestamp_and_binary_tags_with_handlers():
    yaml_text = textwrap.dedent(
        """
        - !!timestamp 2025-01-01
        - !!timestamp 2025-01-01 21:59:43.10-05:00
        - !!binary UiBpcyBBd2Vzb21l
        """
    )
    parsed = yaml12.parse_yaml(yaml_text)
    assert isinstance(parsed[0], Yaml) and parsed[0].tag.endswith("timestamp")
    assert isinstance(parsed[2], Yaml) and parsed[2].tag.endswith("binary")

    def ts_handler(value):
        return str(value) + "Z"

    def binary_handler(value):
        return base64.b64decode(str(value))

    converted = yaml12.parse_yaml(
        yaml_text,
        handlers={
            "tag:yaml.org,2002:timestamp": ts_handler,
            "tag:yaml.org,2002:binary": binary_handler,
        },
    )
    assert converted[0] == "2025-01-01Z"
    assert converted[1].endswith("-05:00Z")
    assert converted[2] == b"R is Awesome"


def test_format_and_parse_multi_document_stream(
    tmp_path: Path, capfd: pytest.CaptureFixture[str]
):
    docs = ["first", "second"]
    text = yaml12.format_yaml(docs, multi=True)
    assert text.startswith("---") and text.rstrip().endswith("...")

    yaml12.write_yaml(docs, path=None, multi=True)
    stdout = capfd.readouterr().out
    assert stdout == f"{text}"

    path = tmp_path / "stream.yaml"
    yaml12.write_yaml(docs, path, multi=True)
    assert yaml12.read_yaml(path, multi=True) == docs


def test_reference_examples_behave():
    assert yaml12.parse_yaml("foo: [1, 2, 3]") == {"foo": [1, 2, 3]}

    stream = textwrap.dedent(
        """
        ---
        first: 1
        ---
        second: 2
        """
    )
    assert yaml12.parse_yaml(stream) == {"first": 1}
    assert yaml12.parse_yaml(stream, multi=True) == [{"first": 1}, {"second": 2}]

    handlers = {"!upper": lambda value: str(value).upper()}
    assert yaml12.parse_yaml("!upper key: !upper value", handlers=handlers) == {
        "KEY": "VALUE"
    }

    lines = [
        "---",
        "title: Front matter only",
        "params:",
        "  answer: 42",
        "---",
        "# Body that is not YAML",
    ]
    assert yaml12.parse_yaml("\n".join(lines)) == {
        "title": "Front matter only",
        "params": {"answer": 42},
    }


def test_format_reference_examples(tmp_path: Path):
    assert yaml12.format_yaml({"foo": 1, "bar": [True, None]})

    docs = [{"foo": 1}, {"bar": [2, None]}]
    multi_text = yaml12.format_yaml(docs, multi=True)
    assert multi_text.startswith("---")

    tagged = Yaml("1 + 1", "!expr")
    assert yaml12.format_yaml(tagged).startswith("!expr 1 + 1")

    path = tmp_path / "example.yaml"
    yaml12.write_yaml({"alpha": 1}, path)
    expected = f"---\n{yaml12.format_yaml({'alpha': 1})}\n...\n"
    assert path.read_text(encoding="utf-8") == expected
