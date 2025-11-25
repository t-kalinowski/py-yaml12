from __future__ import annotations

import base64
import math
import textwrap
from typing import Callable
from pathlib import Path
import tempfile

import pytest

import yaml12
from yaml12 import Tagged
import base64


def test_format_yaml_round_trip_nested_structures():
    obj = {"foo": "bar", "baz": [True, 123], "qux": {"sub": ["nested", None]}}

    encoded = yaml12.format_yaml(obj)

    assert isinstance(encoded, str)
    assert yaml12.parse_yaml(encoded) == obj


def test_format_yaml_preserves_tagged_values():
    obj = Tagged(
        {
            "scalar": Tagged("bar", "!expr"),
            "seq": Tagged([1, 2], "!seq"),
        },
        "!custom",
    )

    encoded = yaml12.format_yaml(obj)

    assert "!custom" in encoded
    assert "!expr" in encoded
    assert "!seq" in encoded

    reparsed = yaml12.parse_yaml(encoded)
    assert isinstance(reparsed, Tagged)
    assert reparsed.tag == "!custom"
    assert isinstance(reparsed.value["scalar"], Tagged)
    assert reparsed.value["scalar"].tag == "!expr"
    assert isinstance(reparsed.value["seq"], Tagged)
    assert reparsed.value["seq"].tag == "!seq"


def test_format_yaml_preserves_binary_tags():
    payload = base64.b64encode(b"hello world").decode("ascii")
    tagged = Tagged(payload, "!!binary")

    out = yaml12.format_yaml(tagged)

    assert out.startswith("!!binary ")

    reparsed = yaml12.parse_yaml(out)
    assert isinstance(reparsed, Tagged)
    assert reparsed.tag == "tag:yaml.org,2002:binary"
    assert reparsed.value == payload


def test_format_yaml_ignores_core_schema_handles():
    obj = Tagged(
        {
            "scalar": Tagged("bar", "!!seq"),
            "seq": Tagged([1, 2], "!!map"),
        },
        "!custom",
    )

    encoded = yaml12.format_yaml(obj)

    assert "!!seq" not in encoded
    assert "!!map" not in encoded
    assert "!seq" not in encoded
    assert "!map" not in encoded
    assert "!custom" in encoded

    reparsed = yaml12.parse_yaml(encoded)
    assert isinstance(reparsed, Tagged)
    assert reparsed.tag == "!custom"
    assert not isinstance(reparsed.value["scalar"], Tagged)
    assert not isinstance(reparsed.value["seq"], Tagged)


def test_format_yaml_round_trips_multi_document_streams():
    docs = [{"foo": 1}, {"bar": [2, None]}]

    encoded = yaml12.format_yaml(docs, multi=True)

    expected = textwrap.dedent(
        """\
        ---
        foo: 1
        ---
        bar:
          - 2
          - ~
        ...
        """
    )

    assert encoded == expected
    assert yaml12.parse_yaml(encoded, multi=True) == docs


def test_format_yaml_rejects_non_sequence_for_multi():
    with pytest.raises(TypeError, match="`value` must be a sequence when `multi=True`"):
        yaml12.format_yaml({"foo": 1}, multi=True)


def test_format_yaml_single_doc_has_no_header_or_trailing_newline():
    encoded = yaml12.format_yaml({"foo": 1})

    assert not encoded.startswith("---")
    assert not encoded.endswith("\n")
    assert yaml12.parse_yaml(encoded) == {"foo": 1}


def test_format_yaml_validates_tagged_inputs():
    with pytest.raises(ValueError, match="tag must not be empty"):
        yaml12.format_yaml(Tagged("value", ""))

    with pytest.raises(ValueError, match="invalid YAML tag"):
        yaml12.format_yaml(Tagged("value", "abc"))

    with pytest.raises(TypeError):
        yaml12.format_yaml(Tagged("value", 123))  # type: ignore[arg-type]


def test_format_yaml_multi_requires_sequence_argument():
    with pytest.raises(TypeError):
        yaml12.format_yaml(1, multi=True)


def test_format_yaml_round_trips_non_string_keys():
    value = {1: "a", None: "c", 3.5: "d"}

    encoded = yaml12.format_yaml(value)
    reparsed = yaml12.parse_yaml(encoded)

    assert reparsed == value


def test_format_yaml_returns_string_and_round_trips():
    obj = {"answer": 42}
    out = yaml12.format_yaml(obj)

    assert isinstance(out, str)
    assert yaml12.parse_yaml(out) == obj


def test_format_yaml_preserves_single_length_collections():
    seq_out = yaml12.format_yaml([[1]])
    reparsed_seq = yaml12.parse_yaml(seq_out)
    assert reparsed_seq == [[1]]

    map_out = yaml12.format_yaml([{"key": 1}])
    reparsed_map = yaml12.parse_yaml(map_out)
    assert reparsed_map == [{"key": 1}]


def test_format_yaml_preserves_empty_and_null_keys():
    obj = {"a": 1, "": 2, None: 3}

    encoded = yaml12.format_yaml(obj)
    reparsed = yaml12.parse_yaml(encoded)

    assert reparsed["a"] == 1
    assert "" in reparsed and reparsed[""] == 2
    assert None in reparsed and reparsed[None] == 3


def test_format_yaml_rejects_invalid_tag_strings():
    with pytest.raises(ValueError, match="invalid YAML tag"):
        yaml12.format_yaml(Tagged("value", "!!"))


def test_parse_yaml_scalars():
    assert yaml12.parse_yaml("null") is None
    assert yaml12.parse_yaml("123") == 123
    assert yaml12.parse_yaml("true") is True
    assert yaml12.parse_yaml("hello") == "hello"


def test_parse_yaml_sequences_and_mappings():
    simple_seq = textwrap.dedent(
        """\
        - a
        - b
        - c
        """
    )
    mapping_text = textwrap.dedent(
        """\
        foo: 1
        bar: baz
        """
    )

    assert yaml12.parse_yaml(simple_seq) == ["a", "b", "c"]
    assert yaml12.parse_yaml(mapping_text) == {"foo": 1, "bar": "baz"}
    assert yaml12.parse_yaml(["foo: 1", "bar: 2"]) == {"foo": 1, "bar": 2}

    with pytest.raises(TypeError, match="must contain only strings"):
        yaml12.parse_yaml(["foo: 1", None])


def test_parse_yaml_multiple_documents():
    yaml = textwrap.dedent(
        """\
        ---
        foo: 1
        ---
        bar: 2
        """
    )

    assert yaml12.parse_yaml(yaml) == {"foo": 1}
    assert yaml12.parse_yaml(yaml, multi=True) == [{"foo": 1}, {"bar": 2}]


def test_parse_yaml_ignores_later_document_errors_when_not_multi():
    yaml = textwrap.dedent(
        """\
        ---
        foo: 1
        ...
        ---
        not: [valid
        """
    )
    assert yaml12.parse_yaml(yaml) == {"foo": 1}
    with pytest.raises(ValueError, match="YAML parse error"):
        yaml12.parse_yaml(yaml, multi=True)


def test_parse_yaml_errors_on_none_inputs():
    with pytest.raises(
        TypeError, match="`text` must be a string, a sequence of strings, or an object with .read"
    ):
        yaml12.parse_yaml(None)

    with pytest.raises(TypeError, match="must contain only strings"):
        yaml12.parse_yaml([None])

    with pytest.raises(TypeError, match="must contain only strings"):
        yaml12.parse_yaml([None, "foo: 1"])

    with pytest.raises(TypeError, match="must contain only strings"):
        yaml12.parse_yaml(["foo: 1", None])

    assert yaml12.parse_yaml([]) is None


def test_parse_yaml_accepts_text_connection():
    path = Path(tempfile.mkstemp(prefix="yaml12-text-")[1])
    path.write_text("foo: 1\nbar: true\n", encoding="utf-8")
    with path.open("r", encoding="utf-8") as fh:
        assert yaml12.parse_yaml(fh) == {"foo": 1, "bar": True}
    path.unlink()


def test_parse_yaml_accepts_bytes_connection_and_validates_utf8():
    path = Path(tempfile.mkstemp(prefix="yaml12-bytes-")[1])
    path.write_bytes(b"foo: 1\n")
    with path.open("rb") as fh:
        assert yaml12.parse_yaml(fh) == {"foo": 1}

    bad_path = Path(tempfile.mkstemp(prefix="yaml12-badbytes-")[1])
    bad_path.write_bytes(b"a\xff")
    with bad_path.open("rb") as fh_bad, pytest.raises(ValueError, match="UTF-8"):
        yaml12.parse_yaml(fh_bad)
    path.unlink()
    bad_path.unlink()


def test_parse_yaml_connection_empty_respects_multi_flag():
    path = Path(tempfile.mkstemp(prefix="yaml12-empty-")[1])
    path.write_text("", encoding="utf-8")
    with path.open("r", encoding="utf-8") as fh:
        assert yaml12.parse_yaml(fh) is None
    with path.open("r", encoding="utf-8") as fh_multi:
        assert yaml12.parse_yaml(fh_multi, multi=True) == []
    path.unlink()


def test_parse_yaml_streaming_large_chunk_size_one():
    items = list(range(5000))
    yaml_text = "".join(f"- {i}\n" for i in items)
    path = Path(tempfile.mkstemp(prefix="yaml12-chunk1-")[1])
    path.write_text(yaml_text, encoding="utf-8")
    reader = _ChunkReader(path, chunk_size=1)
    try:
        parsed = yaml12.parse_yaml(reader)
        assert parsed == items
    finally:
        reader.close()
        path.unlink()


def test_parse_yaml_streaming_multi_documents():
    yaml_text = "---\nfoo: 1\n---\nbar: [1, 2, 3]\n"
    path = Path(tempfile.mkstemp(prefix="yaml12-chunk-multi-")[1])
    path.write_text(yaml_text, encoding="utf-8")
    reader = _ChunkReader(path, chunk_size=2)
    try:
        parsed = yaml12.parse_yaml(reader, multi=True)
        assert parsed == [{"foo": 1}, {"bar": [1, 2, 3]}]
    finally:
        reader.close()
        path.unlink()


def test_parse_yaml_streaming_read_error_propagates():
    with pytest.raises(RuntimeError, match="boom stream"):
        yaml12.parse_yaml(_ErroringReader())


def test_parse_yaml_streaming_bad_type_error():
    with pytest.raises(TypeError, match="str or bytes"):
        yaml12.parse_yaml(_BadTypeReader())


def test_parse_yaml_streaming_midway_error():
    with pytest.raises(RuntimeError, match="boom later"):
        yaml12.parse_yaml(_ErroringReaderAfterFirst(), multi=True)


class _ChunkReader:
    def __init__(self, path: Path, chunk_size: int = 7):
        self.f = path.open("r", encoding="utf-8")
        self.chunk_size = chunk_size

    def read(self, size: int = -1) -> str:
        chunk_size = self.chunk_size if size < 0 else size
        return self.f.read(chunk_size)

    def close(self):
        self.f.close()


class _ChunkBytesReader:
    def __init__(self, path: Path, chunk_size: int = 5):
        self.f = path.open("rb")
        self.chunk_size = chunk_size

    def read(self, size: int = -1) -> bytes:
        chunk_size = self.chunk_size if size < 0 else size
        return self.f.read(chunk_size)

    def close(self):
        self.f.close()


class _ErroringReader:
    def read(self, size: int = -1):  # noqa: ARG002
        raise RuntimeError("boom stream")


class _BadTypeReader:
    def read(self, size: int = -1):  # noqa: ARG002
        return 123  # not bytes/str


class _ErroringReaderAfterFirst:
    def __init__(self):
        self.calls = 0

    def read(self, size: int = -1):  # noqa: ARG002
        self.calls += 1
        if self.calls == 1:
            return "---\nfoo: 1\n"
        raise RuntimeError("boom later")


def test_parse_numeric_sequences_keep_types():
    yaml = "[1, 2.5, 0x10, .inf, null]"

    parsed = yaml12.parse_yaml(yaml)
    assert parsed == [1, 2.5, 16, math.inf, None]
    assert isinstance(parsed[0], int)
    assert isinstance(parsed[1], float)
    assert isinstance(parsed[2], int)
    assert isinstance(parsed[3], float)


def test_parse_signed_infinities():
    yaml = "[-.inf, +.inf, .INF]"
    parsed = yaml12.parse_yaml(yaml)

    assert parsed == [-math.inf, math.inf, math.inf]


def test_parse_nans():
    yaml = "[.nan, .NaN]"
    parsed = yaml12.parse_yaml(yaml)

    assert len(parsed) == 2
    assert all(math.isnan(x) for x in parsed)


def test_parse_signed_ints_and_float_mix():
    yaml = "[-1, +2, 3.0]"
    parsed = yaml12.parse_yaml(yaml)

    assert parsed == [-1, 2, 3.0]
    assert isinstance(parsed[0], int)
    assert isinstance(parsed[1], int)
    assert isinstance(parsed[2], float)


def test_parse_yaml_handles_trailing_newline():
    yaml = textwrap.dedent(
        """\
        foo: 1
        """
    )
    assert yaml12.parse_yaml(yaml) == {"foo": 1}


def test_parse_yaml_preserves_custom_tags():
    tagged = yaml12.parse_yaml("!custom 3")
    assert isinstance(tagged, Tagged)
    assert tagged.tag == "!custom"
    assert tagged.value == "3"

    nested = yaml12.parse_yaml("values: !seq [1, 2]")
    assert isinstance(nested["values"], Tagged)
    assert nested["values"].tag == "!seq"
    assert nested["values"].value == [1, 2]


def test_parse_yaml_preserves_timestamp_tags():
    yaml = textwrap.dedent(
        """\
        - !!timestamp 2025-01-01
        - !!timestamp 2025-01-01 21:59:43.10 -5
        """
    )
    parsed = yaml12.parse_yaml(yaml)

    assert isinstance(parsed, list)
    assert len(parsed) == 2
    expected_values = ["2025-01-01", "2025-01-01 21:59:43.10 -5"]
    for item, expected in zip(parsed, expected_values):
        assert isinstance(item, Tagged)
        assert item.tag == "tag:yaml.org,2002:timestamp"
        assert item.value == expected


def test_parse_yaml_applies_handlers_to_tagged_nodes():
    handlers: dict[str, Callable[[object], object]] = {
        "!expr": lambda value: eval(str(value)),
        "!wrap": lambda value: {"value": value},
    }

    assert yaml12.parse_yaml("foo: !expr 1+1", handlers=handlers) == {"foo": 2}
    assert yaml12.parse_yaml("items: !wrap [a, b]", handlers=handlers) == {
        "items": {"value": ["a", "b"]}
    }


def test_parse_yaml_applies_handlers_to_mapping_keys():
    text = textwrap.dedent(
        """\
        ? !upper key
        : value
        """
    )
    handlers: dict[str, Callable[[object], object]] = {"!upper": lambda value: str(value).upper()}

    result = yaml12.parse_yaml(text, handlers=handlers)
    assert result == {"KEY": "value"}


def test_parse_yaml_key_handler_receives_tagged_value():
    text = textwrap.dedent(
        """\
        ? !upper key
        : value
        """
    )
    calls: list[str] = []

    def handler(key: str) -> str:
        calls.append(key)
        return f"meta:{key}"

    result = yaml12.parse_yaml(text, handlers={"!upper": handler})

    assert calls == ["key"]
    assert result == {"meta:key": "value"}


def test_parse_yaml_handler_errors_propagate():
    with pytest.raises(RuntimeError, match="boom"):
        yaml12.parse_yaml(
            "foo: !boom bar",
            handlers={"!boom": lambda _: (_ for _ in ()).throw(RuntimeError("boom"))},
        )


def test_parse_yaml_handles_many_handlers():
    tags = [f"!h{idx}" for idx in range(1, 11)]
    calls: dict[str, object] = {}

    def make_handler(tag: str):
        def handler(value: object) -> str:
            calls[tag] = value
            return f"{tag}:{value}"

        return handler

    handlers = {tag: make_handler(tag) for tag in tags}

    result = yaml12.parse_yaml("value: !h9 bar", handlers=handlers)

    assert result == {"value": "!h9:bar"}
    assert calls == {"!h9": "bar"}


def test_parse_yaml_validates_handlers_argument():
    with pytest.raises(TypeError, match="must be a dict"):
        yaml12.parse_yaml("foo: !expr 1", handlers=12)  # type: ignore[arg-type]

    with pytest.raises(TypeError, match="callable"):
        yaml12.parse_yaml("foo: !expr 1", handlers={"!expr": "not a function"})  # type: ignore[arg-type]


def test_parse_yaml_resolves_canonical_null_tags():
    canonical_cases = [
        "!!null ~",
        "!<tag:yaml.org,2002:null> ~",
    ]
    for yaml in canonical_cases:
        parsed = yaml12.parse_yaml(yaml)
        assert parsed is None

    informative_cases = {
        "!<!!null> ~": "!!null",
        "!<!null> ~": "!null",
        "!null ~": "!null",
    }
    for yaml, expected_tag in informative_cases.items():
        parsed = yaml12.parse_yaml(yaml)
        assert isinstance(parsed, Tagged)
        assert parsed.tag == expected_tag
        assert parsed.value == "~"


def test_parse_yaml_errors_on_invalid_canonical_tags():
    with pytest.raises(ValueError):
        yaml12.parse_yaml("!!int foo")
    with pytest.raises(ValueError):
        yaml12.parse_yaml("!!null foo")


def test_parse_yaml_non_string_keys_round_trip():
    yaml = textwrap.dedent(
        """\
        1: a
        2: b
        3.5: c
        string: d
        """
    )
    result = yaml12.parse_yaml(yaml)

    assert result[1] == "a"
    assert result[2] == "b"
    assert result[3.5] == "c"
    assert result["string"] == "d"


def test_parse_yaml_preserves_tagged_mapping_keys():
    yaml = textwrap.dedent(
        """\
        !custom foo: 1
        """
    )

    parsed = yaml12.parse_yaml(yaml)
    assert isinstance(parsed, dict)

    key = next(iter(parsed.keys()))
    assert isinstance(key, Tagged)
    assert key.tag == "!custom"
    assert key.value == "foo"
    assert parsed[key] == 1

    encoded = yaml12.format_yaml(parsed)
    reparsed = yaml12.parse_yaml(encoded)
    assert reparsed == parsed
    reparsed_key = next(iter(reparsed.keys()))
    assert isinstance(reparsed_key, Tagged)
    assert reparsed_key.tag == "!custom"
    assert reparsed_key.value == "foo"


def test_format_and_parse_roundtrip_non_specific_tag():
    tagged = Tagged("value", "!")

    yaml = yaml12.format_yaml(tagged)
    assert yaml.startswith("! ")

    reparsed = yaml12.parse_yaml(yaml)
    assert isinstance(reparsed, Tagged)
    assert reparsed.tag == "!"
    assert reparsed.value == "value"


def test_parse_yaml_canonical_string_tag_forms():
    cases = [
        ("!!str true", "true"),
        ("!str true", Tagged("true", "!str")),
        ("!<str> true", Tagged("true", "str")),
        ("!<!str> true", Tagged("true", "!str")),
        ("!<!!str> true", Tagged("true", "!!str")),
        ("!<tag:yaml.org,2002:str> true", "true"),
    ]

    for yaml, expected in cases:
        parsed = yaml12.parse_yaml(yaml)
        if isinstance(expected, Tagged):
            assert isinstance(parsed, Tagged), f"{yaml} should produce Tagged"
            assert parsed.tag == expected.tag
            assert parsed.value == expected.value
        else:
            assert parsed == expected, f"{yaml} should parse to {expected!r}"


def test_parse_yaml_roundtrip_newline_in_short_string():
    original = {"foo": "bar!\nbar!", "baz": 42}
    round_tripped = yaml12.parse_yaml(yaml12.format_yaml(original))
    assert original == round_tripped


def test_parse_yaml_resolves_anchors_and_aliases():
    yaml = textwrap.dedent(
        """\
        a1: &DEFAULT
          b1: 4
        a2: *DEFAULT
        """
    )
    parsed = yaml12.parse_yaml(yaml)

    assert parsed["a1"]["b1"] == 4
    assert parsed["a2"]["b1"] == 4


def test_parse_yaml_resolves_anchors_inside_sequences():
    yaml = textwrap.dedent(
        """\
        - &A 1
        - 2
        - *A
        """
    )

    parsed = yaml12.parse_yaml(yaml)

    assert parsed == [1, 2, 1]
