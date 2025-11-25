from __future__ import annotations

import textwrap

import yaml12


def test_sequence_key_returns_mapping_key():
    result = yaml12.parse_yaml("? [a, b, c]\n: 1\n")

    key = next(iter(result))
    assert isinstance(key, yaml12.MappingKey)
    assert key.value == ["a", "b", "c"]
    assert list(key) == ["a", "b", "c"]
    assert key[1] == "b"
    assert len(key) == 3
    assert result[yaml12.MappingKey(["a", "b", "c"])] == 1


def test_mapping_key_returns_mapping_key():
    result = yaml12.parse_yaml("? {foo: 1, bar: [2]}\n: value\n")

    key = next(iter(result))
    assert isinstance(key, yaml12.MappingKey)
    assert isinstance(key.value, dict)
    assert key.value["foo"] == 1
    assert key["foo"] == 1
    assert list(key) == ["foo", "bar"]
    assert result[yaml12.MappingKey({"foo": 1, "bar": [2]})] == "value"


def test_scalar_keys_remain_plain_types():
    parsed = yaml12.parse_yaml("foo: 1\n2: bar")

    assert parsed["foo"] == 1
    assert parsed[2] == "bar"
    assert not any(isinstance(k, yaml12.MappingKey) for k in parsed)


def test_handler_returning_mapping_is_wrapped():
    text = "? !wrap foo\n: bar"
    handlers = {"!wrap": lambda value: {"key": value}}

    parsed = yaml12.parse_yaml(text, handlers=handlers)
    key = next(iter(parsed))

    assert isinstance(key, yaml12.MappingKey)
    assert key.value == {"key": "foo"}
    assert parsed[yaml12.MappingKey({"key": "foo"})] == "bar"


def test_mapping_key_round_trip_format_and_parse():
    key = yaml12.MappingKey({"foo": [1, 2]})
    original = {key: "value"}

    encoded = yaml12.format_yaml(original)
    reparsed = yaml12.parse_yaml(encoded)
    reparsed_key = next(iter(reparsed))

    assert isinstance(reparsed_key, yaml12.MappingKey)
    assert reparsed_key.value == {"foo": [1, 2]}
    assert reparsed == {yaml12.MappingKey({"foo": [1, 2]}): "value"}
    assert reparsed == original


def test_mapping_key_with_tagged_mapping_proxies_inner_value():
    parsed = yaml12.parse_yaml("? !foo {bar: 1}\n: baz\n")
    key = next(iter(parsed))

    assert isinstance(key, yaml12.MappingKey)
    assert isinstance(key.value, yaml12.Tagged)
    assert key.value.tag == "!foo"
    assert key.value.value == {"bar": 1}
    assert key["bar"] == 1
    assert list(key) == ["bar"]
    assert parsed[yaml12.MappingKey(key.value)] == "baz"


def test_mapping_key_hashes_by_structure():
    k1 = yaml12.MappingKey({"b": 2, "a": 1})
    k2 = yaml12.MappingKey({"a": 1, "b": 2})

    assert k1 == k2
    assert hash(k1) == hash(k2)


def test_mapping_key_with_tagged_value_hashes_and_compares():
    k1 = yaml12.MappingKey(yaml12.Tagged({"a": 1}, "!tag"))
    k2 = yaml12.MappingKey(yaml12.Tagged({"a": 1}, "!tag"))

    assert k1 == k2
    assert hash(k1) == hash(k2)

    mapping = {k1: "value"}
    assert mapping[k2] == "value"


def test_mapping_key_tagged_round_trip_format_and_parse():
    key = yaml12.MappingKey(yaml12.Tagged("foo", "!k"))
    original = {key: "v"}

    encoded = yaml12.format_yaml(original)
    reparsed = yaml12.parse_yaml(encoded)
    reparsed_key = next(iter(reparsed))

    assert isinstance(reparsed_key, yaml12.Tagged)
    assert reparsed_key.tag == "!k"
    assert reparsed_key.value == "foo"
    assert reparsed == {yaml12.Tagged("foo", "!k"): "v"}


def test_collection_values_stay_plain():
    parsed = yaml12.parse_yaml(
        "top:\n"
        "  - [1, 2]\n"
        "  - {foo: bar}\n"
    )

    items = parsed["top"]
    assert items[0] == [1, 2]
    assert isinstance(items[1], dict)
    assert not any(isinstance(k, yaml12.MappingKey) for k in items[1])


def test_tagged_outer_mapping_with_tagged_keys_round_trip():
    yaml_text = "!outer\n!k1 foo: 1\n!k2 bar: 2\n"

    parsed = yaml12.parse_yaml(yaml_text)
    assert isinstance(parsed, yaml12.Tagged)
    assert parsed.tag == "!outer"
    assert isinstance(parsed.value, dict)
    keys = list(parsed.value.keys())
    assert all(isinstance(k, yaml12.Tagged) for k in keys)
    assert {k.tag for k in keys} == {"!k1", "!k2"}
    assert parsed.value[keys[0]] in (1, 2)

    encoded = yaml12.format_yaml(parsed)
    reparsed = yaml12.parse_yaml(encoded)
    assert reparsed == parsed


def test_complex_tagged_and_untagged_mapping_keys_round_trip():
    yaml_text = textwrap.dedent(
        """\
        ? [a, b]
        : plain-seq
        ? {foo: bar}
        : !val {x: 1}
        ? !tagged-key scalar
        : [3, 4]
        ? tagged_value_key
        : !tagged-seq [5, 6]
        """
    )

    parsed = yaml12.parse_yaml(yaml_text)
    assert isinstance(parsed, dict)
    assert len(parsed) == 4

    keys = list(parsed.keys())
    assert any(isinstance(k, yaml12.MappingKey) for k in keys)

    seq_key = yaml12.MappingKey(["a", "b"])
    map_key = yaml12.MappingKey({"foo": "bar"})
    tagged_scalar_key = yaml12.Tagged("scalar", "!tagged-key")
    plain_scalar_key = "tagged_value_key"

    assert parsed[seq_key] == "plain-seq"
    assert isinstance(parsed[map_key], yaml12.Tagged)
    assert parsed[map_key].tag == "!val"
    assert parsed[map_key].value == {"x": 1}
    assert parsed[tagged_scalar_key] == [3, 4]
    tagged_value = parsed[plain_scalar_key]
    assert isinstance(tagged_value, yaml12.Tagged)
    assert tagged_value.tag == "!tagged-seq"
    assert tagged_value.value == [5, 6]

    encoded = yaml12.format_yaml(parsed)
    reparsed = yaml12.parse_yaml(encoded)
    assert reparsed == parsed


def test_tagged_scalar_mapping_key_remains_tagged():
    parsed = yaml12.parse_yaml("!tag key: value\n")

    key = next(iter(parsed))
    assert isinstance(key, yaml12.Tagged)
    assert not isinstance(key, yaml12.MappingKey)
    assert key.tag == "!tag"
    assert key.value == "key"
    assert parsed[key] == "value"


def test_tagged_collection_mapping_key_wraps_with_mapping_key():
    parsed = yaml12.parse_yaml("? !seq [a, b]\n: val\n")

    key = next(iter(parsed))
    assert isinstance(key, yaml12.MappingKey)
    assert isinstance(key.value, yaml12.Tagged)
    assert key.value.tag == "!seq"
    assert key.value.value == ["a", "b"]
    assert parsed[yaml12.MappingKey(yaml12.Tagged(["a", "b"], "!seq"))] == "val"
