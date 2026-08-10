import yaml12


def test_parse_simple_mapping():
    out = yaml12.parse_yaml("foo: 1\nbar: true")
    assert out == {"foo": 1, "bar": True}


def test_parse_string_keys_fast_path():
    out = yaml12.parse_yaml("a: 1\nb: 2\nc: three")
    assert out == {"a": 1, "b": 2, "c": "three"}


def test_parse_scalar_sequence_fast_path():
    out = yaml12.parse_yaml("- 1\n- 2\n- 3")
    assert out == [1, 2, 3]


def test_parse_empty_is_none():
    assert yaml12.parse_yaml("") is None


def test_multi_roundtrip():
    docs = ["first", "second"]
    text = yaml12.format_yaml(docs, multi=True)
    assert text == "---\nfirst\n---\nsecond\n"
    parsed = yaml12.parse_yaml(text, multi=True)
    assert parsed == docs


def test_tagged_roundtrip():
    tagged = yaml12.Yaml(5, "!custom")
    text = yaml12.format_yaml(tagged)
    assert text.startswith("!custom")
    reparsed = yaml12.parse_yaml(text)
    assert isinstance(reparsed, yaml12.Yaml)
    assert reparsed.tag == "!custom"
    assert reparsed.value == "5"
