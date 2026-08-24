from __future__ import annotations

import inspect
import math
import random

import pytest

import yaml12


def assert_scalar_emission(value: str, expected: str) -> None:
    encoded = yaml12.format_yaml(value, width=None)

    assert encoded == expected
    assert yaml12.parse_yaml(encoded) == value

    encoded_key = yaml12.format_yaml({value: 1}, width=None)
    assert encoded_key == f"{expected}: 1"
    assert yaml12.parse_yaml(encoded_key) == {value: 1}


def test_yaml_formatting_defaults_to_integer_width() -> None:
    for function in [yaml12.format_yaml, yaml12.write_yaml]:
        default = inspect.signature(function).parameters["width"].default
        assert type(default) is int
        assert default == 80


@pytest.mark.parametrize(
    "value",
    [
        "yes",
        "No",
        "on",
        "OFF",
        "don't",
        'say "hi"',
        r"a\b",
        "a,b",
        "f[0]",
        "x{1}",
        "foo#bar",
        "foo:bar",
        "-x",
        "?x",
        ":x",
        ".gitignore",
        "0xg",
        "0o9",
        "1_000",
        "inf",
        "nan",
    ],
)
def test_format_yaml_emits_yaml_12_plain_strings(value: str) -> None:
    assert_scalar_emission(value, value)


@pytest.mark.parametrize(
    "value",
    [
        "~",
        "null",
        "NULL",
        "true",
        "False",
        "12",
        "+7",
        "-3",
        "0x1F",
        "0o17",
        "3.5",
        "-2e10",
        ".5",
        ".inf",
        "-.Inf",
        ".NaN",
        "0x8000000000000000",
        "0xFFFFFFFFFFFFFFFF",
        "0o1000000000000000000000",
        "10000000000000000000000000000000000000000",
    ],
)
def test_format_yaml_quotes_core_schema_strings(value: str) -> None:
    assert_scalar_emission(value, f'"{value}"')


@pytest.mark.parametrize(
    "value",
    [
        "",
        " x",
        "x ",
        "- x",
        "-",
        "?",
        ":",
        ": x",
        "foo: bar",
        "foo:",
        "a #b",
        "[x",
        "]x",
        ",x",
        "#x",
        "&x",
        "*x",
        "!x",
        "|x",
        ">x",
        "'x",
        "%x",
        "@x",
        "`x",
        "---",
        "--- x",
        "... x",
        "\ufeffx",
    ],
)
def test_format_yaml_quotes_structurally_unsafe_strings(value: str) -> None:
    assert_scalar_emission(value, f'"{value}"')


def test_format_yaml_wraps_long_strings_losslessly() -> None:
    value = " ".join(["word"] * 30)
    obj = {"outer": {"body": value}}

    encoded = yaml12.format_yaml(obj)

    assert "body: >-\n" in encoded
    assert all(len(line) <= 80 for line in encoded.splitlines())
    assert yaml12.parse_yaml(encoded) == obj


def test_format_yaml_width_controls_wrapping() -> None:
    value = "alpha beta gamma delta epsilon"

    encoded = yaml12.format_yaml({"body": value}, width=20)
    assert encoded == "body: >-\n  alpha beta gamma\n  delta epsilon"
    assert yaml12.parse_yaml(encoded) == {"body": value}

    encoded = yaml12.format_yaml({"body": value}, width=20.0)
    assert encoded == "body: >-\n  alpha beta gamma\n  delta epsilon"

    encoded = yaml12.format_yaml({"body": value}, width=20.9)
    assert encoded == "body: >-\n  alpha beta gamma\n  delta epsilon"

    encoded = yaml12.format_yaml({"body": value}, width=None)
    assert encoded == "body: alpha beta gamma delta epsilon"

    encoded = yaml12.format_yaml({"body": value}, width=2**32)
    assert encoded == "body: alpha beta gamma delta epsilon"

    encoded = yaml12.format_yaml({"body": value}, width=2**63 - 1)
    assert encoded == "body: alpha beta gamma delta epsilon"

    with pytest.raises(OverflowError):
        yaml12.format_yaml({"body": value}, width=2**63)


@pytest.mark.parametrize("width", [0, -1])
def test_format_yaml_rejects_invalid_integer_widths(width: int) -> None:
    with pytest.raises(ValueError, match="must be >= 1, or None"):
        yaml12.format_yaml({"key": "value"}, width=width)


@pytest.mark.parametrize("width", [0.0, 0.9, -1.0])
def test_format_yaml_rejects_invalid_float_widths(width: float) -> None:
    with pytest.raises(ValueError, match="width"):
        yaml12.format_yaml({"key": "value"}, width=width)


@pytest.mark.parametrize("width", [math.inf, -math.inf, math.nan])
def test_format_yaml_non_finite_widths_disable_wrapping(width: float) -> None:
    value = {"body": "alpha beta gamma delta epsilon"}

    assert yaml12.format_yaml(value, width=width) == (
        "body: alpha beta gamma delta epsilon"
    )


def test_format_yaml_preserves_paragraph_and_line_oriented_multiline_strings() -> None:
    paragraph = "alpha beta gamma delta epsilon"
    paragraphs = f"{paragraph}\n\n{paragraph}"

    encoded = yaml12.format_yaml({"body": paragraphs}, width=20)
    assert encoded == (
        "body: >-\n"
        "  alpha beta gamma\n"
        "  delta epsilon\n\n\n"
        "  alpha beta gamma\n"
        "  delta epsilon"
    )
    assert yaml12.parse_yaml(encoded) == {"body": paragraphs}

    lines = f"{paragraph}\n{paragraph}"
    encoded = yaml12.format_yaml({"body": lines}, width=20)
    assert encoded == f"body: |-\n  {paragraph}\n  {paragraph}"
    assert yaml12.parse_yaml(encoded) == {"body": lines}


def test_format_yaml_preserves_leading_whitespace_in_literal_blocks() -> None:
    value = "  indented\nnext"

    encoded = yaml12.format_yaml({"body": value})

    assert encoded == "body: |2-\n    indented\n  next"
    assert yaml12.parse_yaml(encoded) == {"body": value}


def test_format_yaml_emits_empty_literal_lines_without_indentation() -> None:
    value = "alpha\n\nomega"

    encoded = yaml12.format_yaml({"body": value})

    assert encoded == "body: |-\n  alpha\n\n  omega"
    assert not any(line.endswith(" ") for line in encoded.splitlines())
    assert yaml12.parse_yaml(encoded) == {"body": value}


def test_format_yaml_quotes_all_newline_values_in_nested_mappings() -> None:
    value = {"result": {"text": "\n", "isError": False}}

    encoded = yaml12.format_yaml(value)

    assert encoded == 'result:\n  text: "\\n"\n  isError: false'
    assert yaml12.parse_yaml(encoded) == value


def test_format_yaml_indents_root_document_markers_in_literal_blocks() -> None:
    value = "foo\n---\nbar"

    encoded = yaml12.format_yaml(value, width=None)

    assert encoded == "|-\n  foo\n  ---\n  bar"
    assert yaml12.parse_yaml(encoded) == value


def test_format_yaml_uses_explicit_syntax_for_overlong_mapping_keys() -> None:
    for character in ["x", "漢"]:
        overlong = character * 1025
        at_limit = character * 1024

        encoded = yaml12.format_yaml({overlong: "payload"}, width=20)
        assert encoded.startswith("? ")
        assert yaml12.parse_yaml(encoded) == {overlong: "payload"}

        encoded = yaml12.format_yaml({at_limit: "payload"}, width=20)
        assert not encoded.startswith("? ")
        assert yaml12.parse_yaml(encoded) == {at_limit: "payload"}


@pytest.mark.parametrize(
    ("value", "expected"),
    [(math.inf, ".Inf"), (-math.inf, "-.Inf"), (math.nan, ".NaN")],
)
def test_format_yaml_uses_canonical_non_finite_spellings(
    value: float, expected: str
) -> None:
    encoded = yaml12.format_yaml(value)

    assert encoded == expected
    reparsed = yaml12.parse_yaml(encoded)
    if math.isnan(value):
        assert math.isnan(reparsed)
    else:
        assert reparsed == value


def test_format_yaml_generated_strings_round_trip() -> None:
    rng = random.Random(20260807)
    tokens = [
        "alpha",
        "beta",
        " ",
        "  ",
        "\t",
        "\n",
        "\n\n",
        "\0",
        "\\",
        '"',
        ":",
        "#",
        "-",
        "?",
        "---",
        "...",
        "é",
        "漢",
        "🙂",
        "\u00a0",
        "\u2028",
    ]
    widths = [1, 5, 20, 80, None]

    for i in range(2_000):
        value = "".join(rng.choice(tokens) for _ in range(rng.randrange(12)))
        objects = [
            value,
            [value],
            {"value": value},
            {"outer": [{"body": value}]},
            {value: "payload"},
        ]
        obj = objects[i % len(objects)]

        encoded = yaml12.format_yaml(obj, width=widths[i % len(widths)])

        assert yaml12.parse_yaml(encoded) == obj
