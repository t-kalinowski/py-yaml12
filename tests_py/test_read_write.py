from __future__ import annotations

from pathlib import Path
import textwrap

import pytest

import yaml12


def test_write_and_read_single_document(tmp_path: Path):
    path = tmp_path / "yaml12-single.yaml"
    value = {"alpha": 1, "nested": [True, None]}

    body = yaml12.format_yaml(value)
    yaml12.write_yaml(value, str(path))

    assert path.exists()
    file_lines = path.read_text(encoding="utf-8").splitlines()
    body_lines = body.splitlines()

    assert file_lines[0] == "---"
    assert file_lines[-1] == "..."
    assert file_lines[1 : 1 + len(body_lines)] == body_lines
    assert yaml12.read_yaml(str(path)) == value


def test_write_yaml_defaults_to_stdout_when_path_is_none(capfd: pytest.CaptureFixture[str]):
    value = {"alpha": 1, "nested": [True, None]}
    encoded = yaml12.format_yaml(value)

    yaml12.write_yaml(value, path=None)
    output = capfd.readouterr().out

    expected = textwrap.dedent(
        f"""\
        ---
        {encoded}
        ...
        """
    )

    assert output == expected
    assert yaml12.parse_yaml(output) == value


def test_write_and_read_multi_document_streams(tmp_path: Path):
    path = tmp_path / "yaml12-multi.yaml"
    docs = [{"foo": 1}, {"bar": [2, None]}]

    encoded = yaml12.format_yaml(docs, multi=True)
    yaml12.write_yaml(docs, str(path), multi=True)

    assert path.read_text(encoding="utf-8") == encoded
    assert yaml12.parse_yaml(encoded, multi=True) == docs
    assert yaml12.read_yaml(str(path), multi=True) == docs


def test_write_yaml_flushes_final_newline_for_files(tmp_path: Path):
    path = tmp_path / "yaml12-flush.yaml"
    value = {"foo": 1}

    yaml12.write_yaml(value, str(path))
    path.read_text(encoding="utf-8")  # should not raise
    assert yaml12.read_yaml(str(path)) == value


def test_write_yaml_preserves_multiline_strings(tmp_path: Path):
    path = tmp_path / "yaml12-multiline.yaml"
    multilines = {"tail": "line1\nline2\n"}

    yaml12.write_yaml(multilines, str(path))

    assert yaml12.read_yaml(str(path)) == multilines
    expected = textwrap.dedent(
        """\
        ---
        tail: |
          line1
          line2
        ...
        """
    )
    assert path.read_text(encoding="utf-8") == expected


def test_read_yaml_errors_when_file_missing(tmp_path: Path):
    missing = tmp_path / "does-not-exist.yaml"
    with pytest.raises(OSError, match="failed to read"):
        yaml12.read_yaml(str(missing))


def test_read_yaml_does_not_simplify_mixed_type_sequences(tmp_path: Path):
    path = tmp_path / "mixed-types.yaml"
    path.write_text(
        textwrap.dedent(
            """\
            - true
            - 1
            """
        ),
        encoding="utf-8",
    )

    result = yaml12.read_yaml(str(path))

    assert isinstance(result, list)
    assert result == [True, 1]


def test_read_yaml_keeps_tagged_sequence_elements(tmp_path: Path):
    path = tmp_path / "tagged-seq.yaml"
    path.write_text(
        textwrap.dedent(
            """\
            - !foo 1
            - 2
            """
        ),
        encoding="utf-8",
    )

    result = yaml12.read_yaml(str(path))
    first = result[0]

    assert isinstance(first, yaml12.Tagged)
    assert first.tag == "!foo"
    assert first.value == "1"


def test_read_yaml_handler_errors_propagate(tmp_path: Path):
    path = tmp_path / "handler-error.yaml"
    path.write_text(
        textwrap.dedent(
            """\
            foo: !err value
            """
        ),
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="handler oops"):
        yaml12.read_yaml(
            str(path), handlers={"!err": lambda _: (_ for _ in ()).throw(RuntimeError("handler oops"))}
        )


def test_read_yaml_errors_on_non_utf8_input(tmp_path: Path):
    path = tmp_path / "latin1.yaml"
    path.write_bytes(bytes([0x61, 0xE9, 0x0A]))  # "a\xE9\n" is invalid UTF-8

    with pytest.raises(OSError, match="valid UTF-8"):
        yaml12.read_yaml(str(path))

    with pytest.raises(OSError, match="valid UTF-8"):
        yaml12.read_yaml(str(path), multi=True)
