from __future__ import annotations

import textwrap
from pathlib import Path
import io

import pytest

import yaml12
from tests_py.test_format_parse import (
    _BadTypeReader,
    _ChunkBytesReader,
    _ChunkReader,
    _ErroringReader,
    _ErroringReaderAfterFirst,
)


def test_write_and_read_single_document(tmp_path: Path):
    path = tmp_path / "yaml12-single.yaml"
    value = {"alpha": 1, "nested": [True, None]}

    body = yaml12.format_yaml(value)
    yaml12.write_yaml(value, str(path))

    expected = f"---\n{body}\n...\n"
    assert path.exists()
    assert path.read_text(encoding="utf-8") == expected
    assert yaml12.read_yaml(str(path)) == value


def test_write_yaml_defaults_to_stdout_when_path_is_none(capfd: pytest.CaptureFixture[str]):
    value = {"alpha": 1, "nested": [True, None]}
    encoded = yaml12.format_yaml(value)

    yaml12.write_yaml(value, path=None)
    output = capfd.readouterr().out

    expected = f"---\n{encoded}\n...\n"

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


def test_read_yaml_accepts_text_connection():
    buf = io.StringIO("foo: 1\nbar: true")
    parsed = yaml12.read_yaml(buf)
    assert parsed == {"foo": 1, "bar": True}


def test_read_yaml_accepts_bytes_connection_and_validates_utf8():
    buf = io.BytesIO(b"foo: 1\nbar: true\n")
    parsed = yaml12.read_yaml(buf)
    assert parsed == {"foo": 1, "bar": True}

    bad = io.BytesIO(b"a\xff")
    with pytest.raises(ValueError, match="UTF-8"):
        yaml12.read_yaml(bad)


def test_read_yaml_streaming_chunks_str(tmp_path: Path):
    items = list(range(1200))
    yaml_text = "".join(f"- {i}\n" for i in items)
    path = tmp_path / "chunks-str.yaml"
    path.write_text(yaml_text, encoding="utf-8")
    reader = _ChunkReader(path, chunk_size=4)
    try:
        parsed = yaml12.read_yaml(reader)
        assert parsed == items
    finally:
        reader.close()


def test_read_yaml_streaming_chunks_bytes(tmp_path: Path):
    yaml_text = "alpha: 1\nbeta: true\ngamma: [1, 2, 3]\n"
    path = tmp_path / "chunks-bytes.yaml"
    path.write_bytes(yaml_text.encode("utf-8"))
    reader = _ChunkBytesReader(path, chunk_size=3)
    try:
        parsed = yaml12.read_yaml(reader)
        assert parsed == {"alpha": 1, "beta": True, "gamma": [1, 2, 3]}
    finally:
        reader.close()


def test_read_yaml_streaming_read_error_propagates():
    with pytest.raises(RuntimeError, match="boom stream"):
        yaml12.read_yaml(_ErroringReader())


def test_read_yaml_streaming_bad_type_error():
    with pytest.raises(TypeError, match="str or bytes"):
        yaml12.read_yaml(_BadTypeReader())


def test_read_yaml_streaming_midway_error():
    with pytest.raises(RuntimeError, match="boom later"):
        yaml12.read_yaml(_ErroringReaderAfterFirst(), multi=True)


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


def test_write_yaml_accepts_text_writer(tmp_path: Path):
    path = tmp_path / "writer-text.yaml"
    handle = path.open("w", encoding="utf-8")
    value = {"foo": 1}
    yaml12.write_yaml(value, handle)
    handle.close()
    assert path.read_text(encoding="utf-8") == f"---\n{yaml12.format_yaml(value)}\n...\n"


def test_write_yaml_accepts_bytes_writer(tmp_path: Path):
    path = tmp_path / "writer-bytes.yaml"
    handle = path.open("wb")
    value = {"foo": 1}
    yaml12.write_yaml(value, handle)
    handle.close()
    assert path.read_text(encoding="utf-8") == f"---\n{yaml12.format_yaml(value)}\n...\n"


def test_write_yaml_bytes_writer_after_str_failure():
    class BytesOnly:
        def __init__(self):
            self.data = bytearray()
            self.calls = 0

        def write(self, payload):
            self.calls += 1
            if isinstance(payload, str):
                raise TypeError("need bytes")
            assert isinstance(payload, (bytes, bytearray))
            self.data.extend(payload)

    sink = BytesOnly()
    value = {"foo": 1}
    yaml12.write_yaml(value, sink)
    assert sink.calls == 2  # first str write fails, second bytes write succeeds
    expected = f"---\n{yaml12.format_yaml(value)}\n...\n".encode()
    assert bytes(sink.data) == expected


def test_write_yaml_multi_to_custom_writer():
    class StringSink:
        def __init__(self):
            self.parts = []

        def write(self, payload: str):
            self.parts.append(payload)

    sink = StringSink()
    docs = [{"foo": 1}, {"bar": [2, None]}]
    yaml12.write_yaml(docs, sink, multi=True)
    out = "".join(sink.parts)
    assert out == yaml12.format_yaml(docs, multi=True)


def test_write_yaml_writer_error_propagates(tmp_path: Path):
    class BadWriter:
        def write(self, data):
            raise RuntimeError("boom write")

    with pytest.raises(RuntimeError, match="boom write"):
        yaml12.write_yaml({"foo": 1}, BadWriter())


def test_read_yaml_errors_on_non_utf8_input(tmp_path: Path):
    path = tmp_path / "latin1.yaml"
    path.write_bytes(bytes([0x61, 0xE9, 0x0A]))  # "a\xE9\n" is invalid UTF-8

    with pytest.raises(OSError, match="valid UTF-8"):
        yaml12.read_yaml(str(path))

    with pytest.raises(OSError, match="valid UTF-8"):
        yaml12.read_yaml(str(path), multi=True)
