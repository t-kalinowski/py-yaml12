from __future__ import annotations

import textwrap
from pathlib import Path
import io
import contextlib
import os
import sys
import pytest

import yaml12


class _ErroringReader:
    def read(self, size: int = -1):  # noqa: ARG002
        raise RuntimeError("boom stream")


class _BadTypeReader:
    def read(self, size: int = -1):  # noqa: ARG002
        return 123  # not bytes/str


def test_write_and_read_single_document(tmp_path: Path):
    path = tmp_path / "yaml12-single.yaml"
    value = {"alpha": 1, "nested": [True, None]}

    body = yaml12.format_yaml(value)
    yaml12.write_yaml(value, str(path))

    expected = f"---\n{body}\n"
    assert path.exists()
    assert path.read_text(encoding="utf-8") == expected
    assert yaml12.read_yaml(str(path)) == value


def test_write_yaml_defaults_to_stdout_when_path_is_none(
    capfd: pytest.CaptureFixture[str],
):
    value = {"alpha": 1, "nested": [True, None]}
    encoded = yaml12.format_yaml(value)

    yaml12.write_yaml(value, path=None)
    output = capfd.readouterr().out

    expected = f"---\n{encoded}\n"

    assert output == expected
    assert yaml12.parse_yaml(output) == value


def test_write_yaml_respects_python_stdout_redirect():
    value = {"alpha": 1, "nested": [True, None]}
    encoded = yaml12.format_yaml(value)
    expected = f"---\n{encoded}\n"

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        yaml12.write_yaml(value, path=None)

    assert buf.getvalue() == expected


def test_write_yaml_stdout_type_error_falls_back_to_real_stdout(
    capfd: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
):
    class BadStdout:
        def write(self, payload):  # noqa: ARG002
            raise TypeError("malformed stdout")

    monkeypatch.setattr(sys, "stdout", BadStdout())

    value = {"alpha": 1, "nested": [True, None]}
    encoded = yaml12.format_yaml(value)
    expected = f"---\n{encoded}\n"

    yaml12.write_yaml(value, path=None)
    output = capfd.readouterr().out
    assert output == expected


def test_write_yaml_stdout_malformed_falls_back_to_real_stdout(
    capfd: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
):
    class BadStdout:
        def write(self, payload):  # noqa: ARG002
            raise RuntimeError("boom write")

        def flush(self):
            raise RuntimeError("boom flush")

    monkeypatch.setattr(sys, "stdout", BadStdout())

    value = {"alpha": 1, "nested": [True, None]}
    encoded = yaml12.format_yaml(value)
    expected = f"---\n{encoded}\n"

    yaml12.write_yaml(value, path=None)
    output = capfd.readouterr().out
    assert output == expected


def test_write_yaml_stdout_none_falls_back_to_real_stdout(
    capfd: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(sys, "stdout", None)

    value = {"alpha": 1}
    encoded = yaml12.format_yaml(value)
    expected = f"---\n{encoded}\n"

    yaml12.write_yaml(value, path=None)
    output = capfd.readouterr().out
    assert output == expected


def test_write_yaml_stdout_missing_falls_back_to_real_stdout(
    capfd: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
):
    # Remove sys.stdout entirely so PySys_GetObject returns NULL.
    monkeypatch.delattr(sys, "stdout", raising=False)

    value = {"alpha": 1}
    encoded = yaml12.format_yaml(value)
    expected = f"---\n{encoded}\n"

    yaml12.write_yaml(value, path=None)
    output = capfd.readouterr().out
    assert output == expected


def test_write_yaml_stdout_without_flush_writes_to_stdout(monkeypatch: pytest.MonkeyPatch):
    class NoFlush:
        def __init__(self):
            self.parts: list[str] = []

        def write(self, payload: str):
            self.parts.append(payload)

    sink = NoFlush()
    monkeypatch.setattr(sys, "stdout", sink)

    value = {"alpha": 1}
    expected = f"---\n{yaml12.format_yaml(value)}\n"

    yaml12.write_yaml(value, path=None)
    assert "".join(sink.parts) == expected


def test_write_yaml_stdout_flush_none_is_ignored(monkeypatch: pytest.MonkeyPatch):
    class FlushNone:
        def __init__(self):
            self.parts: list[str] = []
            self.flush = None

        def write(self, payload: str):
            self.parts.append(payload)

    sink = FlushNone()
    monkeypatch.setattr(sys, "stdout", sink)

    value = {"alpha": 1}
    expected = f"---\n{yaml12.format_yaml(value)}\n"

    yaml12.write_yaml(value, path=None)
    assert "".join(sink.parts) == expected


def test_write_yaml_stdout_flush_error_is_ignored(monkeypatch: pytest.MonkeyPatch):
    class FlushError:
        def __init__(self):
            self.parts: list[str] = []

        def write(self, payload: str):
            self.parts.append(payload)

        def flush(self):
            raise RuntimeError("boom flush")

    sink = FlushError()
    monkeypatch.setattr(sys, "stdout", sink)

    value = {"alpha": 1}
    expected = f"---\n{yaml12.format_yaml(value)}\n"

    yaml12.write_yaml(value, path=None)
    assert "".join(sink.parts) == expected


def test_write_and_read_multi_document_streams(tmp_path: Path):
    path = tmp_path / "yaml12-multi.yaml"
    docs = [{"foo": 1}, {"bar": [2, None]}]

    encoded = yaml12.format_yaml(docs, multi=True)
    yaml12.write_yaml(docs, str(path), multi=True)

    assert path.read_text(encoding="utf-8") == encoded
    assert yaml12.parse_yaml(encoded, multi=True) == docs
    assert yaml12.read_yaml(str(path), multi=True) == docs


def test_write_yaml_appends_documents(tmp_path: Path):
    path = tmp_path / "yaml12-append.yaml"
    first = {"first": 1}
    second = {"second": 2}
    docs = [{"third": 3}, {"fourth": 4}]

    yaml12.write_yaml(first, str(path), append=True)
    yaml12.write_yaml(second, path, False, True)
    yaml12.write_yaml(docs, path, multi=True, append=True)

    assert "..." not in path.read_text(encoding="utf-8").splitlines()
    assert yaml12.read_yaml(path, multi=True) == [first, second, *docs]

    replacement = [{"replacement": 5}]
    yaml12.write_yaml(replacement, path, multi=True)
    assert yaml12.read_yaml(path, multi=True) == replacement


def test_write_yaml_width_controls_wrapping(tmp_path: Path):
    path = tmp_path / "yaml12-width.yaml"
    value = {"body": "alpha beta gamma delta epsilon"}

    yaml12.write_yaml(value, path, width=20)

    assert "body: >-\n" in path.read_text(encoding="utf-8")
    assert yaml12.read_yaml(path) == value

    yaml12.write_yaml(value, path, width=None)

    assert "body: alpha beta gamma delta epsilon" in path.read_text(encoding="utf-8")
    assert yaml12.read_yaml(path) == value

    yaml12.write_yaml(value, path, width=20.0)

    assert "body: >-\n" in path.read_text(encoding="utf-8")
    assert yaml12.read_yaml(path) == value

    for width in [float("inf"), float("-inf"), float("nan")]:
        yaml12.write_yaml(value, path, width=width)

        assert "body: alpha beta gamma delta epsilon" in path.read_text(
            encoding="utf-8"
        )
        assert yaml12.read_yaml(path) == value


def test_write_yaml_multi_empty_sequence_emits_empty_document(tmp_path: Path):
    path = tmp_path / "yaml12-empty-multi.yaml"

    yaml12.write_yaml([], str(path), multi=True)

    assert path.read_text(encoding="utf-8") == "---\n"
    assert yaml12.read_yaml(str(path), multi=True) == [None]


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


def test_read_yaml_read_error_propagates():
    with pytest.raises(RuntimeError, match="boom stream"):
        yaml12.read_yaml(_ErroringReader())


def test_read_yaml_read_fallback_to_size_arg():
    class RequiresSize:
        def __init__(self, text: str):
            self.text = text

        def read(self, size: int) -> str:  # noqa: ARG002
            return self.text

    reader = RequiresSize("foo: 1\nbar: true\n")
    assert yaml12.read_yaml(reader) == {"foo": 1, "bar": True}


def test_read_yaml_connection_empty_respects_multi_flag():
    buf = io.StringIO("")
    assert yaml12.read_yaml(buf) is None
    buf_multi = io.StringIO("")
    assert yaml12.read_yaml(buf_multi, multi=True) == []


def test_read_yaml_streaming_bad_type_error():
    with pytest.raises(TypeError, match="str or bytes"):
        yaml12.read_yaml(_BadTypeReader())


def test_read_yaml_accepts_pathlike(tmp_path: Path):
    path = tmp_path / "pathlike-read.yaml"
    path.write_text("foo: 1\n", encoding="utf-8")

    class PathLike:
        def __fspath__(self):
            return str(path)

    assert yaml12.read_yaml(path) == {"foo": 1}
    assert yaml12.read_yaml(PathLike()) == {"foo": 1}


def test_read_and_write_yaml_expand_user_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("USERPROFILE", str(home))
    value = {"alpha": 1, "beta": "two"}

    paths = ["~/string.yaml", Path("~/pathlike.yaml")]
    for path in paths:
        yaml12.write_yaml(value, path)

        expanded = home / Path(path).name
        assert expanded.exists()
        assert yaml12.read_yaml(path) == value


def test_user_path_errors_report_the_expanded_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("USERPROFILE", str(home))

    with pytest.raises(OSError) as excinfo:
        yaml12.read_yaml("~/missing.yaml")

    assert os.path.expanduser("~/missing.yaml") in str(excinfo.value)


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

    assert isinstance(first, yaml12.Yaml)
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
            str(path),
            handlers={
                "!err": lambda _: (_ for _ in ()).throw(RuntimeError("handler oops"))
            },
        )


def test_write_yaml_accepts_text_writer(tmp_path: Path):
    path = tmp_path / "writer-text.yaml"
    handle = path.open("w", encoding="utf-8")
    value = {"foo": 1}
    yaml12.write_yaml(value, handle)
    handle.close()
    assert (
        path.read_text(encoding="utf-8") == f"---\n{yaml12.format_yaml(value)}\n"
    )


def test_write_yaml_rejects_bytes_writer(tmp_path: Path):
    path = tmp_path / "writer-bytes.yaml"
    handle = path.open("wb")
    try:
        with pytest.raises(TypeError, match="writer must accept str"):
            yaml12.write_yaml({"foo": 1}, handle)
    finally:
        handle.close()


def test_write_yaml_text_writer_handles_partial_writes():
    class PartialTextWriter:
        def __init__(self, chunk: int = 1):
            self.chunk = chunk
            self.parts: list[str] = []

        def write(self, payload: str) -> int:
            n = min(self.chunk, len(payload))
            self.parts.append(payload[:n])
            return n

    value = {"snowman": "☃"}
    sink = PartialTextWriter(chunk=1)
    yaml12.write_yaml(value, sink)

    expected = f"---\n{yaml12.format_yaml(value)}\n"
    assert "".join(sink.parts) == expected


def test_write_yaml_raises_when_writer_returns_zero():
    class ZeroWriter:
        def write(self, payload: str) -> int:  # noqa: ARG002
            return 0

    with pytest.raises(OSError, match="returned 0"):
        yaml12.write_yaml({"alpha": 1}, ZeroWriter())


def test_write_yaml_rejects_buffered_writer():
    sink = io.BytesIO()
    with pytest.raises(TypeError, match="writer must accept str"):
        yaml12.write_yaml({"alpha": 1}, sink)


def test_write_yaml_prefers_text_for_textiobase():
    class Sink(io.TextIOBase):
        def __init__(self):
            self.parts: list[str] = []

        def writable(self) -> bool:
            return True

        def write(self, s: str) -> int:  # type: ignore[override]
            assert isinstance(s, str)
            # Simulate a partial write contract.
            n = max(1, len(s) // 2)
            self.parts.append(s[:n])
            return n

    sink = Sink()
    yaml12.write_yaml({"alpha": 1}, sink)
    expected = f"---\n{yaml12.format_yaml({'alpha': 1})}\n"
    assert "".join(sink.parts) == expected


def test_write_yaml_accepts_pathlike(tmp_path: Path):
    value = {"foo": 1}
    path_from_path = tmp_path / "writer-path.yaml"
    path_from_fspath = tmp_path / "writer-fspath.yaml"

    class PathLike:
        def __init__(self, path: Path):
            self.path = path

        def __fspath__(self):
            return str(self.path)

    yaml12.write_yaml(value, path_from_path)
    yaml12.write_yaml(value, PathLike(path_from_fspath))

    expected = f"---\n{yaml12.format_yaml(value)}\n"
    assert path_from_path.read_text(encoding="utf-8") == expected
    assert path_from_fspath.read_text(encoding="utf-8") == expected


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
