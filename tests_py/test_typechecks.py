from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import tempfile
from pathlib import Path

def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _have_module(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def test_typechecks_pass() -> None:
    root = _repo_root()
    if not _have_module("mypy") or not _have_module("pyright"):
        raise AssertionError(
            "Missing `mypy`/`pyright` in test environment; install them to run the test suite"
        )

    with tempfile.TemporaryDirectory(dir=root) as tmpdir:
        tmpdir_path = Path(tmpdir)
        usage = tmpdir_path / "usage.py"
        usage.write_text(
            """\
from __future__ import annotations

import io
from pathlib import Path
from typing import Callable

import yaml12

handlers: dict[str, Callable[[object], object]] = {"!upper": lambda v: str(v).upper()}
yaml12.parse_yaml("!upper foo", handlers=handlers)
yaml12.parse_yaml("---\\n1\\n---\\n2\\n", multi=True)

p = Path(__file__).with_suffix(".yaml")
p.write_text("foo: 1\\n", encoding="utf-8")
yaml12.read_yaml(p)
yaml12.read_yaml(p, multi=True)

yaml12.read_yaml(io.StringIO("foo: 1\\n"))
yaml12.read_yaml(io.BytesIO(b"foo: 1\\n"))

class CustomTextReader:
    def __init__(self, text: str):
        self._text = text

    def read(self, size: int = -1) -> str:  # noqa: ARG002
        return self._text

class CustomBytesReader:
    def __init__(self, payload: bytes):
        self._payload = payload

    def read(self, size: int = -1) -> bytes:  # noqa: ARG002
        return self._payload

yaml12.read_yaml(CustomTextReader("foo: 1\\n"))
yaml12.read_yaml(CustomBytesReader(b"foo: 1\\n"))

class CustomTextWriter:
    def __init__(self):
        self.parts: list[str] = []

    def write(self, data: str) -> int:
        self.parts.append(data)
        return len(data)

yaml12.write_yaml({"a": 1}, path=CustomTextWriter())
yaml12.format_yaml({"a": 1}, width=None)
yaml12.write_yaml({"a": 1}, path=p, width=None)
""",
            encoding="utf-8",
        )

        bad = tmpdir_path / "bad_usage.py"
        bad.write_text(
            """\
from __future__ import annotations

import yaml12

yaml12.parse_yaml(123)
yaml12.read_yaml(123)
yaml12.write_yaml({"a": 1}, path=123)
yaml12.format_yaml({"a": 1}, width=20.0)
yaml12.write_yaml({"a": 1}, width=float("inf"))
""",
            encoding="utf-8",
        )

        # mypy stubtest: stubs match runtime interface for top-level module
        subprocess.run(
            [sys.executable, "-m", "mypy.stubtest", "yaml12", "--ignore-missing-stub"],
            check=True,
            cwd=root,
        )

        # mypy positive/negative
        subprocess.run([sys.executable, "-m", "mypy", str(usage)], check=True, cwd=root)
        bad_proc = subprocess.run(
            [sys.executable, "-m", "mypy", str(bad)],
            check=False,
            cwd=root,
            text=True,
            capture_output=True,
        )
        assert bad_proc.returncode != 0
        bad_output = (bad_proc.stdout or "") + (bad_proc.stderr or "")
        assert bad_output.count("error:") >= 5

        # verify typing artifacts are installed alongside the extension
        import yaml12  # noqa: WPS433

        pkg_dir = Path(yaml12.__file__).resolve().parent
        assert (pkg_dir / "py.typed").exists()
        assert (pkg_dir / "__init__.pyi").exists()

        # pyright positive and a python-version override
        subprocess.run(
            [
                sys.executable,
                "-m",
                "pyright",
                "--pythonpath",
                sys.executable,
                str(usage),
            ],
            check=True,
            cwd=root,
        )
        subprocess.run(
            [
                sys.executable,
                "-m",
                "pyright",
                "--pythonpath",
                sys.executable,
                "--pythonversion",
                "3.14",
                str(usage),
            ],
            check=True,
            cwd=root,
        )

        # pyright negative: expect at least one error
        neg = subprocess.run(
            [
                sys.executable,
                "-m",
                "pyright",
                "--pythonpath",
                sys.executable,
                "--outputjson",
                str(bad),
            ],
            check=False,
            cwd=root,
            text=True,
            capture_output=True,
        )
        data = json.loads(neg.stdout or "{}")
        assert data.get("summary", {}).get("errorCount", 0) >= 5
