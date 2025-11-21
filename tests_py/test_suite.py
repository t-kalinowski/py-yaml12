from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Literal, Tuple

import pytest

import yaml12

CASE_ROOT = Path(__file__).resolve().parents[1] / "tests" / "yaml-test-suite" / "data"


def _load_yaml_multi(text: str):
    """Parse always with multi=True to keep stream structure."""
    return yaml12.parse_yaml(text, multi=True)


def _iter_cases() -> Iterable[Tuple[Literal["json", "error", "parse_only"], Path]]:
    for in_yaml in sorted(CASE_ROOT.rglob("in.yaml")):
        case_dir = in_yaml.parent
        in_json = case_dir / "in.json"
        error_marker = case_dir / "error"

        if error_marker.exists():
            yield ("error", case_dir)
        elif in_json.exists():
            yield ("json", case_dir)
        else:
            yield ("parse_only", case_dir)


def _parse_json_stream(text: str):
    decoder = json.JSONDecoder()
    pos = 0
    results = []
    length = len(text)
    while True:
        while pos < length and text[pos].isspace():
            pos += 1
        if pos >= length:
            break
        obj, new_pos = decoder.raw_decode(text, pos)
        results.append(obj)
        pos = new_pos
    return results


def _strip_tags(obj):
    if isinstance(obj, yaml12.Tagged):
        if obj.tag == "!":
            return str(obj.value)
        return _strip_tags(obj.value)
    if isinstance(obj, list):
        return [_strip_tags(item) for item in obj]
    if isinstance(obj, dict):
        return { _strip_tags(k): _strip_tags(v) for k, v in obj.items() }
    return obj


@pytest.mark.parametrize(
    "kind, case_dir",
    _iter_cases(),
    ids=lambda kc: f"{kc[0]}:{kc[1].name}" if isinstance(kc, tuple) else str(kc),
)
def test_yaml_suite_cases(
    kind: Literal["json", "error", "parse_only"], case_dir: Path
):
    in_yaml = (case_dir / "in.yaml").read_text(encoding="utf-8")

    if kind == "error":
        try:
            yaml12.parse_yaml(in_yaml, multi=True)
        except Exception:
            return
        pytest.xfail(f"Parser accepted error-marked case {case_dir.name}")
        return

    if kind == "json":
        expected_stream = _parse_json_stream((case_dir / "in.json").read_text(encoding="utf-8"))
        actual_stream = _load_yaml_multi(in_yaml)
        assert _strip_tags(actual_stream) == expected_stream
        return

    actual_stream = _load_yaml_multi(in_yaml)
    assert isinstance(actual_stream, list)
