from __future__ import annotations

import os
import subprocess
import sys
import sysconfig

import pytest


@pytest.mark.skipif(
    sysconfig.get_config_var("Py_GIL_DISABLED") != 1,
    reason="requires a free-threaded Python build",
)
def test_import_and_concurrent_use_keep_the_gil_disabled() -> None:
    script = """
import sys
from concurrent.futures import ThreadPoolExecutor

assert not sys._is_gil_enabled()
from yaml12 import format_yaml, parse_yaml
assert not sys._is_gil_enabled()

value = {
    "documents": [
        {"name": f"document-{index}", "values": list(range(20))}
        for index in range(20)
    ]
}
text = format_yaml(value)

def round_trip(_):
    return parse_yaml(format_yaml(parse_yaml(text)))

with ThreadPoolExecutor(max_workers=8) as pool:
    results = list(pool.map(round_trip, range(64)))

assert results == [value] * 64
"""
    env = os.environ.copy()
    env.pop("PYTHON_GIL", None)
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        env=env,
        text=True,
    )
    assert result.returncode == 0, result.stderr
