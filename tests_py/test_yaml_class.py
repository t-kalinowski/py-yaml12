from __future__ import annotations

from pathlib import Path

import pytest

import yaml12


def test_yaml12_is_extension_module() -> None:
    native = yaml12.yaml12
    module_path = Path(native.__file__).resolve()
    assert module_path.suffix in {".so", ".pyd", ".dll"}
    assert yaml12.parse_yaml is native.parse_yaml
    assert yaml12.Yaml is native.Yaml


def test_yaml_is_not_a_dataclass() -> None:
    assert not hasattr(yaml12.Yaml, "__dataclass_fields__")
    assert yaml12.Yaml.__module__ == "yaml12"

def test_yaml_has_docstring() -> None:
    doc = yaml12.Yaml.__doc__
    assert isinstance(doc, str)
    assert "Tagged node" in doc


def test_yaml_is_immutable_and_proxies_collections() -> None:
    wrapped = yaml12.Yaml([1, 2, 3])
    assert wrapped[0] == 1
    assert list(wrapped) == [1, 2, 3]
    assert len(wrapped) == 3

    nested = yaml12.Yaml(wrapped)
    assert nested[1] == 2

    with pytest.raises(AttributeError):
        wrapped.value = [9]  # type: ignore[misc]
    with pytest.raises(AttributeError):
        wrapped.tag = "!x"  # type: ignore[misc]


def test_yaml_constructor_normalizes_simple_local_tags() -> None:
    tagged = yaml12.Yaml("value", "gizmo")
    assert tagged.tag == "!gizmo"


def test_yaml_constructor_rejects_non_string_tag() -> None:
    with pytest.raises(TypeError):
        yaml12.Yaml(3, 4)  # type: ignore[arg-type]


def test_yaml_is_picklable() -> None:
    import pickle

    obj = yaml12.Yaml({"a": [1, 2]}, "!tag")
    data = pickle.dumps(obj)
    out = pickle.loads(data)

    assert isinstance(out, yaml12.Yaml)
    assert out == obj
