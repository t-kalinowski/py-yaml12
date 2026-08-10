from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from os import PathLike
from typing import Any, Literal, Protocol, TypeAlias, final, overload

__all__: list[str]

_YamlScalar: TypeAlias = None | bool | int | float | str
_YamlKey: TypeAlias = _YamlScalar | "Yaml"
_YamlOutput: TypeAlias = _YamlScalar | "Yaml" | list["_YamlOutput"] | dict[_YamlKey, "_YamlOutput"]

class _Readable(Protocol):
    def read(self, size: int = ..., /) -> str | bytes: ...

class _Writable(Protocol):
    def write(self, data: str, /) -> int | None: ...


@final
class Yaml:
    """Tagged node or hashable wrapper for unhashable mapping keys."""

    def __new__(cls, value: Any, tag: str | None = ...) -> Yaml: ...
    def __init__(self) -> None: ...

    @property
    def value(self) -> Any: ...

    @property
    def tag(self) -> str | None: ...

    def __getitem__(self, key: Any, /) -> Any: ...
    def __iter__(self) -> Any: ...
    def __len__(self) -> int: ...
    def __hash__(self) -> int: ...
    def __eq__(self, other: object, /) -> bool: ...


@overload
def parse_yaml(
    text: str | Iterable[str],
    multi: Literal[False] = False,
    handlers: None = None,
) -> _YamlOutput: ...


@overload
def parse_yaml(
    text: str | Iterable[str],
    multi: Literal[True],
    handlers: None = None,
) -> list[_YamlOutput]: ...


@overload
def parse_yaml(
    text: str | Iterable[str],
    multi: bool = False,
    handlers: Mapping[str, Callable[[Any], Any]] | None = None,
) -> Any: ...

@overload
def read_yaml(
    path: str | PathLike[str] | _Readable,
    multi: Literal[False] = False,
    handlers: None = None,
) -> _YamlOutput: ...


@overload
def read_yaml(
    path: str | PathLike[str] | _Readable,
    multi: Literal[True],
    handlers: None = None,
) -> list[_YamlOutput]: ...


@overload
def read_yaml(
    path: str | PathLike[str] | _Readable,
    multi: bool = False,
    handlers: Mapping[str, Callable[[Any], Any]] | None = None,
) -> Any: ...


def _dbg_yaml(text: str | Iterable[str] | _Readable) -> None: ...


def write_yaml(
    value: Any,
    path: str | PathLike[str] | _Writable | None = None,
    multi: bool = False,
    append: bool = False,
    width: int | None = 80,
) -> None: ...

def format_yaml(value: Any, multi: bool = False, width: int | None = 80) -> str: ...
__version__: str
