from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from os import PathLike
from typing import Any, Literal, Protocol, TypeVar, overload

T_co = TypeVar("T_co", covariant=True)
T_contra = TypeVar("T_contra", contravariant=True)


class SupportsRead(Protocol[T_co]):
    @overload
    def read(self) -> T_co: ...
    @overload
    def read(self, size: int, /) -> T_co: ...


class SupportsWrite(Protocol[T_contra]):
    def write(self, data: T_contra, /) -> Any: ...


Handlers = Mapping[str, Callable[[Any], Any]]


@dataclass(frozen=True)
class Yaml:
    """Tagged node or hashable wrapper for unhashable mapping keys."""

    value: Any
    tag: str | None = ...

    def __getitem__(self, key: Any, /) -> Any: ...
    def __iter__(self) -> Any: ...
    def __len__(self) -> int: ...


@overload
def parse_yaml(
    text: str | Iterable[str],
    multi: Literal[False] = False,
    handlers: Handlers | None = None,
) -> Any: ...


@overload
def parse_yaml(
    text: str | Iterable[str],
    multi: Literal[True],
    handlers: Handlers | None = None,
) -> list[Any]: ...


@overload
def read_yaml(
    path: str | PathLike[str] | SupportsRead[str] | SupportsRead[bytes],
    multi: Literal[False] = False,
    handlers: Handlers | None = None,
) -> Any: ...


@overload
def read_yaml(
    path: str | PathLike[str] | SupportsRead[str] | SupportsRead[bytes],
    multi: Literal[True],
    handlers: Handlers | None = None,
) -> list[Any]: ...


def format_yaml(value: Any, multi: bool = False) -> str: ...


def write_yaml(
    value: Any,
    path: str | PathLike[str] | SupportsWrite[str] | SupportsWrite[bytes] | None = None,
    multi: bool = False,
) -> None: ...


def _dbg_yaml(text: str | Iterable[str] | SupportsRead[str] | SupportsRead[bytes]) -> None: ...


__version__: str

