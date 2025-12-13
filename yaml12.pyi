from __future__ import annotations

import io
import sys
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from os import PathLike
from typing import Any, BinaryIO, Literal, TextIO, overload

__all__: list[str]


@dataclass(frozen=True)
class Yaml:
    """Tagged node or hashable wrapper for unhashable mapping keys."""

    value: Any
    tag: str | None = ...

    def __post_init__(self) -> None: ...

    def __getitem__(self, key: Any, /) -> Any: ...
    def __iter__(self) -> Any: ...
    def __len__(self) -> int: ...


@overload
def parse_yaml(
    text: str | Iterable[str],
    multi: Literal[False] = False,
    handlers: Mapping[str, Callable[[Any], Any]] | None = None,
) -> Any: ...


@overload
def parse_yaml(
    text: str | Iterable[str],
    multi: Literal[True],
    handlers: Mapping[str, Callable[[Any], Any]] | None = None,
) -> list[Any]: ...

if sys.version_info >= (3, 14):
    @overload
    def read_yaml(
        path: str | PathLike[str] | io.Reader[str] | io.Reader[bytes],
        multi: Literal[False] = False,
        handlers: Mapping[str, Callable[[Any], Any]] | None = None,
    ) -> Any: ...

    @overload
    def read_yaml(
        path: str | PathLike[str] | io.Reader[str] | io.Reader[bytes],
        multi: Literal[True],
        handlers: Mapping[str, Callable[[Any], Any]] | None = None,
    ) -> list[Any]: ...
else:
    @overload
    def read_yaml(
        path: str | PathLike[str] | TextIO | BinaryIO,
        multi: Literal[False] = False,
        handlers: Mapping[str, Callable[[Any], Any]] | None = None,
    ) -> Any: ...

    @overload
    def read_yaml(
        path: str | PathLike[str] | TextIO | BinaryIO,
        multi: Literal[True],
        handlers: Mapping[str, Callable[[Any], Any]] | None = None,
    ) -> list[Any]: ...


def format_yaml(value: Any, multi: bool = False) -> str: ...

if sys.version_info >= (3, 14):
    def write_yaml(
        value: Any,
        path: str | PathLike[str] | io.Writer[str] | io.Writer[bytes] | None = None,
        multi: bool = False,
    ) -> None: ...
else:
    def write_yaml(
        value: Any,
        path: str | PathLike[str] | TextIO | BinaryIO | None = None,
        multi: bool = False,
    ) -> None: ...

if sys.version_info >= (3, 14):
    def _dbg_yaml(
        text: str | Iterable[str] | io.Reader[str] | io.Reader[bytes],
    ) -> None: ...
else:
    def _dbg_yaml(text: str | Iterable[str] | TextIO | BinaryIO) -> None: ...


__version__: str
