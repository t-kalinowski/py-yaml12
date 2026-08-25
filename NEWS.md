# yaml12 0.2.0 (2026-08-25)

## Formatting

- `format_yaml()` and `write_yaml()` now produce more human-readable YAML while
  preserving exact round trips:

  - The new integer `width` argument defaults to 80 columns and wraps long
    strings at safe word boundaries. Pass `None` to disable wrapping.

  - Strings are emitted without quotes when YAML 1.2 permits. Strings that the
    core schema would read as null, a boolean, or a number remain quoted,
    including arbitrary-sized decimal, octal, and hexadecimal integer strings.

  - Multiline strings use readable folded or literal block styles when
    possible. Paragraph breaks, leading whitespace, empty lines, trailing
    newlines, and embedded document markers round-trip unchanged. Other
    multiline strings use a quoted fallback.

- Mapping keys longer than YAML's 1,024-character simple-key limit now use
  explicit key syntax.

- Non-finite floats now use the canonical spellings `.Inf`, `-.Inf`, and
  `.NaN` while continuing to round-trip as floats.

- `write_yaml()` and `format_yaml(..., multi=True)` no longer add optional
  document end (`...`) markers. Written documents still begin with `---` and
  end with a newline.

## Parsing and file handling

- By default, `parse_yaml()` and `read_yaml()` now preserve tags in the
  `tag:yaml.org,2002:` namespace that are not converted to built-in Python
  values in a `Yaml` wrapper. Explicit handlers still run when supplied.

- The parser now accepts and ignores reserved directives such as `%***`, and
  treats indented `---` text as part of a multiline plain scalar.

- `write_yaml()` gains an `append` argument for adding complete YAML documents
  to a filesystem path. Its default behavior still replaces the file.
  `read_yaml()` and `write_yaml()` also expand paths beginning with `~` using
  `os.path.expanduser()`.

## Installation

- Prebuilt wheels now include Intel macOS, Windows ARM64, and CPython 3.14
  free-threaded builds. Free-threading support is currently beta. Source builds
  now require Rust 1.83 or newer.
