# yaml12 (development version)

- `write_yaml()` and multi-document `format_yaml()` no longer add optional document end (`...`) markers. Written documents still begin with `---` and end with a newline.

- `format_yaml()` and `write_yaml()` now wrap long strings at word boundaries. The new integer `width` argument defaults to 80 columns; pass `None` to disable wrapping.

- YAML formatting now emits YAML 1.2-safe strings without unnecessary quotes, while still quoting strings that the core schema would resolve as another type. This includes arbitrary-sized decimal, octal, and hexadecimal integer strings.

- Multiline strings now use lossless folded or literal block styles. Formatting preserves paragraph breaks, leading whitespace, empty lines, trailing newlines, and root-level document markers. Mapping keys longer than YAML's 1,024-character simple-key limit use explicit key syntax.

- Non-finite floats now use the canonical spellings `.Inf`, `-.Inf`, and `.NaN` while continuing to round-trip as floats.

- `write_yaml()` gains an `append` argument for adding complete YAML documents to an existing file. Its default behavior still replaces the file.

- `read_yaml()` and `write_yaml()` now expand filesystem paths beginning with `~` using `os.path.expanduser()`.
