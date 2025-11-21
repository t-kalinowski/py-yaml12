# Agent Instructions

- Do not hand-edit generated artifacts. Keep edits focused on `src/` Rust code, `Cargo.toml`, and Python-facing module setup.
- Avoid destructive git operations (no reset --hard, no checkout --) unless explicitly requested.
- Prefer `rg` for searching and run `cargo fmt` and `cargo clippy` before finishing substantial edits.
- Use the existing `saphyr`/`saphyr-parser` git dependencies; do not change crate sources or pins without approval.
- For Python bindings, preserve the exposed API (`parse_yaml`, `read_yaml`, `format_yaml`, `write_yaml`) and the `Tagged` dataclass contract. Tagged values wrap non-core tags only; canonical/core tags are treated as untagged.
- Keep conversions zero-copy where practical and prefer borrowing (`&str`, slices) over allocating new `String`s when possible.
- When running checks, use `cargo check` or `cargo test` from the project root; avoid adding new test frameworks without approval. Use `cargo fmt` before final delivery if Rust code changed.
