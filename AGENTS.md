# Agent Instructions

- Do not hand-edit generated artifacts. Keep edits focused on `src/` Rust code, `Cargo.toml`, and Python-facing module setup.
- Avoid destructive git operations (no reset --hard, no checkout --) unless explicitly requested.
- Prefer `rg` for searching and run `cargo fmt` and `cargo clippy` before finishing substantial edits.
- How to run tests (always run the relevant suites before finishing work):
  - Rust unit/integration tests: `cargo test`.
  - Python tests: in a venv with the package installed (`.venv/bin/maturin develop` or `.venv/bin/pip install -e .`), ensure `pytest` is installed (`.venv/bin/pip install pytest`), then run `.venv/bin/python -m pytest tests_py`.
- Always (re)install the local package into the venv before running Python tests with: `.venv/bin/pip install -e . --no-build-isolation`.
- Always run the tests yourself and report results; do not tell the user to run them.
- Use the existing `saphyr`/`saphyr-parser` git dependencies; do not change crate sources or pins without approval.
- For Python bindings, preserve the exposed API (`parse_yaml`, `read_yaml`, `format_yaml`, `write_yaml`) and the `Tagged` dataclass contract. Tagged values wrap non-core tags only; canonical/core tags are treated as untagged.
- Keep conversions zero-copy where practical and prefer borrowing (`&str`, slices) over allocating new `String`s when possible.
- Performance is the top priority; if a measurable speedup requires hairy or off-contract techniques, prefer the faster approach and document the choice.
- When running checks, use `cargo check` or `cargo test` from the project root; avoid adding new test frameworks without approval. Use `cargo fmt` before final delivery if Rust code changed.
- Before wrapping up, always run: `cargo fmt`, `cargo check`, `cargo test`, `cargo build`, `cargo clippy`, `.venv/bin/pip install -e . --no-build-isolation`, and `.venv/bin/python -m pytest tests_py`.
