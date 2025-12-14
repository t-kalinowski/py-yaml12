# Agent Instructions

- Do not hand-edit generated artifacts. Keep edits focused on `src/` Rust code, `Cargo.toml`, and Python-facing module setup.
- Avoid destructive git operations (no reset --hard, no checkout --) unless explicitly requested.
- Prefer `rg` for searching and run `cargo fmt` and `cargo clippy` before finishing substantial edits.
- How to run tests (always run the relevant suites before finishing work):
  - Rust unit/integration tests: `cargo test`.
  - Python tests: use the project `.venv` (safe to recreate). If missing or stale, run `python3 -m venv .venv && source .venv/bin/activate && pip install -U pip maturin pytest`. Install the extension with `.venv/bin/maturin develop --locked`, then run `.venv/bin/python -m pytest tests_py`.
- Avoid ad-hoc local installs like `local-py`; stick to `maturin develop --locked` in `.venv` or an existing project venv.
- If `maturin develop` is blocked by sandboxed networking or permissions for git dependencies, request elevated permissions.
- Always run the tests yourself and report results; do not tell the user to run them.
- When opening PRs, do not include a "Testing" section in the PR description; assume CI will run the project's test checks.
- Use the existing `saphyr`/`saphyr-parser` git dependencies; do not change crate sources or pins without approval.
- For Python bindings, preserve the exposed API (`parse_yaml`, `read_yaml`, `format_yaml`, `write_yaml`) and the `Tagged` dataclass contract. Tagged values wrap non-core tags only; canonical/core tags are treated as untagged.
- Keep conversions zero-copy where practical and prefer borrowing (`&str`, slices) over allocating new `String`s when possible.
- Performance is the top priority; if a measurable speedup requires hairy or off-contract techniques, prefer the faster approach and document the choice.
- When running checks, use `cargo check` or `cargo test` from the project root; avoid adding new test frameworks without approval. Use `cargo fmt` before final delivery if Rust code changed.
- Before wrapping up, always run: `cargo fmt`, `cargo check`, `cargo test`, `cargo build`, `cargo clippy`, `.venv/bin/pip install -e . --no-build-isolation`, and `.venv/bin/python -m pytest tests_py`.
- After every set of changes, emit a draft commit message. If you are asked for revisions, when you're done, emit
  an updated draft commit message.
