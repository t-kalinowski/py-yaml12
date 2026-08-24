# Release process

The `CI` workflow builds release artifacts on every `v*` tag and publishes
them to PyPI with trusted publishing. A manual workflow run builds the same
artifacts without publishing them.

## Prepare the release

1. Start from the current `main` branch and confirm its CI and documentation
   workflows pass.
2. Choose the release version and update it in `Cargo.toml`. Run `cargo check`
   to update the root package entry in `Cargo.lock`.
3. Replace the development heading in `NEWS.md` with the version and release
   date. Update the version and year in
   `user_guide/90-authors-and-citation.qmd`.
4. Review direct dependencies and run `cargo update --dry-run`. Do not move
   the `saphyr` git revision as part of a release unless that change was
   reviewed and tested separately.
5. Run the complete local checks:

   ```bash
   cargo fmt
   cargo check
   cargo test
   cargo build
   cargo clippy
   .venv/bin/python -m pip install --group test
   .venv/bin/pip install -e . --no-build-isolation
   .venv/bin/python -m pytest tests_py
   ```

6. Build the local release artifacts and inspect their metadata and contents:

   ```bash
   .venv/bin/maturin build --locked --release --out dist
   .venv/bin/maturin sdist --out dist
   ```

7. Review the full diff, commit the release preparation, push it, and wait for
   the branch workflows to pass. A manual `CI` workflow run can exercise the
   complete wheel matrix before tagging.

## Publish

1. Create an annotated `vX.Y.Z` tag on the reviewed release commit. The tag
   version must match `Cargo.toml`.
2. Push the tag. Watch the tag-triggered `CI` workflow through the `Release`
   job. It attests the wheel and source artifacts before publishing them to
   PyPI.
3. Confirm the published file list includes the source distribution, stable
   ABI CPython wheels, CPython 3.14 free-threaded wheels, and the intended
   PyPy wheels for every documented platform.
4. Install the release in a fresh environment from PyPI and run a parse and
   format smoke test.
5. Create the GitHub release from the existing tag using release notes derived
   from `NEWS.md`.

PyPI files cannot be replaced. If publication is incomplete, diagnose the
failed target before deciding whether to retry the workflow or publish a new
patch version.
