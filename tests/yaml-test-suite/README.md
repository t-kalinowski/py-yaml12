# yaml-test-suite data

This directory vendors generated test data from the upstream
[YAML test suite](https://github.com/yaml/yaml-test-suite) under its MIT
license. The snapshot was imported from upstream commit
`ccfa74e56afb53da960847ff6e6976c0a0825709`.

Only the generated `data/` directory and upstream `License` are included.
Each case contains the YAML input and the applicable expected JSON, emitted
YAML, parse events, or error marker. `tests_py/test_suite.py` exercises the
parser through the public `parse_yaml()` API.

To update the snapshot, check out the intended upstream commit, run its
documented data-generation process, replace `data/` and `License`, and update
the commit above. Do not edit generated case files individually.
