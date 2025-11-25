# API Reference (now under "Reference" in the nav)

The detailed API pages now live in the Reference section:

- [`parse_yaml()` and `read_yaml()`](reference/parse_yaml.md)
- [`format_yaml()` and `write_yaml()`](reference/format_yaml.md)

`Yaml` is the single wrapper type for tagged nodes and unhashable
mapping keys. Use plain Python types whenever possible; reach for
`Yaml` when you need to preserve a tag or keep a list/dict hashable as a
mapping key. For a quick overview of the surface area, see
[Reference](reference/index.md).
