use pyo3::types::{PyDict, PyList, PyModule};
use pyo3::{prelude::*, PyResult};
use saphyr::Yaml;
use saphyr_parser::Parser;

fn init_module(py: Python<'_>) -> PyResult<Bound<'_, PyModule>> {
    let module = PyModule::new(py, "yaml12_test")?;
    yaml12::yaml12(py, &module)?;
    Ok(module)
}

#[test]
fn parse_simple_mapping() -> PyResult<()> {
    pyo3::prepare_freethreaded_python();
    Python::with_gil(|py| {
        let module = init_module(py)?;
        let parse = module.getattr("parse_yaml")?;
        let obj = parse.call1(("foo: 1\nbar: true",))?;
        let mapping: Bound<'_, PyDict> = obj.downcast_into()?;
        let foo = mapping
            .get_item("foo")?
            .expect("foo key should be present")
            .extract::<i64>()?;
        let bar = mapping
            .get_item("bar")?
            .expect("bar key should be present")
            .extract::<bool>()?;
        assert_eq!(foo, 1);
        assert!(bar);
        Ok(())
    })
}

#[test]
fn roundtrip_multi_documents() -> PyResult<()> {
    pyo3::prepare_freethreaded_python();
    Python::with_gil(|py| {
        let module = init_module(py)?;
        let format_yaml = module.getattr("format_yaml")?;
        let parse_yaml = module.getattr("parse_yaml")?;

        let docs = PyList::new(py, ["first", "second"])?;
        let yaml = format_yaml.call1((docs, true))?.extract::<String>()?;
        assert!(
            yaml.starts_with("---"),
            "expected multi-doc stream to start with document marker"
        );
        assert!(
            yaml.trim_end().ends_with("..."),
            "expected multi-doc stream to end with terminator"
        );

        let parsed = parse_yaml.call1((yaml.as_str(), true))?;
        let list: Bound<'_, PyList> = parsed.downcast_into()?;
        assert_eq!(list.len(), 2);
        assert_eq!(list.get_item(0)?.extract::<String>()?, "first");
        assert_eq!(list.get_item(1)?.extract::<String>()?, "second");
        Ok(())
    })
}

#[test]
fn preserves_non_core_tags() -> PyResult<()> {
    pyo3::prepare_freethreaded_python();
    Python::with_gil(|py| {
        let module = init_module(py)?;
        let parse_yaml = module.getattr("parse_yaml")?;
        let format_yaml = module.getattr("format_yaml")?;
        let yaml_cls = module.getattr("Yaml")?;

        // Parsing produces Yaml nodes for non-core tags.
        let parsed = parse_yaml.call1(("!foo 1",))?;
        assert!(parsed.is_instance(&yaml_cls)?);
        assert_eq!(parsed.getattr("tag")?.extract::<String>()?, "!foo");
        assert_eq!(parsed.getattr("value")?.extract::<String>()?, "1");

        // Formatting a tagged node should emit the tag and round-trip.
        let tagged = yaml_cls.call1((2, "!bar"))?;
        let yaml = format_yaml.call1((tagged,))?.extract::<String>()?;
        assert!(yaml.starts_with("!bar "));
        let reparsed = parse_yaml.call1((yaml.as_str(),))?;
        assert!(reparsed.is_instance(&yaml_cls)?);
        assert_eq!(reparsed.getattr("tag")?.extract::<String>()?, "!bar");
        assert_eq!(reparsed.getattr("value")?.extract::<String>()?, "2");
        Ok(())
    })
}

#[test]
fn non_specific_tag_parts_are_consistent() {
    let mut loader = saphyr::YamlLoader::default();
    loader.early_parse(false);
    let mut parser = Parser::new_from_str("! 001");
    parser
        .load(&mut loader, false)
        .expect("parser should load non-specific tag");
    let mut docs = loader.into_documents();
    let doc = docs.pop().expect("document should be present");
    match doc {
        Yaml::Tagged(tag, _) => {
            assert_eq!(tag.handle.as_str(), "");
            assert_eq!(tag.suffix.as_str(), "!");
        }
        Yaml::Representation(_, _, Some(tag)) => {
            assert_eq!(tag.handle.as_str(), "");
            assert_eq!(tag.suffix.as_str(), "!");
        }
        other => panic!("expected tagged or representation node, got {other:?}"),
    }
}
