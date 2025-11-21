use pyo3::types::{PyDict, PyList, PyModule, PyString};
use pyo3::{prelude::*, PyResult};

fn init_module(py: Python<'_>) -> PyResult<Bound<'_, PyModule>> {
    let module = PyModule::new(py, "yaml12_test")?;
    crate::yaml12(py, &module)?;
    Ok(module)
}

#[test]
fn parse_simple_mapping() -> PyResult<()> {
    pyo3::prepare_freethreaded_python();
    Python::with_gil(|py| {
        let module = init_module(py)?;
        let parse = module.getattr("parse_yaml")?;
        let obj = parse.call1(("foo: 1\nbar: true",))?;
        let mapping: Bound<'_, PyDict> = obj.downcast(py)?;
        assert_eq!(mapping.get_item("foo").unwrap().extract::<i64>()?, 1);
        assert_eq!(mapping.get_item("bar").unwrap().extract::<bool>()?, true);
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

        let docs = PyList::new(py, &["first", "second"])?;
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
        let list: Bound<'_, PyList> = parsed.downcast(py)?;
        assert_eq!(list.len(), 2);
        assert_eq!(list.get_item(0)?.downcast::<PyString>()?.to_str()?, "first");
        assert_eq!(list.get_item(1)?.downcast::<PyString>()?.to_str()?, "second");
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
        let tagged_cls = module.getattr("Tagged")?;

        // Parsing produces Tagged for non-core tags.
        let parsed = parse_yaml.call1(("!foo 1",))?;
        assert!(parsed.is_instance(&tagged_cls)?);
        assert_eq!(parsed.getattr("tag")?.extract::<String>()?, "!foo");
        assert_eq!(parsed.getattr("value")?.extract::<i64>()?, 1);

        // Formatting a Tagged should emit the tag and round-trip.
        let tagged = tagged_cls.call1((2, "!bar"))?;
        let yaml = format_yaml.call1((tagged,))?.extract::<String>()?;
        assert!(yaml.starts_with("!bar "));
        let reparsed = parse_yaml.call1((yaml.as_str(),))?;
        assert!(reparsed.is_instance(&tagged_cls)?);
        assert_eq!(reparsed.getattr("tag")?.extract::<String>()?, "!bar");
        assert_eq!(reparsed.getattr("value")?.extract::<i64>()?, 2);
        Ok(())
    })
}
