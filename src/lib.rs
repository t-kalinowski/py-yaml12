use pyo3::exceptions::{PyIOError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::sync::GILOnceCell;
use pyo3::types::{
    PyAnyMethods, PyDict, PyDictMethods, PyList, PyModule, PySequence, PySequenceMethods, PyString,
    PyTuple,
};
use pyo3::Bound;
use pyo3::IntoPyObjectExt;
use saphyr::{Mapping, Scalar, Tag, Yaml, YamlEmitter};
use saphyr_parser::{Parser, ScalarStyle};
use std::borrow::Cow;
use std::collections::HashMap;
use std::ffi::CString;
use std::fs;
use std::io::{self, Write};
use std::mem;

type Result<T> = PyResult<T>;

static TAGGED_CLASS: GILOnceCell<Py<PyAny>> = GILOnceCell::new();

#[derive(Copy, Clone, PartialEq, Eq)]
enum CanonicalTagKind {
    CoreString,
    CoreNull,
}

enum TagClass {
    Canonical(CanonicalTagKind),
    Core,
    NonCore,
}

#[derive(Clone, Hash, PartialEq, Eq)]
struct HandlerKeyOwned {
    handle: String,
    suffix: String,
}

impl HandlerKeyOwned {
    fn matches(&self, key: HandlerKeyRef<'_>) -> bool {
        self.handle == key.handle && self.suffix == key.suffix
    }
}

#[derive(Copy, Clone, Hash, PartialEq, Eq)]
struct HandlerKeyRef<'a> {
    handle: &'a str,
    suffix: &'a str,
}

impl<'a> From<&'a Tag> for HandlerKeyRef<'a> {
    fn from(tag: &'a Tag) -> Self {
        Self {
            handle: tag.handle.as_str(),
            suffix: tag.suffix.as_str(),
        }
    }
}

struct HandlerEntry {
    key: HandlerKeyOwned,
    handler: Py<PyAny>,
}

enum HandlerStore {
    Small(Vec<HandlerEntry>),
    Large(HashMap<HandlerKeyOwned, Py<PyAny>>),
}

struct HandlerRegistry {
    store: HandlerStore,
}

impl HandlerRegistry {
    fn from_py(_py: Python<'_>, handlers: Option<&Bound<'_, PyAny>>) -> Result<Option<Self>> {
        let Some(obj) = handlers else {
            return Ok(None);
        };

        if obj.is_none() {
            return Ok(None);
        }

        let dict: &Bound<'_, PyDict> = obj.downcast().map_err(|_| {
            PyTypeError::new_err("`handlers` must be a dict mapping tag strings to callables")
        })?;
        if dict.is_empty() {
            return Ok(None);
        }

        const HASHMAP_MIN_LEN: usize = 8;
        let use_hash_map = dict.len() >= HASHMAP_MIN_LEN;

        if use_hash_map {
            let mut handlers_map = HashMap::with_capacity(dict.len());
            for (key_obj, value_obj) in dict.iter() {
                let key_str = key_obj.downcast::<PyString>().map_err(|_| {
                    PyTypeError::new_err("handler keys must be strings or subclasses of str")
                })?;
                let key_text = key_str.to_str()?;
                let key = parse_handler_name(key_text)?;
                if !value_obj.is_callable() {
                    return Err(PyTypeError::new_err(format!(
                        "handler `{}` must be callable",
                        key_text
                    )));
                }
                handlers_map.insert(key, value_obj.unbind());
            }
            return Ok(Some(Self {
                store: HandlerStore::Large(handlers_map),
            }));
        }

        let mut entries: Vec<HandlerEntry> = Vec::with_capacity(dict.len());
        for (key_obj, value_obj) in dict.iter() {
            let key_str = key_obj.downcast::<PyString>().map_err(|_| {
                PyTypeError::new_err("handler keys must be strings or subclasses of str")
            })?;
            let key_text = key_str.to_str()?;
            let key = parse_handler_name(key_text)?;
            if !value_obj.is_callable() {
                return Err(PyTypeError::new_err(format!(
                    "handler `{}` must be callable",
                    key_text
                )));
            }

            if let Some(existing) = entries.iter_mut().find(|entry| entry.key == key) {
                existing.handler = value_obj.unbind();
            } else {
                entries.push(HandlerEntry {
                    key,
                    handler: value_obj.unbind(),
                });
            }
        }

        Ok(Some(Self {
            store: HandlerStore::Small(entries),
        }))
    }

    fn get_for_tag(&self, tag: &Tag) -> Option<&Py<PyAny>> {
        let key_ref = HandlerKeyRef::from(tag);
        match &self.store {
            HandlerStore::Small(entries) => entries
                .iter()
                .find(|entry| entry.key.matches(key_ref))
                .map(|entry| &entry.handler),
            HandlerStore::Large(map) => {
                let lookup = HandlerKeyOwned {
                    handle: key_ref.handle.to_string(),
                    suffix: key_ref.suffix.to_string(),
                };
                map.get(&lookup)
            }
        }
    }

    fn apply(&self, py: Python<'_>, handler: &Py<PyAny>, arg: PyObject) -> Result<PyObject> {
        handler.bind(py).call1((arg,)).map(|obj| obj.unbind())
    }
}

fn parse_handler_name(name: &str) -> Result<HandlerKeyOwned> {
    if let Some((handle, suffix)) = split_tag_name(name) {
        return Ok(HandlerKeyOwned {
            handle: handle.to_string(),
            suffix: suffix.to_string(),
        });
    }
    Err(PyTypeError::new_err(
        "`handlers` keys must be valid YAML tag strings",
    ))
}

fn split_tag_name(name: &str) -> Option<(&str, &str)> {
    if let Some(pos) = name.rfind('!') {
        if pos + 1 < name.len() {
            let (handle, suffix) = name.split_at(pos + 1);
            return Some((handle, suffix));
        }
    }
    if let Some(pos) = name.rfind(':') {
        if pos + 1 < name.len() {
            let (handle, suffix) = name.split_at(pos + 1);
            return Some((handle, suffix));
        }
    }
    None
}

#[pyfunction(signature = (text, multi=false, handlers=None))]
fn parse_yaml(
    py: Python<'_>,
    text: PyObject,
    multi: bool,
    handlers: Option<PyObject>,
) -> Result<PyObject> {
    let handler_registry = HandlerRegistry::from_py(py, handlers.as_ref().map(|obj| obj.bind(py)))?;
    let handlers = handler_registry.as_ref();

    let bound = text.bind(py);
    let joined = join_text(bound)?;
    if joined.is_none() {
        return Ok(py.None());
    }
    let src = joined.as_deref().unwrap();
    let docs = load_yaml_documents(src, multi)?;
    let mut out = docs_to_python(py, docs, multi, handlers)?;
    if !multi {
        if let Some(tag) = leading_tag(src) {
            if let (Some(registry), Ok(parsed_tag)) = (handlers, parse_tag_string(&tag)) {
                if let Some(handler) = registry.get_for_tag(&parsed_tag) {
                    return registry.apply(py, handler, out);
                }
            }
            if !is_tagged_instance(py, &out)? {
                out = make_tagged(py, out, &tag)?;
            }
        }
    }
    Ok(out)
}

#[pyfunction(signature = (path, multi=false, handlers=None))]
fn read_yaml(
    py: Python<'_>,
    path: &str,
    multi: bool,
    handlers: Option<PyObject>,
) -> Result<PyObject> {
    let handler_registry = HandlerRegistry::from_py(py, handlers.as_ref().map(|obj| obj.bind(py)))?;
    let handlers = handler_registry.as_ref();

    let contents = fs::read_to_string(path)
        .map_err(|err| PyIOError::new_err(format!("failed to read `{path}`: {err}")))?;
    let docs = load_yaml_documents(&contents, multi)?;
    docs_to_python(py, docs, multi, handlers)
}

#[pyfunction(signature = (value, multi=false))]
fn format_yaml(py: Python<'_>, value: PyObject, multi: bool) -> Result<String> {
    let bound = value.bind(py);
    let yaml = py_to_yaml(py, bound, false)?;
    let mut output = format_yaml_impl(&yaml, multi)?;
    if multi {
        output.push_str("...\n");
    }
    Ok(yaml_body(&output, multi).to_string())
}

#[pyfunction(signature = (value, path=None, multi=false))]
fn write_yaml(py: Python<'_>, value: PyObject, path: Option<&str>, multi: bool) -> Result<()> {
    let bound = value.bind(py);
    let yaml = py_to_yaml(py, bound, false)?;
    let mut output = format_yaml_impl(&yaml, multi)?;
    if multi {
        output.push_str("...\n");
    } else {
        output.push_str("\n...\n");
    }
    if let Some(path) = path {
        fs::write(path, &output)
            .map_err(|err| PyIOError::new_err(format!("failed to write `{path}`: {err}")))?;
    } else {
        write_to_stdout(&output)?;
    }
    Ok(())
}

fn join_text(text: &Bound<'_, PyAny>) -> Result<Option<String>> {
    if let Ok(s) = text.downcast::<PyString>() {
        return Ok(Some(s.to_str()?.to_owned()));
    }

    if let Ok(seq) = text.downcast::<PySequence>() {
        let len = seq.len()?;
        if len == 0 {
            return Ok(None);
        }
        let mut parts: Vec<String> = Vec::with_capacity(len);
        for idx in 0..len {
            let item = seq.get_item(idx)?;
            let s = item.downcast::<PyString>().map_err(|_| {
                PyTypeError::new_err("`text` sequence must contain only strings without None")
            })?;
            parts.push(s.to_str()?.to_owned());
        }
        return Ok(Some(parts.join("\n")));
    }

    Err(PyTypeError::new_err(
        "`text` must be a string or a sequence of strings",
    ))
}

fn docs_to_python(
    py: Python<'_>,
    mut docs: Vec<Yaml<'_>>,
    multi: bool,
    handlers: Option<&HandlerRegistry>,
) -> Result<PyObject> {
    if multi {
        let mut values = Vec::with_capacity(docs.len());
        for doc in docs.iter_mut() {
            values.push(yaml_to_py(py, doc, false, handlers)?);
        }
        Ok(PyList::new(py, values)?.unbind().into())
    } else {
        match docs.first_mut() {
            Some(doc) => yaml_to_py(py, doc, false, handlers),
            None => Ok(py.None()),
        }
    }
}

fn load_yaml_documents<'input>(text: &'input str, multi: bool) -> Result<Vec<Yaml<'input>>> {
    let mut parser = Parser::new_from_str(text);
    let mut loader = saphyr::YamlLoader::default();
    loader.early_parse(false);
    parser
        .load(&mut loader, multi)
        .map_err(|err| PyValueError::new_err(format!("YAML parse error: {err}")))?;
    Ok(loader.into_documents())
}

fn leading_tag(text: &str) -> Option<String> {
    let trimmed = text.trim_start();
    if !trimmed.starts_with('!') {
        return None;
    }
    let end = trimmed.find(char::is_whitespace).unwrap_or(trimmed.len());
    let tag = &trimmed[..end];
    if tag == "!" {
        return None;
    }
    Some(tag.to_string())
}

fn resolve_representation(node: &mut Yaml, _simplify: bool) {
    let (value, style, tag) = match mem::replace(node, Yaml::BadValue) {
        Yaml::Representation(value, style, tag) => (value, style, tag),
        other => {
            *node = other;
            return;
        }
    };

    let is_plain_empty = style == ScalarStyle::Plain && value.trim().is_empty();

    let parsed = match tag {
        Some(tag) => {
            let owned_tag = tag.into_owned();
            match classify_tag(&owned_tag) {
                TagClass::Canonical(kind) => {
                    if kind == CanonicalTagKind::CoreNull && is_plain_empty {
                        Yaml::Value(Scalar::Null)
                    } else {
                        let canonical_tag = Cow::Owned(make_canonical_tag(kind));
                        Yaml::value_from_cow_and_metadata(value, style, Some(&canonical_tag))
                    }
                }
                TagClass::Core => {
                    let core_tag = Cow::Owned(owned_tag);
                    Yaml::value_from_cow_and_metadata(value, style, Some(&core_tag))
                }
                TagClass::NonCore => Yaml::Tagged(
                    Cow::Owned(owned_tag),
                    Box::new(Yaml::Value(Scalar::String(value))),
                ),
            }
        }
        None if is_plain_empty => Yaml::Value(Scalar::Null),
        None => Yaml::value_from_cow_and_metadata(value, style, None),
    };
    *node = parsed;
}

fn yaml_to_py(
    py: Python<'_>,
    node: &mut Yaml,
    is_key: bool,
    handlers: Option<&HandlerRegistry>,
) -> Result<PyObject> {
    match node {
        Yaml::Value(scalar) => Ok(scalar_to_py(py, scalar)),
        Yaml::Tagged(tag, inner) => convert_tagged(py, tag, inner.as_mut(), is_key, handlers),
        Yaml::Sequence(seq) => sequence_to_py(py, seq, is_key, handlers),
        Yaml::Mapping(map) => mapping_to_py(py, map, is_key, handlers),
        Yaml::Alias(_) => Err(PyValueError::new_err(
            "internal error: encountered unresolved YAML alias node",
        )),
        Yaml::BadValue => Err(PyValueError::new_err(
            "encountered an invalid YAML scalar value",
        )),
        Yaml::Representation(_, _, _) => {
            resolve_representation(node, true);
            yaml_to_py(py, node, is_key, handlers)
        }
    }
}

fn scalar_to_py(py: Python<'_>, scalar: &Scalar) -> PyObject {
    match scalar {
        Scalar::Null => py.None(),
        Scalar::Boolean(v) => (*v)
            .into_py_any(py)
            .expect("bool conversion to Python should not fail"),
        Scalar::Integer(v) => (*v)
            .into_py_any(py)
            .expect("int conversion to Python should not fail"),
        Scalar::FloatingPoint(v) => v
            .into_inner()
            .into_py_any(py)
            .expect("float conversion to Python should not fail"),
        Scalar::String(v) => v
            .as_ref()
            .into_py_any(py)
            .expect("string conversion to Python should not fail"),
    }
}

fn scalar_to_string(scalar: &Scalar) -> String {
    match scalar {
        Scalar::Null => "null".to_string(),
        Scalar::Boolean(v) => v.to_string(),
        Scalar::Integer(v) => v.to_string(),
        Scalar::FloatingPoint(v) => v.into_inner().to_string(),
        Scalar::String(v) => v.to_string(),
    }
}

fn sequence_to_py(
    py: Python<'_>,
    seq: &mut [Yaml],
    is_key: bool,
    handlers: Option<&HandlerRegistry>,
) -> Result<PyObject> {
    let mut values = Vec::with_capacity(seq.len());
    for node in seq {
        resolve_representation(node, true);
        values.push(yaml_to_py(py, node, is_key, handlers)?);
    }
    if is_key {
        Ok(PyTuple::new(py, values)?.unbind().into())
    } else {
        Ok(PyList::new(py, values)?.unbind().into())
    }
}

fn mapping_to_py(
    py: Python<'_>,
    map: &mut Mapping,
    is_key: bool,
    handlers: Option<&HandlerRegistry>,
) -> Result<PyObject> {
    let len = map.len();

    if is_key {
        let mut pairs: Vec<(PyObject, PyObject)> = Vec::with_capacity(len);
        for (mut key, mut value) in mem::take(map) {
            resolve_representation(&mut key, true);
            resolve_representation(&mut value, true);
            let k_obj = yaml_to_py(py, &mut key, true, handlers)?;
            let v_obj = yaml_to_py(py, &mut value, true, handlers)?;
            pairs.push((k_obj, v_obj));
        }
        return Ok(PyTuple::new(py, pairs)?.unbind().into());
    }

    let dict = PyDict::new(py);
    for (mut key, mut value) in mem::take(map) {
        resolve_representation(&mut key, true);
        let key_obj = yaml_to_py(py, &mut key, true, handlers)?;
        // Ensure the key is hashable; propagate Python's TypeError for clarity.
        key_obj.bind(py).hash()?;
        let value_obj = yaml_to_py(py, &mut value, false, handlers)?;
        dict.set_item(key_obj, value_obj)?;
    }
    Ok(dict.into())
}

fn convert_tagged(
    py: Python<'_>,
    tag: &Tag,
    node: &mut Yaml,
    is_key: bool,
    handlers: Option<&HandlerRegistry>,
) -> Result<PyObject> {
    let rendered = render_tag(tag);
    if rendered == "!" {
        if let Yaml::Value(scalar) = node {
            let s = scalar_to_string(scalar);
            return Ok(PyString::new(py, &s).unbind().into());
        }
        return yaml_to_py(py, node, is_key, handlers);
    }
    if let Some(registry) = handlers {
        if let Some(handler) = registry.get_for_tag(tag) {
            let value = yaml_to_py(py, node, is_key, handlers)?;
            return registry.apply(py, handler, value);
        }
    }
    match classify_tag(tag) {
        TagClass::Canonical(_) => yaml_to_py(py, node, is_key, handlers),
        TagClass::Core | TagClass::NonCore => {
            let value = yaml_to_py(py, node, is_key, handlers)?;
            make_tagged(py, value, &rendered)
        }
    }
}

fn make_tagged(py: Python<'_>, value: PyObject, tag: &str) -> Result<PyObject> {
    let cls = TAGGED_CLASS
        .get(py)
        .ok_or_else(|| PyValueError::new_err("Tagged class is not initialized"))?;
    cls.bind(py).call1((value, tag)).map(|obj| obj.into())
}

fn is_tagged_instance(py: Python<'_>, obj: &PyObject) -> Result<bool> {
    if let Some(cls) = TAGGED_CLASS.get(py) {
        obj.bind(py).is_instance(cls.bind(py))
    } else {
        Ok(false)
    }
}

fn canonical_tag_kind(tag: &Tag) -> Option<CanonicalTagKind> {
    match (tag.handle.as_str(), tag.suffix.as_str()) {
        ("tag:yaml.org,2002:", "str") => Some(CanonicalTagKind::CoreString),
        ("!", "str") => Some(CanonicalTagKind::CoreString),
        ("", "str") => Some(CanonicalTagKind::CoreString),
        ("", "!str") => Some(CanonicalTagKind::CoreString),
        ("", "!!str") => Some(CanonicalTagKind::CoreString),
        ("", "tag:yaml.org,2002:str") => Some(CanonicalTagKind::CoreString),
        ("tag:yaml.org,2002:", "null") => Some(CanonicalTagKind::CoreNull),
        ("", "null") => Some(CanonicalTagKind::CoreNull),
        ("", "!null") => Some(CanonicalTagKind::CoreNull),
        ("", "!!null") => Some(CanonicalTagKind::CoreNull),
        ("", "tag:yaml.org,2002:null") => Some(CanonicalTagKind::CoreNull),
        _ => None,
    }
}

fn classify_tag(tag: &Tag) -> TagClass {
    if let Some(kind) = canonical_tag_kind(tag) {
        TagClass::Canonical(kind)
    } else if tag.is_yaml_core_schema() {
        TagClass::Core
    } else {
        TagClass::NonCore
    }
}

fn make_canonical_tag(kind: CanonicalTagKind) -> Tag {
    let suffix = match kind {
        CanonicalTagKind::CoreString => "str",
        CanonicalTagKind::CoreNull => "null",
    };
    Tag {
        handle: "tag:yaml.org,2002:".to_string(),
        suffix: suffix.to_string(),
    }
}

fn render_tag(tag: &Tag) -> String {
    let mut rendered = String::with_capacity(tag.handle.len() + tag.suffix.len());
    rendered.push_str(tag.handle.as_str());
    rendered.push_str(tag.suffix.as_str());
    rendered
}

fn yaml_body(yaml: &str, multi: bool) -> &str {
    if multi || !yaml.starts_with("---\n") {
        yaml
    } else {
        &yaml[4..]
    }
}

fn emit_yaml_documents(docs: &[Yaml<'static>], multi: bool) -> Result<String> {
    if docs.is_empty() {
        return Ok(String::new());
    }
    let mut output = String::new();
    let mut emitter = YamlEmitter::new(&mut output);
    emitter.multiline_strings(true);
    if multi {
        emitter
            .dump_docs(docs)
            .map_err(|err| PyValueError::new_err(err.to_string()))?;
    } else {
        emitter
            .dump(&docs[0])
            .map_err(|err| PyValueError::new_err(err.to_string()))?;
    }
    Ok(output)
}

fn format_yaml_impl(value: &Yaml<'static>, multi: bool) -> Result<String> {
    if multi {
        let docs: Vec<Yaml<'static>> = match value {
            Yaml::Sequence(seq) => seq.clone(),
            Yaml::Value(Scalar::Null) => Vec::new(),
            _ => {
                return Err(PyTypeError::new_err(
                    "`value` must be a sequence when `multi=True`",
                ))
            }
        };
        if docs.is_empty() {
            return Ok(String::new());
        }
        emit_yaml_documents(&docs, true)
    } else {
        emit_yaml_documents(std::slice::from_ref(value), false)
    }
}

fn write_to_stdout(content: &str) -> Result<()> {
    let mut stdout = io::stdout();
    stdout
        .write_all(content.as_bytes())
        .map_err(|err| PyIOError::new_err(format!("failed to write to stdout: {err}")))?;
    stdout
        .flush()
        .map_err(|err| PyIOError::new_err(format!("failed to flush stdout: {err}")))
}

#[allow(clippy::only_used_in_recursion)]
fn py_to_yaml(py: Python<'_>, obj: &Bound<'_, PyAny>, is_key: bool) -> Result<Yaml<'static>> {
    if obj.is_none() {
        return Ok(Yaml::Value(Scalar::Null));
    }

    if let Ok(b) = obj.extract::<bool>() {
        return Ok(Yaml::Value(Scalar::Boolean(b)));
    }

    if let Ok(i) = obj.extract::<i128>() {
        if i < i64::MIN as i128 || i > i64::MAX as i128 {
            return Err(PyValueError::new_err("integer out of range for YAML"));
        }
        return Ok(Yaml::Value(Scalar::Integer(i as i64)));
    }

    if let Ok(f) = obj.extract::<f64>() {
        if f.is_nan() {
            return Ok(Yaml::Value(Scalar::Null));
        }
        return Ok(Yaml::Value(Scalar::FloatingPoint(f.into())));
    }

    if let Ok(s) = obj.downcast::<PyString>() {
        return Ok(Yaml::Value(Scalar::String(Cow::Owned(
            s.to_str()?.to_string(),
        ))));
    }

    if is_tagged(py, obj)? {
        let value_obj = obj.getattr("value")?;
        let tag_obj = obj.getattr("tag")?;
        let tag_str: &str = tag_obj.downcast::<PyString>()?.to_str()?;
        let tag = parse_tag_string(tag_str)?;
        let inner = py_to_yaml(py, &value_obj, is_key)?;
        return Ok(Yaml::Tagged(Cow::Owned(tag), Box::new(inner)));
    }

    if let Ok(dict) = obj.downcast::<PyDict>() {
        let mut mapping = Mapping::with_capacity(dict.len());
        for (key_obj, value_obj) in dict.iter() {
            let key_yaml = py_to_yaml(py, &key_obj, true)?;
            let value_yaml = py_to_yaml(py, &value_obj, false)?;
            mapping.insert(key_yaml, value_yaml);
        }
        return Ok(Yaml::Mapping(mapping));
    }

    if let Ok(seq) = obj.downcast::<PySequence>() {
        // Avoid treating strings as sequences since PySequence accepts them.
        if obj.downcast::<PyString>().is_ok() {
            // Already handled above; unreachable.
        } else {
            let len = seq.len()?;
            let mut values = Vec::with_capacity(len);
            for idx in 0..len {
                let item = seq.get_item(idx)?;
                values.push(py_to_yaml(py, &item, false)?);
            }
            return Ok(Yaml::Sequence(values));
        }
    }

    Err(PyTypeError::new_err("unsupported type for YAML conversion"))
}

fn parse_tag_string(tag: &str) -> Result<Tag> {
    if tag.is_empty() {
        return Err(PyValueError::new_err("tag must not be empty"));
    }
    if let Some(pos) = tag.rfind('!') {
        if pos + 1 >= tag.len() {
            return Err(PyValueError::new_err(format!("invalid YAML tag `{tag}`")));
        }
        let handle = &tag[..pos];
        let suffix = &tag[pos + 1..];
        if handle.is_empty() {
            Ok(Tag {
                handle: "!".to_string(),
                suffix: suffix.to_string(),
            })
        } else {
            Ok(Tag {
                handle: handle.to_string(),
                suffix: suffix.to_string(),
            })
        }
    } else {
        Err(PyValueError::new_err(format!("invalid YAML tag `{tag}`")))
    }
}

fn is_tagged(py: Python<'_>, obj: &Bound<'_, PyAny>) -> Result<bool> {
    if let Some(cls) = TAGGED_CLASS.get(py) {
        obj.is_instance(cls.bind(py))
    } else {
        Ok(false)
    }
}

fn init_tagged_class(py: Python<'_>, module: &Bound<'_, PyModule>) -> Result<()> {
    let cls = TAGGED_CLASS.get_or_try_init(py, || -> Result<Py<PyAny>> {
        let code = r#"
from dataclasses import dataclass

@dataclass(frozen=True)
class Tagged:
    value: object
    tag: str
"#;
        let filename = CString::new("tagged.py").unwrap();
        let modname = CString::new("tagged").unwrap();
        let code_cstr = CString::new(code).unwrap();
        let tagged_mod = PyModule::from_code(
            py,
            code_cstr.as_c_str(),
            filename.as_c_str(),
            modname.as_c_str(),
        )?;
        let cls = tagged_mod.getattr("Tagged")?;
        Ok(cls.unbind())
    })?;

    module.add("Tagged", cls.clone_ref(py))?;
    Ok(())
}

#[pymodule]
pub fn yaml12(py: Python<'_>, m: &Bound<'_, PyModule>) -> Result<()> {
    init_tagged_class(py, m)?;
    m.add_function(wrap_pyfunction!(parse_yaml, m)?)?;
    m.add_function(wrap_pyfunction!(read_yaml, m)?)?;
    m.add_function(wrap_pyfunction!(format_yaml, m)?)?;
    m.add_function(wrap_pyfunction!(write_yaml, m)?)?;
    Ok(())
}
