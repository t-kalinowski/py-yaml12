use pyo3::exceptions::{PyIOError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::sync::GILOnceCell;
use pyo3::types::{
    PyAnyMethods, PyBytes, PyDict, PyDictMethods, PyList, PyModule, PySequence, PySequenceMethods,
    PyString, PyTuple,
};
use pyo3::Bound;
use pyo3::IntoPyObjectExt;
use saphyr::{Mapping, Scalar, Tag, Yaml, YamlEmitter};
use saphyr_parser::{Parser, ScalarStyle};
use std::borrow::Cow;
use std::cell::RefCell;
use std::collections::HashMap;
use std::ffi::CString;
use std::fs;
use std::io::{self, Write};
use std::mem;
use std::rc::Rc;

type Result<T> = PyResult<T>;

static TAGGED_CLASS: GILOnceCell<Py<PyAny>> = GILOnceCell::new();
const READ_CHUNK_SIZE: usize = 8192;

#[derive(Clone, PartialEq, Eq, Hash)]
struct HandlerKeyOwned {
    handle: String,
    suffix: String,
}

impl PartialEq<HandlerKeyRef<'_>> for HandlerKeyOwned {
    fn eq(&self, other: &HandlerKeyRef<'_>) -> bool {
        self.handle == other.handle && self.suffix == other.suffix
    }
}

impl PartialEq<HandlerKeyOwned> for HandlerKeyRef<'_> {
    fn eq(&self, other: &HandlerKeyOwned) -> bool {
        self.handle == other.handle && self.suffix == other.suffix
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

impl HandlerKeyOwned {
    fn from_parts(handle: &str, suffix: &str) -> Self {
        Self {
            handle: handle.to_string(),
            suffix: suffix.to_string(),
        }
    }

    fn from_tag(tag: &Tag) -> Self {
        Self::from_parts(tag.handle.as_str(), tag.suffix.as_str())
    }

    fn is_non_specific(&self) -> bool {
        self.handle.is_empty() && self.suffix == "!"
    }
}

impl<'a> HandlerKeyRef<'a> {
    fn is_non_specific(self) -> bool {
        self.handle.is_empty() && self.suffix == "!"
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
    non_specific: Option<Py<PyAny>>,
}

impl HandlerRegistry {
    fn from_py(py: Python<'_>, handlers: Option<&Bound<'_, PyAny>>) -> Result<Option<Self>> {
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

        let mut non_specific: Option<Py<PyAny>> = None;

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
                let handler = value_obj.unbind();
                if key.is_non_specific() {
                    non_specific = Some(handler.clone_ref(py));
                }
                handlers_map.insert(key, handler);
            }
            return Ok(Some(Self {
                store: HandlerStore::Large(handlers_map),
                non_specific,
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

            let handler = value_obj.unbind();
            if key.is_non_specific() {
                non_specific = Some(handler.clone_ref(py));
            }

            if let Some(existing) = entries.iter_mut().find(|entry| entry.key == key) {
                existing.handler = handler;
            } else {
                entries.push(HandlerEntry { key, handler });
            }
        }

        Ok(Some(Self {
            store: HandlerStore::Small(entries),
            non_specific,
        }))
    }

    fn get_for_tag(&self, tag: &Tag) -> Option<&Py<PyAny>> {
        match &self.store {
            HandlerStore::Small(entries) => {
                let key_ref = HandlerKeyRef::from(tag);
                entries
                    .iter()
                    .find(|entry| entry.key == key_ref)
                    .map(|entry| &entry.handler)
            }
            HandlerStore::Large(map) => {
                let lookup = HandlerKeyOwned::from_tag(tag);
                map.get(&lookup)
            }
        }
        .or_else(|| {
            if HandlerKeyRef::from(tag).is_non_specific() {
                self.non_specific.as_ref()
            } else {
                None
            }
        })
    }

    fn apply(&self, py: Python<'_>, handler: &Py<PyAny>, arg: PyObject) -> Result<PyObject> {
        handler.bind(py).call1((arg,)).map(|obj| obj.unbind())
    }
}

fn parse_handler_name(name: &str) -> Result<HandlerKeyOwned> {
    if let Some((handle, suffix)) = split_tag_name(name) {
        return Ok(HandlerKeyOwned::from_parts(handle, suffix));
    }
    Err(PyTypeError::new_err(
        "`handlers` keys must be valid YAML tag strings",
    ))
}

fn split_tag_name(name: &str) -> Option<(&str, &str)> {
    if name == "!" {
        return Some(("", "!"));
    }
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
/// Parse YAML text into Python values.
///
/// Args:
///     text (str | Sequence[str]): YAML text or sequence joined with newlines.
///     multi (bool): Return a list of documents when true; otherwise a single document or None for empty input.
///     handlers (dict[str, Callable] | None): Optional tag handlers for values and keys; matching handlers receive the parsed value.
///
/// Returns:
///     object: Parsed value(s); non-core tags become Tagged when no handler matches.
///
/// Raises:
///     ValueError: On YAML parse errors or invalid tag strings.
///     TypeError: When inputs are the wrong type or handlers are not callables.
///     Exception: Propagated directly from user-provided handlers.
///
/// Examples:
///     >>> parse_yaml('foo: 1\nbar: true')
///     {'foo': 1, 'bar': True}
fn parse_yaml(
    py: Python<'_>,
    text: PyObject,
    multi: bool,
    handlers: Option<PyObject>,
) -> Result<PyObject> {
    let handler_registry = HandlerRegistry::from_py(py, handlers.as_ref().map(|obj| obj.bind(py)))?;
    let handlers = handler_registry.as_ref();

    let bound = text.bind(py);
    if let Ok(s) = bound.downcast::<PyString>() {
        let src = s.to_str()?;
        if src.is_empty() {
            return if multi {
                Ok(PyList::empty(py).unbind().into())
            } else {
                Ok(py.None())
            };
        }
        let docs = load_yaml_documents(src, multi)?;
        return docs_to_python(py, docs, multi, handlers);
    }

    if let Ok(seq) = bound.downcast::<PySequence>() {
        let len = seq.len()?;
        if len == 0 {
            return if multi {
                Ok(PyList::empty(py).unbind().into())
            } else {
                Ok(py.None())
            };
        }
        let mut lines: Vec<Py<PyString>> = Vec::with_capacity(len);
        for idx in 0..len {
            let item = seq.get_item(idx)?;
            let s: Bound<'_, PyString> = item.downcast_into().map_err(|_| {
                PyTypeError::new_err("`text` sequence must contain only strings without None")
            })?;
            lines.push(s.unbind());
        }
        let iter = JoinedLinesIter::new(py, &lines);
        let docs = load_yaml_documents_iter(iter, multi)?;
        return docs_to_python(py, docs, multi, handlers);
    }

    if let Ok(read) = bound.getattr("read") {
        let reader = read.unbind();
        let error_flag = Rc::new(RefCell::new(None));
        let iter = PyReadIter::new(py, reader, error_flag.clone());
        let docs = load_yaml_documents_iter(iter, multi)?;
        if let Some(err) = error_flag.borrow_mut().take() {
            return Err(err);
        }
        return docs_to_python(py, docs, multi, handlers);
    }

    Err(PyTypeError::new_err(
        "`text` must be a string, a sequence of strings, or an object with .read()",
    ))
}

#[pyfunction(signature = (path, multi=false, handlers=None))]
/// Read a YAML file from `path` and parse it.
///
/// Args:
///     path (str | object with .read): Filesystem path or readable object yielding str/bytes.
///     multi (bool): Return a list of documents when true; otherwise a single document or None for empty input.
///     handlers (dict[str, Callable] | None): Optional tag handlers for values and keys; matching handlers receive the parsed value.
///
/// Returns:
///     object: Parsed value(s); non-core tags become Tagged when no handler matches.
///
/// Raises:
///     IOError: When the file cannot be read.
///     ValueError: On YAML parse errors or invalid tag strings.
///     TypeError: When handlers are not callables or inputs are the wrong type.
///     Exception: Propagated directly from user-provided handlers.
///
/// Examples:
///     >>> read_yaml('config.yml')
///     {'debug': True}
fn read_yaml(
    py: Python<'_>,
    path: PyObject,
    multi: bool,
    handlers: Option<PyObject>,
) -> Result<PyObject> {
    let handler_registry = HandlerRegistry::from_py(py, handlers.as_ref().map(|obj| obj.bind(py)))?;
    let handlers = handler_registry.as_ref();

    let bound = path.bind(py);
    if let Ok(s) = bound.downcast::<PyString>() {
        let path_str = s.to_str()?;
        let contents = fs::read_to_string(path_str)
            .map_err(|err| PyIOError::new_err(format!("failed to read `{path_str}`: {err}")))?;
        let docs = load_yaml_documents(&contents, multi)?;
        return docs_to_python(py, docs, multi, handlers);
    }

    if let Ok(read) = bound.getattr("read") {
        let reader = read.unbind();
        let error_flag = Rc::new(RefCell::new(None));
        let iter = PyReadIter::new(py, reader, error_flag.clone());
        let docs = load_yaml_documents_iter(iter, multi)?;
        if let Some(err) = error_flag.borrow_mut().take() {
            return Err(err);
        }
        return docs_to_python(py, docs, multi, handlers);
    }

    Err(PyTypeError::new_err(
        "`path` must be a string or an object with .read()",
    ))
}

#[pyfunction(signature = (value, multi=false))]
/// Serialize a Python value to a YAML string.
///
/// Args:
///     value (object): Python value or Tagged to serialize; for `multi` the value must be a sequence of documents.
///     multi (bool): Emit a multi-document stream when true; otherwise a single document.
///
/// Returns:
///     str: YAML text; multi-document streams end with `...`.
///
/// Raises:
///     TypeError: When `multi` is true and value is not a sequence, or unsupported types are provided.
///
/// Examples:
///     >>> format_yaml({'foo': 1})
///     'foo: 1\n'
///     >>> format_yaml(['first', 'second'], multi=True).endswith('...\n')
///     True
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
/// Write a Python value to YAML at `path` or stdout.
///
/// Args:
///     value (object): Python value or Tagged to serialize; for `multi` the value must be a sequence of documents.
///     path (str | file-like | None): Destination path or object with `.write()`; when None the YAML is written to stdout.
///     multi (bool): Emit a multi-document stream when true; otherwise a single document.
///
/// Returns:
///     None
///
/// Raises:
///     IOError: When writing to the file or stdout fails.
///     TypeError: When `multi` is true and value is not a sequence, or unsupported types are provided.
///
/// Examples:
///     >>> write_yaml({'foo': 1}, path='out.yml')
///     >>> Path('out.yml').exists()
///     True
///     >>> write_yaml(['first', 'second'], multi=True)  # prints YAML ending with '...'
fn write_yaml(py: Python<'_>, value: PyObject, path: Option<PyObject>, multi: bool) -> Result<()> {
    let bound = value.bind(py);
    let yaml = py_to_yaml(py, bound, false)?;
    let mut output = format_yaml_impl(&yaml, multi)?;
    if multi {
        output.push_str("...\n");
    } else {
        output.push_str("\n...\n");
    }
    let Some(path_obj) = path else {
        write_to_stdout(&output)?;
        return Ok(());
    };

    let bound_path = path_obj.bind(py);
    if bound_path.is_none() {
        write_to_stdout(&output)?;
        return Ok(());
    }

    if let Ok(path_str) = bound_path.downcast::<PyString>() {
        let p = path_str.to_str()?;
        fs::write(p, &output)
            .map_err(|err| PyIOError::new_err(format!("failed to write `{p}`: {err}")))?;
        return Ok(());
    }

    if let Ok(write) = bound_path.getattr("write") {
        let writer = write.unbind();
        let try_str = writer.bind(py).call1((output.as_str(),));
        if let Err(err) = try_str {
            if err.is_instance_of::<PyTypeError>(py) {
                let bytes = PyBytes::new(py, output.as_bytes());
                writer.bind(py).call1((bytes,))?;
            } else {
                return Err(err);
            }
        }
        return Ok(());
    }

    Err(PyTypeError::new_err(
        "`path` must be None, a string path, or an object with .write()",
    ))
}

#[pyfunction]
/// Debug helper: pretty-print parsed YAML nodes without converting to Python values.
fn _dbg_yaml(py: Python<'_>, text: PyObject) -> Result<()> {
    let bound = text.bind(py);
    if let Ok(s) = bound.downcast::<PyString>() {
        let src = s.to_str()?;
        if src.is_empty() {
            return Ok(());
        }
        let docs = load_yaml_documents(src, true)?;
        println!("{docs:#?}");
        return Ok(());
    }

    if let Ok(seq) = bound.downcast::<PySequence>() {
        let len = seq.len()?;
        if len == 0 {
            return Ok(());
        }
        let mut lines: Vec<Py<PyString>> = Vec::with_capacity(len);
        for idx in 0..len {
            let item = seq.get_item(idx)?;
            let s: Bound<'_, PyString> = item.downcast_into().map_err(|_| {
                PyTypeError::new_err("`text` sequence must contain only strings without None")
            })?;
            lines.push(s.unbind());
        }
        let iter = JoinedLinesIter::new(py, &lines);
        let docs = load_yaml_documents_iter(iter, true)?;
        println!("{docs:#?}");
        return Ok(());
    }

    if let Ok(read) = bound.getattr("read") {
        let reader = read.unbind();
        let error_flag = Rc::new(RefCell::new(None));
        let iter = PyReadIter::new(py, reader, error_flag.clone());
        let docs = load_yaml_documents_iter(iter, true)?;
        if let Some(err) = error_flag.borrow_mut().take() {
            return Err(err);
        }
        println!("{docs:#?}");
        return Ok(());
    }

    Err(PyTypeError::new_err(
        "`text` must be a string, a sequence of strings, or an object with .read()",
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

fn load_yaml_documents_iter<'input, I>(iter: I, multi: bool) -> Result<Vec<Yaml<'input>>>
where
    I: Iterator<Item = char> + 'input,
{
    let mut parser = Parser::new_from_iter(iter);
    let mut loader = saphyr::YamlLoader::default();
    loader.early_parse(false);
    parser
        .load(&mut loader, multi)
        .map_err(|err| PyValueError::new_err(format!("YAML parse error: {err}")))?;
    Ok(loader.into_documents())
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
            if tag.is_yaml_core_schema() {
                match tag.suffix.as_str() {
                    "str" => Yaml::value_from_cow_and_metadata(value, style, Some(&tag)),
                    "null" => {
                        if is_plain_empty {
                            Yaml::Value(Scalar::Null)
                        } else {
                            Yaml::value_from_cow_and_metadata(value, style, Some(&tag))
                        }
                    }
                    "binary" | "set" | "omap" | "pairs" | "timestamp" => {
                        Yaml::Tagged(tag, Box::new(Yaml::Value(Scalar::String(value))))
                    }
                    _ => {
                        let parsed =
                            Yaml::value_from_cow_and_metadata(value.clone(), style, Some(&tag));
                        if matches!(parsed, Yaml::BadValue)
                            && !matches!(
                                tag.suffix.as_str(),
                                "bool" | "int" | "float" | "null" | "str"
                            )
                        {
                            Yaml::Tagged(tag, Box::new(Yaml::Value(Scalar::String(value))))
                        } else {
                            parsed
                        }
                    }
                }
            } else {
                Yaml::Tagged(tag, Box::new(Yaml::Value(Scalar::String(value))))
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
    if let Some(registry) = handlers {
        if let Some(handler) = registry.get_for_tag(tag) {
            let value = yaml_to_py(py, node, is_key, handlers)?;
            return registry.apply(py, handler, value);
        }
    }

    let value = yaml_to_py(py, node, is_key, handlers)?;

    let normalized_suffix = normalized_suffix(tag.suffix.as_str());
    if is_core_tag(tag) {
        return match normalized_suffix {
            "str" | "null" | "bool" | "int" | "float" | "seq" | "map" => Ok(value),
            "timestamp" | "set" | "omap" | "pairs" | "binary" => {
                let rendered = render_tag(tag);
                make_tagged(py, value, &rendered)
            }
            _ => {
                let rendered = render_tag(tag);
                Err(PyValueError::new_err(format!(
                    "unsupported core-schema tag `{rendered}`"
                )))
            }
        };
    }

    let rendered = render_tag(tag);
    make_tagged(py, value, &rendered)
}

fn make_tagged(py: Python<'_>, value: PyObject, tag: &str) -> Result<PyObject> {
    let cls = TAGGED_CLASS
        .get(py)
        .ok_or_else(|| PyValueError::new_err("Tagged class is not initialized"))?;
    cls.bind(py).call1((value, tag)).map(|obj| obj.into())
}

#[cfg(test)]
fn is_core_string_tag(tag: &Tag) -> bool {
    tag.is_yaml_core_schema() && tag.suffix.as_str() == "str"
}

#[cfg(test)]
fn is_core_null_tag(tag: &Tag) -> bool {
    tag.is_yaml_core_schema() && tag.suffix.as_str() == "null"
}

fn is_core_scalar_tag(tag: &Tag) -> bool {
    if !is_core_tag(tag) {
        return false;
    }
    matches!(
        normalized_suffix(tag.suffix.as_str()),
        "str" | "null" | "bool" | "int" | "float" | "seq" | "map"
    )
}

fn normalized_suffix(suffix: &str) -> &str {
    let suffix = suffix.trim_start_matches('!');
    suffix.strip_prefix("tag:yaml.org,2002:").unwrap_or(suffix)
}

fn is_core_tag(tag: &Tag) -> bool {
    tag.is_yaml_core_schema()
        || (tag.handle.as_str() == "!" && tag.suffix.as_str().starts_with('!'))
        || (tag.handle.is_empty() && tag.suffix.as_str().starts_with("tag:yaml.org,2002:"))
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

struct PyReadIter<'py> {
    py: Python<'py>,
    reader: Py<PyAny>,
    chars: std::str::Chars<'static>,
    done: bool,
    error: Rc<RefCell<Option<PyErr>>>,
    buffer: String,
}

impl<'py> PyReadIter<'py> {
    fn new(py: Python<'py>, reader: Py<PyAny>, error: Rc<RefCell<Option<PyErr>>>) -> Self {
        Self {
            py,
            reader,
            chars: "".chars(),
            done: false,
            error,
            buffer: String::new(),
        }
    }

    fn record_error(&self, err: PyErr) {
        *self.error.borrow_mut() = Some(err);
    }

    fn fetch_next_chunk(&mut self) -> bool {
        let read = self.reader.bind(self.py);
        let result = read.call1((READ_CHUNK_SIZE,)).or_else(|err| {
            if err.is_instance_of::<PyTypeError>(self.py) {
                read.call0()
            } else {
                Err(err)
            }
        });

        let obj = match result {
            Ok(obj) => obj,
            Err(err) => {
                self.record_error(err);
                return false;
            }
        };

        if obj.is_instance_of::<PyString>() {
            let s: Bound<'_, PyString> = obj.downcast_into().expect("type checked above");
            let text = match s.to_str() {
                Ok(text) => text,
                Err(err) => {
                    self.record_error(err);
                    return false;
                }
            };
            if text.is_empty() {
                return false;
            }
            self.buffer.clear();
            self.buffer.push_str(text);
            // Safe because `self.buffer` lives for the lifetime of the iterator.
            self.chars = unsafe {
                std::mem::transmute::<std::str::Chars<'_>, std::str::Chars<'static>>(
                    self.buffer.chars(),
                )
            };
            true
        } else if obj.is_instance_of::<PyBytes>() {
            let bytes: Bound<'_, PyBytes> = obj.downcast_into().expect("type checked above");
            let slice = bytes.as_bytes();
            if slice.is_empty() {
                return false;
            }
            let text = match std::str::from_utf8(slice) {
                Ok(text) => text,
                Err(err) => {
                    self.record_error(PyValueError::new_err(format!(
                        "connection.read() returned non-UTF-8 bytes ({err})"
                    )));
                    return false;
                }
            };
            self.buffer.clear();
            self.buffer.push_str(text);
            // Safe because `self.buffer` lives for the lifetime of the iterator.
            self.chars = unsafe {
                std::mem::transmute::<std::str::Chars<'_>, std::str::Chars<'static>>(
                    self.buffer.chars(),
                )
            };
            true
        } else {
            self.record_error(PyTypeError::new_err(
                "`read` must return str or bytes objects",
            ));
            false
        }
    }
}

impl<'py> Iterator for PyReadIter<'py> {
    type Item = char;

    fn next(&mut self) -> Option<Self::Item> {
        if self.done || self.error.borrow().is_some() {
            return None;
        }
        loop {
            if let Some(ch) = self.chars.next() {
                return Some(ch);
            }
            if !self.fetch_next_chunk() {
                self.done = true;
                return None;
            }
        }
    }
}

struct JoinedLinesIter<'py, 'a: 'py> {
    py: Python<'py>,
    lines: &'a [Py<PyString>],
    next_line: usize,
    current: std::str::Chars<'py>,
    has_current: bool,
}

impl<'py, 'a: 'py> JoinedLinesIter<'py, 'a> {
    fn new(py: Python<'py>, lines: &'a [Py<PyString>]) -> Self {
        let mut iter = Self {
            py,
            lines,
            next_line: 0,
            current: "".chars(),
            has_current: false,
        };
        iter.advance_line();
        iter
    }

    fn advance_line(&mut self) {
        if self.next_line >= self.lines.len() {
            self.has_current = false;
            self.current = "".chars();
            return;
        }
        let line = self.lines[self.next_line].bind(self.py);
        self.next_line += 1;
        self.current = line
            .to_str()
            .expect("PyString should contain valid UTF-8")
            .chars();
        self.has_current = true;
    }
}

impl<'py, 'a> Iterator for JoinedLinesIter<'py, 'a> {
    type Item = char;

    fn next(&mut self) -> Option<Self::Item> {
        if !self.has_current {
            return None;
        }
        if let Some(ch) = self.current.next() {
            return Some(ch);
        }
        if self.next_line >= self.lines.len() {
            self.has_current = false;
            return None;
        }
        self.advance_line();
        Some('\n')
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
            s.to_str()?.to_owned(),
        ))));
    }

    if is_tagged(py, obj)? {
        let value_obj = obj.getattr("value")?;
        let tag_obj = obj.getattr("tag")?;
        let tag_str = tag_obj.downcast::<PyString>()?.to_str()?;
        let tag = parse_tag_string(tag_str)?;
        let inner = py_to_yaml(py, &value_obj, is_key)?;
        return if is_core_scalar_tag(&tag) {
            Ok(inner)
        } else {
            Ok(Yaml::Tagged(Cow::Owned(tag), Box::new(inner)))
        };
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
    let trimmed = tag.trim();
    if trimmed.is_empty() {
        return Err(PyValueError::new_err("tag must not be empty"));
    }

    let invalid_tag_error = || PyValueError::new_err(format!("invalid YAML tag `{trimmed}`"));

    if !trimmed.contains('!') && !trimmed.contains(':') {
        return Err(invalid_tag_error());
    }

    let tag = if trimmed == "!" {
        Tag {
            handle: String::new(),
            suffix: "!".to_string(),
        }
    } else if let Some(rest) = trimmed.strip_prefix("!!") {
        if rest.is_empty() {
            return Err(invalid_tag_error());
        }
        let mut suffix = String::with_capacity(rest.len() + 1);
        suffix.push('!');
        suffix.push_str(rest);
        Tag {
            handle: "!".to_string(),
            suffix,
        }
    } else if let Some(rest) = trimmed.strip_prefix('!') {
        if rest.is_empty() {
            return Err(invalid_tag_error());
        }
        Tag {
            handle: "!".to_string(),
            suffix: rest.to_string(),
        }
    } else if let Some((handle, suffix)) = trimmed.rsplit_once('!') {
        if suffix.is_empty() {
            return Err(invalid_tag_error());
        }
        Tag {
            handle: handle.to_string(),
            suffix: suffix.to_string(),
        }
    } else {
        Tag {
            handle: String::new(),
            suffix: trimmed.to_string(),
        }
    };

    // saphyr cannot emit a bare tag represented as handle="" / suffix="!".
    // Normalize to handle="!" / suffix="" so round-tripping `!` works.
    if tag.handle.is_empty() && tag.suffix.as_str() == "!" {
        Ok(Tag {
            handle: "!".to_string(),
            suffix: String::new(),
        })
    } else {
        Ok(tag)
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
    m.add_function(wrap_pyfunction!(_dbg_yaml, m)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use saphyr::{LoadableYamlNode, Scalar, Tag};

    #[derive(Debug, Copy, Clone, PartialEq, Eq)]
    enum ParsedValueKind {
        String,
        Boolean,
    }

    fn load_scalar(input: &str) -> Yaml<'_> {
        let mut docs = Yaml::load_from_str(input).expect("parser should load tagged scalar");
        docs.pop().expect("expected one document")
    }

    #[test]
    fn canonical_string_tags_cover_all_forms() {
        let canonical_string = Tag {
            handle: "tag:yaml.org,2002:".to_string(),
            suffix: "str".to_string(),
        };
        assert!(is_core_string_tag(&canonical_string));

        let cases = [
            ("!!str true", ParsedValueKind::String),
            ("!str true", ParsedValueKind::Boolean),
            ("!<str> true", ParsedValueKind::Boolean),
            ("!<!str> true", ParsedValueKind::Boolean),
            ("!<!!str> true", ParsedValueKind::Boolean),
            ("!<tag:yaml.org,2002:str> true", ParsedValueKind::String),
        ];

        for (input, expected_value) in cases {
            let mut parsed = load_scalar(input);
            resolve_representation(&mut parsed, true);
            match parsed {
                Yaml::Value(Scalar::String(value)) => {
                    assert_eq!(
                        expected_value,
                        ParsedValueKind::String,
                        "input `{input}` should resolve to string value"
                    );
                    assert_eq!(value.as_ref(), "true");
                }
                Yaml::Tagged(tag, inner) => {
                    assert_eq!(
                        is_core_string_tag(&tag),
                        tag.is_yaml_core_schema()
                            && normalized_suffix(tag.suffix.as_str()) == "str",
                        "input `{input}` canonical detection should match core `str` suffix",
                    );
                    match (expected_value, inner.as_ref()) {
                        (ParsedValueKind::Boolean, Yaml::Value(Scalar::Boolean(value))) => {
                            assert!(
                                *value,
                                "input `{input}` should parse to boolean `true` when not core"
                            );
                        }
                        (expected, other) => {
                            panic!(
                                "input `{input}` expected value kind {expected:?}, got {other:?}"
                            )
                        }
                    }
                }
                other => panic!("input `{input}` expected tagged or string value, got {other:?}"),
            }
        }
    }

    #[test]
    fn canonical_null_tags_cover_all_forms() {
        let canonical_null = Tag {
            handle: "tag:yaml.org,2002:".to_string(),
            suffix: "null".to_string(),
        };
        assert!(is_core_null_tag(&canonical_null));

        let cases = [
            "!!null null",
            "!<null> null",
            "!<!null> null",
            "!<!!null> null",
            "!<tag:yaml.org,2002:null> null",
        ];

        for input in cases {
            let mut parsed = load_scalar(input);
            resolve_representation(&mut parsed, true);
            match parsed {
                Yaml::Value(Scalar::Null) => {
                    // Canonical null scalars should not carry tags.
                }
                Yaml::Tagged(tag, inner) => {
                    assert_eq!(
                        is_core_null_tag(&tag),
                        tag.is_yaml_core_schema()
                            && normalized_suffix(tag.suffix.as_str()) == "null",
                        "input `{input}` canonical detection should match core `null` suffix",
                    );
                    assert!(
                        matches!(inner.as_ref(), Yaml::Value(Scalar::Null)),
                        "input `{input}` should parse to tagged null scalar"
                    );
                }
                other => panic!("input `{input}` expected null scalar, got {other:?}"),
            }
        }
    }
}
