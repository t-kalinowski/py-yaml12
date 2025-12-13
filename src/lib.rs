use pyo3::exceptions::{PyAttributeError, PyIOError, PyTypeError, PyValueError};
use pyo3::ffi;
use pyo3::prelude::*;
use pyo3::sync::GILOnceCell;
use pyo3::types::{
    PyAnyMethods, PyBool, PyByteArray, PyBytes, PyDict, PyDictMethods, PyFloat, PyInt, PyIterator,
    PyList, PyModule, PySequence, PySequenceMethods, PyString, PyTuple,
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
use std::path::PathBuf;

type Result<T> = PyResult<T>;
type BuiltinTypes = (Py<PyAny>, Py<PyAny>, Py<PyAny>, Py<PyAny>);

static YAML_CLASS: GILOnceCell<Py<PyAny>> = GILOnceCell::new();
static ABC_TYPES: GILOnceCell<(Py<PyAny>, Py<PyAny>)> = GILOnceCell::new();
static BUILTIN_TYPES: GILOnceCell<BuiltinTypes> = GILOnceCell::new();
const GIL_RELEASE_MIN_PARSE_LEN: usize = 2048;
const GIL_RELEASE_MIN_EMIT_DOCS: usize = 4;
const GIL_RELEASE_MIN_EMIT_COLLECTION_LEN: usize = 32;

fn pathlike_to_pathbuf(obj: &Bound<'_, PyAny>) -> Result<Option<PathBuf>> {
    match obj.extract::<PathBuf>() {
        Ok(path) => Ok(Some(path)),
        Err(err) => {
            if err.is_instance_of::<PyTypeError>(obj.py()) {
                Ok(None)
            } else {
                Err(err)
            }
        }
    }
}

#[derive(Clone, PartialEq, Eq, Hash)]
struct HandlerKeyOwned {
    handle: String,
    suffix: String,
}

impl HandlerKeyOwned {
    fn from_parts(handle: &str, suffix: &str) -> Self {
        Self {
            handle: handle.to_string(),
            suffix: suffix.to_string(),
        }
    }
}

type HandlerMap = HashMap<String, HashMap<String, Py<PyAny>>>;

struct HandlerRegistry {
    map: HandlerMap,
}

impl HandlerRegistry {
    fn from_py(handlers: Option<&Bound<'_, PyAny>>) -> Result<Option<Self>> {
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

        let mut handlers_map: HandlerMap = HashMap::with_capacity(dict.len());
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
            handlers_map
                .entry(key.handle)
                .or_default()
                .insert(key.suffix, value_obj.unbind());
        }

        Ok(Some(Self { map: handlers_map }))
    }

    fn get_for_tag(&self, tag: &Tag) -> Option<&Py<PyAny>> {
        self.map
            .get(tag.handle.as_str())
            .and_then(|suffixes| suffixes.get(tag.suffix.as_str()))
    }

    fn apply(&self, py: Python<'_>, handler: &Py<PyAny>, arg: PyObject) -> Result<PyObject> {
        handler.call1(py, (arg,))
    }
}

fn builtin_types(py: Python<'_>) -> Result<&BuiltinTypes> {
    BUILTIN_TYPES.get_or_try_init(py, || -> Result<_> {
        Ok((
            py.get_type::<PyBool>().unbind().into(),
            py.get_type::<PyInt>().unbind().into(),
            py.get_type::<PyFloat>().unbind().into(),
            py.get_type::<PyString>().unbind().into(),
        ))
    })
}

fn handler_registry_from_arg(
    py: Python<'_>,
    handlers: Option<&PyObject>,
) -> Result<Option<HandlerRegistry>> {
    match handlers {
        None => Ok(None),
        Some(obj) => {
            let bound = obj.bind(py);
            if bound.is_none() {
                Ok(None)
            } else {
                HandlerRegistry::from_py(Some(bound))
            }
        }
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
///     object: Parsed value(s); tagged nodes become `Yaml` when no handler matches or when used for unhashable mapping keys.
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
    let handler_registry = handler_registry_from_arg(py, handlers.as_ref())?;
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
        let docs = load_yaml_documents(py, src, multi)?;
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

    Err(PyTypeError::new_err(
        "`text` must be a string or a sequence of strings",
    ))
}

#[pyfunction(signature = (path, multi=false, handlers=None))]
/// Read a YAML file from `path` and parse it.
///
/// Args:
///     path (str | os.PathLike | object with .read): Filesystem path or readable object whose `.read()` returns str/bytes.
///     multi (bool): Return a list of documents when true; otherwise a single document or None for empty input.
///     handlers (dict[str, Callable] | None): Optional tag handlers for values and keys; matching handlers receive the parsed value.
///
/// Returns:
///     object: Parsed value(s); tagged nodes become `Yaml` when no handler matches or when used for unhashable mapping keys.
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
    let handler_registry = handler_registry_from_arg(py, handlers.as_ref())?;
    let handlers = handler_registry.as_ref();

    let bound = path.bind(py);
    if let Ok(s) = bound.downcast::<PyString>() {
        let path_str = s.to_str()?;
        let contents = py
            .allow_threads(|| fs::read_to_string(path_str))
            .map_err(|err| PyIOError::new_err(format!("failed to read `{path_str}`: {err}")))?;
        let docs = load_yaml_documents(py, &contents, multi)?;
        return docs_to_python(py, docs, multi, handlers);
    }

    if let Some(path_buf) = pathlike_to_pathbuf(bound)? {
        let contents = py
            .allow_threads(|| fs::read_to_string(&path_buf))
            .map_err(|err| {
                PyIOError::new_err(format!("failed to read `{}`: {err}", path_buf.display()))
            })?;
        let docs = load_yaml_documents(py, &contents, multi)?;
        return docs_to_python(py, docs, multi, handlers);
    }

    if let Ok(read) = bound.getattr("read") {
        let reader = read.unbind();
        let reader_bound = reader.bind(py);
        let read_result = reader_bound.call0().or_else(|err| {
            if err.is_instance_of::<PyTypeError>(py) {
                reader_bound.call1((-1isize,))
            } else {
                Err(err)
            }
        })?;

        if let Ok(s) = read_result.downcast::<PyString>() {
            let text = s.to_str()?;
            let docs = load_yaml_documents(py, text, multi)?;
            return docs_to_python(py, docs, multi, handlers);
        }

        if let Ok(bytes) = read_result.downcast::<PyBytes>() {
            let text = std::str::from_utf8(bytes.as_bytes()).map_err(|err| {
                PyValueError::new_err(format!(
                    "connection.read() returned non-UTF-8 bytes ({err})"
                ))
            })?;
            let docs = load_yaml_documents(py, text, multi)?;
            return docs_to_python(py, docs, multi, handlers);
        }

        return Err(PyTypeError::new_err(
            "`read` must return str or bytes objects",
        ));
    }

    Err(PyTypeError::new_err(
        "`path` must be a string, a path-like object, or an object with .read()",
    ))
}

#[pyfunction(signature = (value, multi=false))]
/// Serialize a Python value to a YAML string.
///
/// Args:
///     value (object): Python value or Yaml to serialize; for `multi` the value must be a sequence of documents.
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
fn format_yaml(py: Python<'_>, value: PyObject, multi: bool) -> Result<PyObject> {
    let bound = value.bind(py);
    let yaml = py_to_yaml(py, bound, false)?;
    let mut output = format_yaml_impl(py, &yaml, multi)?;
    if multi {
        output.push_str("...\n");
        return PyString::new(py, output.as_str()).into_py_any(py);
    }
    let body = output.strip_prefix("---\n").unwrap_or(output.as_str());
    PyString::new(py, body).into_py_any(py)
}

#[pyfunction(signature = (value, path=None, multi=false))]
/// Write a Python value to YAML at `path` or stdout.
///
/// Args:
///     value (object): Python value or Yaml to serialize; for `multi` the value must be a sequence of documents.
///     path (str | os.PathLike | file-like | None): Destination path or object with `.write()`; when None the YAML is written to stdout.
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
    let mut output = format_yaml_impl(py, &yaml, multi)?;
    if multi {
        output.push_str("...\n");
    } else {
        output.push_str("\n...\n");
    }
    let Some(path_obj) = path else {
        write_to_stdout(py, &output)?;
        return Ok(());
    };

    let bound_path = path_obj.bind(py);
    if bound_path.is_none() {
        write_to_stdout(py, &output)?;
        return Ok(());
    }

    if let Ok(path_str) = bound_path.downcast::<PyString>() {
        let p = path_str.to_str()?;
        py.allow_threads(|| fs::write(p, &output))
            .map_err(|err| PyIOError::new_err(format!("failed to write `{p}`: {err}")))?;
        return Ok(());
    }

    if let Some(path_buf) = pathlike_to_pathbuf(bound_path)? {
        py.allow_threads(|| fs::write(&path_buf, &output))
            .map_err(|err| {
                PyIOError::new_err(format!("failed to write `{}`: {err}", path_buf.display()))
            })?;
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
        "`path` must be None, a string or path-like path, or an object with .write()",
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
        let docs = load_yaml_documents(py, src, true)?;
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
        let reader_bound = reader.bind(py);
        let read_result = reader_bound.call0().or_else(|err| {
            if err.is_instance_of::<PyTypeError>(py) {
                reader_bound.call1((-1isize,))
            } else {
                Err(err)
            }
        })?;

        if let Ok(s) = read_result.downcast::<PyString>() {
            let text = s.to_str()?;
            let docs = load_yaml_documents(py, text, true)?;
            println!("{docs:#?}");
            return Ok(());
        }

        if let Ok(bytes) = read_result.downcast::<PyBytes>() {
            let text = std::str::from_utf8(bytes.as_bytes()).map_err(|err| {
                PyValueError::new_err(format!(
                    "connection.read() returned non-UTF-8 bytes ({err})"
                ))
            })?;
            let docs = load_yaml_documents(py, text, true)?;
            println!("{docs:#?}");
            return Ok(());
        }

        return Err(PyTypeError::new_err(
            "`read` must return str or bytes objects",
        ));
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

fn should_release_gil_for_parse(text_len: usize) -> bool {
    text_len >= GIL_RELEASE_MIN_PARSE_LEN
}

fn should_release_gil_for_emit(value: &Yaml<'static>, multi: bool) -> bool {
    if multi {
        matches!(
            value,
            Yaml::Sequence(seq) if seq.len() >= GIL_RELEASE_MIN_EMIT_DOCS
        )
    } else {
        match value {
            Yaml::Sequence(seq) => seq.len() >= GIL_RELEASE_MIN_EMIT_COLLECTION_LEN,
            Yaml::Mapping(map) => map.len() >= GIL_RELEASE_MIN_EMIT_COLLECTION_LEN,
            _ => false,
        }
    }
}

fn load_yaml_documents<'py, 'input>(
    py: Python<'py>,
    text: &'input str,
    multi: bool,
) -> Result<Vec<Yaml<'input>>> {
    let mut parser = Parser::new_from_str(text);
    let mut loader = saphyr::YamlLoader::default();
    loader.early_parse(false);
    let mut load = || parser.load(&mut loader, multi);
    let result = if should_release_gil_for_parse(text.len()) {
        py.allow_threads(load)
    } else {
        load()
    };
    result.map_err(|err| PyValueError::new_err(format!("YAML parse error: {err}")))?;
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

fn resolve_representation(node: &mut Yaml) {
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
        Yaml::Representation(_, _, _) => {
            resolve_representation(node);
            yaml_to_py(py, node, is_key, handlers)
        }
        Yaml::Value(scalar) => Ok(scalar_to_py(py, scalar)),
        Yaml::Sequence(seq) => {
            let value = sequence_to_py(py, seq, handlers)?;
            if is_key {
                make_yaml_node(py, value, None)
            } else {
                Ok(value)
            }
        }
        Yaml::Mapping(map) => {
            let value = mapping_to_py(py, map, handlers)?;
            if is_key {
                make_yaml_node(py, value, None)
            } else {
                Ok(value)
            }
        }
        Yaml::Tagged(tag, inner) => convert_tagged(py, tag, inner.as_mut(), is_key, handlers),
        Yaml::Alias(_) => Err(PyValueError::new_err(
            "internal error: encountered unresolved YAML alias node",
        )),
        Yaml::BadValue => Err(PyValueError::new_err(
            "encountered an invalid YAML scalar value",
        )),
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
    handlers: Option<&HandlerRegistry>,
) -> Result<PyObject> {
    let mut values = Vec::with_capacity(seq.len());
    for node in seq {
        values.push(yaml_to_py(py, node, false, handlers)?);
    }
    Ok(PyList::new(py, values)?.unbind().into())
}

fn mapping_to_py(
    py: Python<'_>,
    map: &mut Mapping,
    handlers: Option<&HandlerRegistry>,
) -> Result<PyObject> {
    let dict = PyDict::new(py);
    for (mut key, mut value) in map.drain() {
        let key_obj = yaml_to_py(py, &mut key, true, handlers)?;
        let value_obj = yaml_to_py(py, &mut value, false, handlers)?;
        dict.set_item(key_obj, value_obj)?;
    }
    Ok(dict.into())
}

fn render_tag_cached<'a>(rendered: &'a mut Option<String>, tag: &Tag) -> &'a str {
    rendered.get_or_insert_with(|| render_tag(tag)).as_str()
}

fn handler_result_needs_wrap(py: Python<'_>, obj: &Bound<'_, PyAny>) -> Result<bool> {
    if is_yaml_node(py, obj)? || obj.is_none() {
        return Ok(false);
    }

    if obj.is_instance_of::<PyDict>() || obj.is_instance_of::<PyList>() {
        return Ok(true);
    }

    if obj.is_instance_of::<PyString>()
        || obj.is_instance_of::<PyBytes>()
        || obj.is_instance_of::<PyByteArray>()
    {
        return Ok(false);
    }

    let (mapping_cls, sequence_cls) = abc_types(py)?;
    if obj.is_instance(mapping_cls.bind(py))? || obj.is_instance(sequence_cls.bind(py))? {
        return hash_is_disabled(py, obj);
    }

    hash_is_disabled(py, obj)
}

fn hash_is_disabled(py: Python<'_>, obj: &Bound<'_, PyAny>) -> Result<bool> {
    obj.getattr("__hash__")
        .map(|hash_attr| hash_attr.is_none())
        .or_else(|err| {
            if err.is_instance_of::<PyAttributeError>(py) {
                Ok(false)
            } else {
                Err(err)
            }
        })
}

fn convert_tagged(
    py: Python<'_>,
    tag: &Tag,
    node: &mut Yaml,
    is_key: bool,
    handlers: Option<&HandlerRegistry>,
) -> Result<PyObject> {
    let mut rendered_tag: Option<String> = None;

    if let Some(registry) = handlers {
        if let Some(handler) = registry.get_for_tag(tag) {
            // Convert inner node in value mode to avoid pre-wrapping keys; the tag logic below
            // handles hashability and tag preservation.
            let value = yaml_to_py(py, node, false, handlers)?;
            let handled = registry.apply(py, handler, value)?;
            if is_key && handler_result_needs_wrap(py, handled.bind(py))? {
                return make_yaml_node(py, handled, None);
            }
            return Ok(handled);
        }
    }

    let value = yaml_to_py(py, node, false, handlers)?;

    let normalized_suffix = normalized_suffix(tag.suffix.as_str());
    if is_core_tag(tag) {
        return match normalized_suffix {
            "str" | "null" | "bool" | "int" | "float" => Ok(value),
            "seq" | "map" => {
                if is_key {
                    make_yaml_node(py, value, None)
                } else {
                    Ok(value)
                }
            }
            "timestamp" | "set" | "omap" | "pairs" | "binary" => {
                let rendered = render_tag_cached(&mut rendered_tag, tag);
                make_yaml_node(py, value, Some(rendered))
            }
            _ => {
                let rendered = render_tag_cached(&mut rendered_tag, tag);
                Err(PyValueError::new_err(format!(
                    "unsupported core-schema tag `{rendered}`"
                )))
            }
        };
    }

    let rendered = render_tag_cached(&mut rendered_tag, tag);
    make_yaml_node(py, value, Some(rendered))
}

fn make_yaml_node(py: Python<'_>, value: PyObject, tag: Option<&str>) -> Result<PyObject> {
    let cls = YAML_CLASS
        .get(py)
        .ok_or_else(|| PyValueError::new_err("Yaml class is not initialized"))?;
    if let Some(tag) = tag {
        cls.bind(py).call1((value, tag)).map(|obj| obj.into())
    } else {
        cls.bind(py).call1((value,)).map(|obj| obj.into())
    }
}

fn is_yaml_node(py: Python<'_>, obj: &Bound<'_, PyAny>) -> Result<bool> {
    if let Some(cls) = YAML_CLASS.get(py) {
        obj.is_instance(cls.bind(py))
    } else {
        Ok(false)
    }
}

fn abc_types(py: Python<'_>) -> Result<&(Py<PyAny>, Py<PyAny>)> {
    ABC_TYPES.get_or_try_init(py, || -> Result<(Py<PyAny>, Py<PyAny>)> {
        let abc = PyModule::import(py, "collections.abc")?;
        Ok((
            abc.getattr("Mapping")?.unbind(),
            abc.getattr("Sequence")?.unbind(),
        ))
    })
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
fn emit_yaml_documents(
    docs: &[Yaml<'static>],
    multi: bool,
) -> std::result::Result<String, saphyr::EmitError> {
    if docs.is_empty() {
        return Ok(String::new());
    }
    let mut output = String::new();
    let mut emitter = YamlEmitter::new(&mut output);
    emitter.multiline_strings(true);
    if multi {
        emitter.dump_docs(docs)?;
    } else {
        emitter.dump(&docs[0])?;
    }
    Ok(output)
}

fn format_yaml_impl(py: Python<'_>, value: &Yaml<'static>, multi: bool) -> Result<String> {
    let emit = |docs: &[Yaml<'static>], multi: bool| {
        if should_release_gil_for_emit(value, multi) {
            py.allow_threads(|| emit_yaml_documents(docs, multi))
        } else {
            emit_yaml_documents(docs, multi)
        }
    };
    if multi {
        match value {
            Yaml::Sequence(seq) => {
                if seq.is_empty() {
                    return Ok(String::from("---\n"));
                }
                emit(seq, true).map_err(|err| PyValueError::new_err(err.to_string()))
            }
            Yaml::Value(Scalar::Null) => Ok(String::from("---\n")),
            _ => Err(PyTypeError::new_err(
                "`value` must be a sequence when `multi=True`",
            )),
        }
    } else {
        emit(std::slice::from_ref(value), false)
            .map_err(|err| PyValueError::new_err(err.to_string()))
    }
}

fn write_to_stdout(py: Python<'_>, content: &str) -> Result<()> {
    let ptr = unsafe { ffi::PySys_GetObject(c"stdout".as_ptr()) };
    if ptr.is_null() {
        return write_to_stdout_fallback(py, content);
    }

    let stdout = unsafe { Bound::from_borrowed_ptr(py, ptr) };
    if stdout.is_none() {
        return write_to_stdout_fallback(py, content);
    }

    if stdout.call_method1("write", (content,)).is_err() {
        // If sys.stdout is malformed, fall back to the real stdout and
        // suppress the Python-level exception.
        return write_to_stdout_fallback(py, content);
    }

    // Best-effort flush; never raise for malformed stdout.
    if let Ok(flush) = stdout.getattr("flush") {
        if !flush.is_none() {
            let _ = flush.call0();
        }
    }

    Ok(())
}

fn write_to_stdout_fallback(py: Python<'_>, content: &str) -> Result<()> {
    py.allow_threads(|| {
        let mut stdout = io::stdout();
        stdout.write_all(content.as_bytes())?;
        stdout.flush()
    })
    .map_err(|err| PyIOError::new_err(format!("failed to write to stdout: {err}")))
}

#[allow(clippy::only_used_in_recursion)]
fn py_to_yaml(py: Python<'_>, obj: &Bound<'_, PyAny>, is_key: bool) -> Result<Yaml<'static>> {
    if obj.is_none() {
        return Ok(Yaml::Value(Scalar::Null));
    }

    if is_yaml_node(py, obj)? {
        let value_obj = obj.getattr("value")?;
        let tag_obj = obj.getattr("tag")?;
        if tag_obj.is_none() {
            return py_to_yaml(py, &value_obj, is_key);
        }
        let tag_str = tag_obj.downcast::<PyString>()?.to_str()?;
        let tag = parse_tag_string(tag_str)?;
        let inner = py_to_yaml(py, &value_obj, is_key)?;
        return if is_core_scalar_tag(&tag) {
            Ok(inner)
        } else {
            Ok(Yaml::Tagged(Cow::Owned(tag), Box::new(inner)))
        };
    }

    let ty = obj.get_type();
    let (bool_type, int_type, float_type, str_type) = builtin_types(py)?;

    if ty.is(bool_type.bind(py)) {
        let b: bool = obj.extract()?;
        return Ok(Yaml::Value(Scalar::Boolean(b)));
    }

    if ty.is(int_type.bind(py)) {
        let i: i128 = obj.extract()?;
        if i < i64::MIN as i128 || i > i64::MAX as i128 {
            return Err(PyValueError::new_err("integer out of range for YAML"));
        }
        return Ok(Yaml::Value(Scalar::Integer(i as i64)));
    }

    if ty.is(float_type.bind(py)) {
        let f: f64 = obj.extract()?;
        if f.is_nan() {
            return Ok(Yaml::Value(Scalar::Null));
        }
        return Ok(Yaml::Value(Scalar::FloatingPoint(f.into())));
    }

    if ty.is(str_type.bind(py)) {
        let s: &Bound<'_, PyString> = obj.downcast()?;
        return Ok(Yaml::Value(Scalar::String(Cow::Owned(
            s.to_str()?.to_owned(),
        ))));
    }

    if obj.is_instance_of::<PyDict>() {
        let dict: &Bound<'_, PyDict> = obj.downcast()?;
        let mut mapping = Mapping::with_capacity(dict.len());
        for (key_obj, value_obj) in dict.iter() {
            let key_yaml = py_to_yaml(py, &key_obj, true)?;
            let value_yaml = py_to_yaml(py, &value_obj, false)?;
            mapping.insert(key_yaml, value_yaml);
        }
        return Ok(Yaml::Mapping(mapping));
    }

    if obj.is_instance_of::<PyList>() {
        let list: &Bound<'_, PyList> = obj.downcast()?;
        let mut values = Vec::with_capacity(list.len());
        for item in list.iter() {
            values.push(py_to_yaml(py, &item, false)?);
        }
        return Ok(Yaml::Sequence(values));
    }

    if obj.is_instance_of::<PyTuple>() {
        let tuple: &Bound<'_, PyTuple> = obj.downcast()?;
        let mut values = Vec::with_capacity(tuple.len());
        for item in tuple.iter() {
            values.push(py_to_yaml(py, &item, false)?);
        }
        return Ok(Yaml::Sequence(values));
    }

    if let Ok(seq) = obj.downcast::<PySequence>() {
        if obj.downcast::<PyString>().is_err() {
            let len = seq.len()?;
            let mut values = Vec::with_capacity(len);
            for idx in 0..len {
                let item = seq.get_item(idx)?;
                values.push(py_to_yaml(py, &item, false)?);
            }
            return Ok(Yaml::Sequence(values));
        }
    }

    let (mapping_cls, sequence_cls) = abc_types(py)?;
    if obj.is_instance(mapping_cls.bind(py))? {
        let items = obj.getattr("items")?;
        let iter = PyIterator::from_object(items.as_any())?;
        let mut mapping = Mapping::new();
        for item in iter {
            let pair = item?;
            let tuple: Bound<'_, PyTuple> = pair
                .downcast_into()
                .map_err(|_| PyTypeError::new_err("mapping items must be (key, value) pairs"))?;
            if tuple.len() != 2 {
                return Err(PyTypeError::new_err(
                    "mapping items must be (key, value) pairs",
                ));
            }
            let key_yaml = py_to_yaml(py, &tuple.get_item(0)?, true)?;
            let value_yaml = py_to_yaml(py, &tuple.get_item(1)?, false)?;
            mapping.insert(key_yaml, value_yaml);
        }
        return Ok(Yaml::Mapping(mapping));
    }

    if obj.is_instance(sequence_cls.bind(py))?
        && !obj.is_instance_of::<PyString>()
        && !obj.is_instance_of::<PyBytes>()
        && !obj.is_instance_of::<PyByteArray>()
    {
        let iter = PyIterator::from_object(obj.as_any())?;
        let mut values = Vec::new();
        for item in iter {
            values.push(py_to_yaml(py, &item?, false)?);
        }
        return Ok(Yaml::Sequence(values));
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

fn init_python_helpers(py: Python<'_>, module: &Bound<'_, PyModule>) -> Result<()> {
    YAML_CLASS.get_or_try_init(py, || -> Result<Py<PyAny>> {
        let code = r#"
from dataclasses import dataclass
from collections.abc import Mapping, Sequence

def _freeze(obj):
    if isinstance(obj, Yaml):
        return ("yaml", obj.tag, _freeze(obj.value))
    if isinstance(obj, Mapping):
        return ("map", tuple(sorted((_freeze(k), _freeze(v)) for k, v in obj.items())))
    if isinstance(obj, Sequence) and not isinstance(obj, (str, bytes, bytearray)):
        return ("seq", tuple(_freeze(x) for x in obj))
    try:
        hash(obj)
        return obj
    except TypeError:
        return ("unhashable", id(obj))

@dataclass(frozen=True)
class Yaml:
    """Tagged node or hashable wrapper for unhashable mapping keys."""
    value: Mapping | Sequence | float | int | bool | str | None
    tag: str | None = None

    def __post_init__(self):
        frozen = ("tagged", self.tag, _freeze(self.value))
        object.__setattr__(self, "_frozen", frozen)
        object.__setattr__(self, "_hash", hash(frozen))

    def __hash__(self):
        return self._hash

    def __eq__(self, other):
        if not isinstance(other, Yaml):
            return NotImplemented
        return self._frozen == other._frozen

    def _proxy_target(self):
        target = self.value.value if isinstance(self.value, Yaml) else self.value
        if isinstance(target, (Mapping, Sequence)) and not isinstance(
            target, (str, bytes, bytearray)
        ):
            return target
        return None

    def __getitem__(self, key):
        target = self._proxy_target()
        if target is not None:
            return target[key]
        raise TypeError("Yaml.value does not support indexing")

    def __iter__(self):
        target = self._proxy_target()
        if target is not None:
            return iter(target)
        raise TypeError("Yaml.value is not iterable")

    def __len__(self):
        target = self._proxy_target()
        if target is not None:
            return len(target)
        raise TypeError("Yaml.value has no len()")

    def __repr__(self):
        tag = f"{self.tag!r}, " if self.tag is not None else ""
        return f"Yaml({tag}{self.value!r})"

"#;
        let filename = CString::new("py_helpers.py").unwrap();
        let modname = CString::new("py_helpers").unwrap();
        let code_cstr = CString::new(code).unwrap();
        let helpers_mod = PyModule::from_code(
            py,
            code_cstr.as_c_str(),
            filename.as_c_str(),
            modname.as_c_str(),
        )?;
        let yaml_cls = helpers_mod.getattr("Yaml")?;
        Ok(yaml_cls.unbind())
    })?;

    let yaml_cls = YAML_CLASS
        .get(py)
        .ok_or_else(|| PyValueError::new_err("Yaml class is not initialized"))?;
    module.add("Yaml", yaml_cls.clone_ref(py))?;
    Ok(())
}

#[pymodule]
pub fn yaml12(py: Python<'_>, m: &Bound<'_, PyModule>) -> Result<()> {
    init_python_helpers(py, m)?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
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
            resolve_representation(&mut parsed);
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
            resolve_representation(&mut parsed);
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

    #[test]
    fn gil_release_thresholds_are_consistent() {
        assert!(!should_release_gil_for_parse(GIL_RELEASE_MIN_PARSE_LEN - 1));
        assert!(should_release_gil_for_parse(GIL_RELEASE_MIN_PARSE_LEN));

        let small_multi = Yaml::Sequence(vec![
            Yaml::Value(Scalar::Null);
            GIL_RELEASE_MIN_EMIT_DOCS - 1
        ]);
        let big_multi = Yaml::Sequence(vec![Yaml::Value(Scalar::Null); GIL_RELEASE_MIN_EMIT_DOCS]);
        assert!(!should_release_gil_for_emit(&small_multi, true));
        assert!(should_release_gil_for_emit(&big_multi, true));

        let mut big_map = Mapping::with_capacity(GIL_RELEASE_MIN_EMIT_COLLECTION_LEN);
        for idx in 0..GIL_RELEASE_MIN_EMIT_COLLECTION_LEN {
            big_map.insert(
                Yaml::Value(Scalar::Integer(idx as i64)),
                Yaml::Value(Scalar::Null),
            );
        }
        let small_map = Yaml::Mapping(Mapping::with_capacity(
            GIL_RELEASE_MIN_EMIT_COLLECTION_LEN - 1,
        ));
        let big_map_yaml = Yaml::Mapping(big_map);
        assert!(!should_release_gil_for_emit(&small_map, false));
        assert!(should_release_gil_for_emit(&big_map_yaml, false));
    }
}
