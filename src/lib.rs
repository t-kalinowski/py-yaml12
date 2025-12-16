use pyo3::exceptions::{PyAttributeError, PyIOError, PyTypeError, PyValueError};
use pyo3::ffi;
use pyo3::prelude::*;
use pyo3::sync::PyOnceLock;
use pyo3::types::{
    PyAnyMethods, PyBool, PyByteArray, PyBytes, PyDict, PyDictMethods, PyFloat, PyInt, PyIterator,
    PyList, PyModule, PySequence, PySequenceMethods, PyString, PyTuple, PyType,
};
use pyo3::Bound;
use pyo3::IntoPyObjectExt;
use saphyr::{Mapping, Scalar, Tag, Yaml, YamlEmitter};
use saphyr_parser::{Parser, ScalarStyle};
use std::borrow::Cow;
use std::collections::HashMap;
use std::fs;
use std::io::{self, Write};
use std::mem;
use std::path::PathBuf;

type Result<T> = PyResult<T>;
type BuiltinTypes = (Py<PyAny>, Py<PyAny>, Py<PyAny>, Py<PyAny>);

static YAML_CLASS: PyOnceLock<Py<PyAny>> = PyOnceLock::new();
static ABC_TYPES: PyOnceLock<(Py<PyAny>, Py<PyAny>)> = PyOnceLock::new();
static BUILTIN_TYPES: PyOnceLock<BuiltinTypes> = PyOnceLock::new();
const GIL_RELEASE_MIN_PARSE_LEN: usize = 2048;
const GIL_RELEASE_MIN_EMIT_DOCS: usize = 4;
const GIL_RELEASE_MIN_EMIT_COLLECTION_LEN: usize = 32;

fn unexpected_item_description(obj: &Bound<'_, PyAny>) -> String {
    let ty = obj
        .get_type()
        .name()
        .map(|name| name.to_string())
        .unwrap_or_else(|_| "unknown".to_string());
    let mut repr = obj
        .repr()
        .and_then(|s| s.to_str().map(|text| text.to_string()))
        .unwrap_or_else(|_| "<repr failed>".to_string());
    const REPR_LIMIT: usize = 200;
    if repr.len() > REPR_LIMIT {
        repr.truncate(REPR_LIMIT);
        repr.push_str("...");
    }
    format!("{ty}: {repr}")
}

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

type HandlerMap = HashMap<String, Py<PyAny>>;

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

        let dict: &Bound<'_, PyDict> = obj.cast().map_err(|_| {
            PyTypeError::new_err("`handlers` must be a dict mapping tag strings to callables")
        })?;

        if dict.is_empty() {
            return Ok(None);
        }

        let mut handlers_map: HandlerMap = HashMap::with_capacity(dict.len());
        for (key_obj, value_obj) in dict.iter() {
            let key_str = key_obj.cast::<PyString>().map_err(|_| {
                PyTypeError::new_err("handler keys must be strings or subclasses of str")
            })?;
            let key_text = key_str.to_str()?;
            let key = normalize_handler_tag_string(key_text)?;
            if !value_obj.is_callable() {
                return Err(PyTypeError::new_err(format!(
                    "handler `{}` must be callable",
                    key_text
                )));
            }
            handlers_map.insert(key, value_obj.unbind());
        }

        Ok(Some(Self { map: handlers_map }))
    }

    fn get_for_tag(&self, tag: &str) -> Option<&Py<PyAny>> {
        self.map.get(tag)
    }

    fn apply(&self, py: Python<'_>, handler: &Py<PyAny>, arg: Py<PyAny>) -> Result<Py<PyAny>> {
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
    handlers: Option<&Py<PyAny>>,
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

fn normalize_handler_tag_string(name: &str) -> Result<String> {
    let trimmed = name.trim();
    if trimmed.is_empty() {
        return Err(PyTypeError::new_err(
            "handler keys must be non-empty strings",
        ));
    }
    if trimmed != name {
        return Err(PyTypeError::new_err(
            "handler keys must not contain leading/trailing whitespace",
        ));
    }
    if trimmed.chars().any(|c| c.is_whitespace()) {
        return Err(PyTypeError::new_err(
            "handler keys must not contain whitespace",
        ));
    }

    // Accept shorthand forms and normalize to the tag strings produced by `render_tag`.
    if let Some(rest) = trimmed.strip_prefix("!!") {
        if rest.is_empty() {
            return Err(PyTypeError::new_err(
                "`handlers` keys must be valid YAML tag strings",
            ));
        }
        return Ok(format!("tag:yaml.org,2002:{rest}"));
    }

    let normalized = if let Some(uri) = trimmed.strip_prefix("!<").and_then(|s| s.strip_suffix('>'))
    {
        if uri.is_empty() {
            return Err(PyTypeError::new_err(
                "`handlers` keys must be valid YAML tag strings",
            ));
        }
        uri
    } else {
        trimmed
    };

    Ok(normalize_simple_tag_name_for_api(normalized).into_owned())
}

fn is_simple_local_tag_name(name: &str) -> bool {
    if name.is_empty() {
        return false;
    }
    name.bytes().all(|b| {
        matches!(
            b,
            b'0'..=b'9' | b'a'..=b'z' | b'A'..=b'Z' | b'_' | b'-' | b'.'
        )
    })
}

fn normalize_simple_tag_name_for_api<'a>(tag: &'a str) -> Cow<'a, str> {
    if tag.starts_with('!') || !is_simple_local_tag_name(tag) {
        Cow::Borrowed(tag)
    } else {
        let mut normalized = String::with_capacity(tag.len() + 1);
        normalized.push('!');
        normalized.push_str(tag);
        Cow::Owned(normalized)
    }
}

#[pyfunction]
fn _normalize_tag(py: Python<'_>, tag: Py<PyAny>) -> Result<Py<PyAny>> {
    let bound = tag.bind(py);
    let tag_str: &Bound<'_, PyString> = bound
        .cast()
        .map_err(|_| PyTypeError::new_err("`tag` must be a string"))?;

    let text = tag_str.to_str()?;
    let normalized_input = text
        .strip_prefix("!<")
        .and_then(|rest| rest.strip_suffix('>'))
        .filter(|inner| !inner.is_empty())
        .unwrap_or(text);

    let normalized = normalize_simple_tag_name_for_api(normalized_input);
    if normalized.as_ref() == text {
        Ok(tag)
    } else {
        Ok(PyString::new(py, normalized.as_ref()).unbind().into_any())
    }
}

#[pyfunction]
fn _set_yaml_class(py: Python<'_>, cls: Py<PyAny>) -> Result<()> {
    let bound = cls.bind(py);
    bound
        .cast::<PyType>()
        .map_err(|_| PyTypeError::new_err("`cls` must be a Python type"))?;

    if let Some(existing) = YAML_CLASS.get(py) {
        if existing.as_ptr() == cls.as_ptr() {
            return Ok(());
        }
        return Err(PyValueError::new_err("Yaml class is already initialized"));
    }

    YAML_CLASS
        .set(py, cls)
        .map_err(|_| PyValueError::new_err("Yaml class is already initialized"))
}

#[pyfunction(signature = (text, multi=false, handlers=None))]
/// Parse YAML text into Python values.
///
/// Args:
///     text (str | Iterable[str]): YAML text, or an iterable yielding text chunks.
///         Chunks are concatenated exactly as provided (no implicit separators are inserted).
///     multi (bool): Return a list of documents when true; otherwise a single document or None for empty input.
///     handlers (dict[str, Callable] | None): Optional tag handlers for values and keys; matching handlers receive the parsed value.
///
/// Returns:
///     object: Parsed value(s): YAML mappings become `dict`, sequences become `list`, and scalars
///         resolve using the YAML 1.2 core schema to `bool`/`int`/`float`/`None`/`str`. Unhashable
///         mapping keys (for example `list`/`dict`) are wrapped in the lightweight `Yaml` dataclass
///         to keep them hashable. Tagged nodes without a matching handler are also wrapped in `Yaml`
///         so the tag can be preserved.
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
    text: Py<PyAny>,
    multi: bool,
    handlers: Option<Py<PyAny>>,
) -> Result<Py<PyAny>> {
    let handler_registry = handler_registry_from_arg(py, handlers.as_ref())?;
    let handlers = handler_registry.as_ref();

    let bound = text.bind(py);
    if let Ok(s) = bound.cast::<PyString>() {
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

    if let Ok(seq) = bound.cast::<PySequence>() {
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
            let s: Bound<'_, PyString> = item.cast_into().map_err(|err| {
                let item = err.into_inner();
                PyTypeError::new_err(format!(
                    "`text` sequence must contain only strings (index {idx} got {})",
                    unexpected_item_description(&item)
                ))
            })?;
            lines.push(s.unbind());
        }

        let mut slices: Vec<&str> = Vec::with_capacity(lines.len());
        let mut total_bytes: usize = 0;
        for line in &lines {
            let slice = line.bind(py).to_str()?;
            total_bytes = total_bytes.saturating_add(slice.len());
            slices.push(slice);
        }
        let release_gil = should_release_gil_for_parse(total_bytes);
        let docs = load_yaml_documents_slices(py, &slices, multi, release_gil)?;
        return docs_to_python(py, docs, multi, handlers);
    }

    let (mapping_cls, _) = abc_types(py)?;
    if bound.is_instance(mapping_cls.bind(py))? {
        return Err(PyTypeError::new_err(
            "`text` must be a string or an iterable of strings",
        ));
    }

    if let Ok(iter) = PyIterator::from_object(bound.as_any()) {
        let mut lines: Vec<Py<PyString>> = Vec::new();
        for (idx, item) in iter.enumerate() {
            let item = item?;
            let s: Bound<'_, PyString> = item.cast_into().map_err(|err| {
                let item = err.into_inner();
                PyTypeError::new_err(format!(
                    "`text` iterable must yield only strings (index {idx} got {})",
                    unexpected_item_description(&item)
                ))
            })?;
            lines.push(s.unbind());
        }
        if lines.is_empty() {
            return if multi {
                Ok(PyList::empty(py).unbind().into())
            } else {
                Ok(py.None())
            };
        }

        let mut slices: Vec<&str> = Vec::with_capacity(lines.len());
        let mut total_bytes: usize = 0;
        for line in &lines {
            let slice = line.bind(py).to_str()?;
            total_bytes = total_bytes.saturating_add(slice.len());
            slices.push(slice);
        }
        let release_gil = should_release_gil_for_parse(total_bytes);
        let docs = load_yaml_documents_slices(py, &slices, multi, release_gil)?;
        return docs_to_python(py, docs, multi, handlers);
    }

    Err(PyTypeError::new_err(
        "`text` must be a string or an iterable of strings",
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
///     object: Parsed value(s): YAML mappings become `dict`, sequences become `list`, and scalars
///         resolve using the YAML 1.2 core schema to `bool`/`int`/`float`/`None`/`str`. Unhashable
///         mapping keys (for example `list`/`dict`) are wrapped in the lightweight `Yaml` dataclass
///         to keep them hashable. Tagged nodes without a matching handler are also wrapped in `Yaml`
///         so the tag can be preserved.
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
    path: Py<PyAny>,
    multi: bool,
    handlers: Option<Py<PyAny>>,
) -> Result<Py<PyAny>> {
    let handler_registry = handler_registry_from_arg(py, handlers.as_ref())?;
    let handlers = handler_registry.as_ref();

    let bound = path.bind(py);
    if let Ok(s) = bound.cast::<PyString>() {
        let path_str = s.to_str()?;
        let contents = py
            .detach(|| fs::read_to_string(path_str))
            .map_err(|err| PyIOError::new_err(format!("failed to read `{path_str}`: {err}")))?;
        let docs = load_yaml_documents(py, &contents, multi)?;
        return docs_to_python(py, docs, multi, handlers);
    }

    if let Some(path_buf) = pathlike_to_pathbuf(bound)? {
        let contents = py.detach(|| fs::read_to_string(&path_buf)).map_err(|err| {
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

        if let Ok(s) = read_result.cast::<PyString>() {
            let text = s.to_str()?;
            let docs = load_yaml_documents(py, text, multi)?;
            return docs_to_python(py, docs, multi, handlers);
        }

        if let Ok(bytes) = read_result.cast::<PyBytes>() {
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
///     'foo: 1'
///     >>> format_yaml(['first', 'second'], multi=True).endswith('...\n')
///     True
fn format_yaml(py: Python<'_>, value: Py<PyAny>, multi: bool) -> Result<Py<PyAny>> {
    let bound = value.bind(py);
    let yaml = py_to_yaml(py, bound, false)?;
    let mut output = format_yaml_impl(py, &yaml, multi)?;
    if multi {
        output.push_str("...\n");
        return Ok(PyString::new(py, output.as_str()).unbind().into_any());
    }
    let body = output.strip_prefix("---\n").unwrap_or(output.as_str());
    Ok(PyString::new(py, body).unbind().into_any())
}

#[pyfunction(signature = (value, path=None, multi=false))]
/// Write a Python value to YAML at `path` or stdout.
///
/// Args:
///     value (object): Python value or Yaml to serialize; for `multi` the value must be a sequence of documents.
///     path (str | os.PathLike | text file-like | None): Destination path or object with `.write(str)`; when None the YAML is written to stdout.
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
fn write_yaml(
    py: Python<'_>,
    value: Py<PyAny>,
    path: Option<Py<PyAny>>,
    multi: bool,
) -> Result<()> {
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

    if let Ok(path_str) = bound_path.cast::<PyString>() {
        let p = path_str.to_str()?;
        py.detach(|| fs::write(p, &output))
            .map_err(|err| PyIOError::new_err(format!("failed to write `{p}`: {err}")))?;
        return Ok(());
    }

    if let Some(path_buf) = pathlike_to_pathbuf(bound_path)? {
        py.detach(|| fs::write(&path_buf, &output)).map_err(|err| {
            PyIOError::new_err(format!("failed to write `{}`: {err}", path_buf.display()))
        })?;
        return Ok(());
    }

    if let Ok(write) = bound_path.getattr("write") {
        let writer = write.unbind();
        match write_all_text(py, &writer, output.as_str()) {
            Ok(()) => {}
            Err(WriteError::Call(err)) if err.is_instance_of::<PyTypeError>(py) => {
                let original = err.to_string();
                let augmented = PyTypeError::new_err(format!(
                    "writer must accept str; open the file in text mode (e.g. `open(path, 'w', encoding='utf-8')`) \
or wrap a binary stream with `io.TextIOWrapper(...)`. (writer.write(str) raised TypeError: {original})"
                ));
                augmented.set_cause(py, Some(err));
                return Err(augmented);
            }
            Err(err) => return Err(err.into_pyerr()),
        }
        return Ok(());
    }

    Err(PyTypeError::new_err(
        "`path` must be None, a string or path-like path, or an object with .write()",
    ))
}

#[pyfunction]
/// Debug helper: pretty-print parsed YAML nodes without converting to Python values.
fn _dbg_yaml(py: Python<'_>, text: Py<PyAny>) -> Result<()> {
    let bound = text.bind(py);
    if let Ok(s) = bound.cast::<PyString>() {
        let src = s.to_str()?;
        if src.is_empty() {
            return Ok(());
        }
        let docs = load_yaml_documents(py, src, true)?;
        println!("{docs:#?}");
        return Ok(());
    }

    if let Ok(seq) = bound.cast::<PySequence>() {
        let len = seq.len()?;
        if len == 0 {
            return Ok(());
        }
        let mut lines: Vec<Py<PyString>> = Vec::with_capacity(len);
        for idx in 0..len {
            let item = seq.get_item(idx)?;
            let s: Bound<'_, PyString> = item.cast_into().map_err(|err| {
                let item = err.into_inner();
                PyTypeError::new_err(format!(
                    "`text` sequence must contain only strings (index {idx} got {})",
                    unexpected_item_description(&item)
                ))
            })?;
            lines.push(s.unbind());
        }
        let mut slices: Vec<&str> = Vec::with_capacity(lines.len());
        let mut total_bytes: usize = 0;
        for line in &lines {
            let slice = line.bind(py).to_str()?;
            total_bytes = total_bytes.saturating_add(slice.len());
            slices.push(slice);
        }
        let release_gil = should_release_gil_for_parse(total_bytes);
        let docs = load_yaml_documents_slices(py, &slices, true, release_gil)?;
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

        if let Ok(s) = read_result.cast::<PyString>() {
            let text = s.to_str()?;
            let docs = load_yaml_documents(py, text, true)?;
            println!("{docs:#?}");
            return Ok(());
        }

        if let Ok(bytes) = read_result.cast::<PyBytes>() {
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
) -> Result<Py<PyAny>> {
    if multi {
        let mut values = Vec::with_capacity(docs.len());
        for doc in docs.iter_mut() {
            values.push(yaml_to_py(py, doc, false, handlers)?);
        }
        Ok(PyList::new(py, values)?.unbind().into_any())
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
        py.detach(load)
    } else {
        load()
    };
    result.map_err(|err| PyValueError::new_err(format!("YAML parse error: {err}")))?;
    Ok(loader.into_documents())
}

fn load_yaml_documents_slices<'py, 'input>(
    py: Python<'py>,
    slices: &'input [&'input str],
    multi: bool,
    release_gil: bool,
) -> Result<Vec<Yaml<'input>>> {
    let mut parser = Parser::new_from_iter(slices.iter().flat_map(|s| s.chars()));
    let mut loader = saphyr::YamlLoader::default();
    loader.early_parse(false);
    let mut load = || parser.load(&mut loader, multi);
    let result = if release_gil { py.detach(load) } else { load() };
    result.map_err(|err| PyValueError::new_err(format!("YAML parse error: {err}")))?;
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
) -> Result<Py<PyAny>> {
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

fn scalar_to_py(py: Python<'_>, scalar: &Scalar) -> Py<PyAny> {
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
) -> Result<Py<PyAny>> {
    let mut values = Vec::with_capacity(seq.len());
    for node in seq {
        values.push(yaml_to_py(py, node, false, handlers)?);
    }
    Ok(PyList::new(py, values)?.unbind().into_any())
}

fn mapping_to_py(
    py: Python<'_>,
    map: &mut Mapping,
    handlers: Option<&HandlerRegistry>,
) -> Result<Py<PyAny>> {
    let dict = PyDict::new(py);
    for (mut key, mut value) in map.drain() {
        let key_obj = yaml_to_py(py, &mut key, true, handlers)?;
        let value_obj = yaml_to_py(py, &mut value, false, handlers)?;
        dict.set_item(key_obj, value_obj)?;
    }
    Ok(dict.unbind().into_any())
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
) -> Result<Py<PyAny>> {
    let mut rendered_tag: Option<String> = None;
    let rendered = render_tag_cached(&mut rendered_tag, tag);
    let public_tag = normalize_simple_tag_name_for_api(rendered);

    if let Some(registry) = handlers {
        if let Some(handler) = registry.get_for_tag(public_tag.as_ref()) {
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
                make_yaml_node(py, value, Some(public_tag.as_ref()))
            }
            _ => Err(PyValueError::new_err(format!(
                "unsupported core-schema tag `{}`",
                public_tag.as_ref()
            ))),
        };
    }

    make_yaml_node(py, value, Some(public_tag.as_ref()))
}

fn make_yaml_node(py: Python<'_>, value: Py<PyAny>, tag: Option<&str>) -> Result<Py<PyAny>> {
    let cls = YAML_CLASS
        .get(py)
        .ok_or_else(|| PyValueError::new_err("Yaml class is not initialized"))?;
    if let Some(tag) = tag {
        Ok(cls.bind(py).call1((value, tag))?.unbind())
    } else {
        Ok(cls.bind(py).call1((value,))?.unbind())
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
            py.detach(|| emit_yaml_documents(docs, multi))
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

fn slice_from_char_offset(s: &str, chars: usize) -> &str {
    if chars == 0 {
        return s;
    }
    match s.char_indices().nth(chars) {
        Some((idx, _)) => &s[idx..],
        None => "",
    }
}

enum WriteError {
    Call(PyErr),
    Contract(PyErr),
}

impl WriteError {
    fn into_pyerr(self) -> PyErr {
        match self {
            WriteError::Call(err) | WriteError::Contract(err) => err,
        }
    }
}

enum WriteStep {
    Done,
    Partial(usize),
}

fn write_text_step(
    py: Python<'_>,
    writer: &Py<PyAny>,
    content: &str,
) -> std::result::Result<WriteStep, WriteError> {
    let result = writer.call1(py, (content,)).map_err(WriteError::Call)?;
    if result.is_none(py) {
        return Ok(WriteStep::Done);
    }
    let written: isize = result.extract(py).map_err(|_| {
        WriteError::Contract(PyTypeError::new_err(
            "writer.write() must return int or None to support partial writes",
        ))
    })?;
    if written <= 0 {
        return Err(WriteError::Contract(PyIOError::new_err(
            "writer.write() returned 0; output may be truncated",
        )));
    }
    Ok(WriteStep::Partial(written as usize))
}

fn write_all_text(
    py: Python<'_>,
    writer: &Py<PyAny>,
    content: &str,
) -> std::result::Result<(), WriteError> {
    let mut remaining = content;
    while !remaining.is_empty() {
        match write_text_step(py, writer, remaining)? {
            WriteStep::Done => return Ok(()),
            WriteStep::Partial(written) => {
                remaining = slice_from_char_offset(remaining, written);
            }
        }
    }
    Ok(())
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

    match stdout.getattr("write") {
        Ok(write) => {
            let writer = write.unbind();
            if write_all_text(py, &writer, content).is_err() {
                return write_to_stdout_fallback(py, content);
            }
        }
        Err(_) => {
            return write_to_stdout_fallback(py, content);
        }
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
    py.detach(|| {
        let mut stdout = io::stdout();
        stdout.write_all(content.as_bytes())?;
        stdout.flush()
    })
    .map_err(|err| PyIOError::new_err(format!("failed to write to stdout: {err}")))
}

fn float_to_yaml_scalar(f: f64) -> Yaml<'static> {
    if f.is_nan() {
        return Yaml::Representation(Cow::Borrowed(".nan"), ScalarStyle::Plain, None);
    }
    if f.is_infinite() {
        return if f.is_sign_negative() {
            Yaml::Representation(Cow::Borrowed("-.inf"), ScalarStyle::Plain, None)
        } else {
            Yaml::Representation(Cow::Borrowed(".inf"), ScalarStyle::Plain, None)
        };
    }
    if f.fract() != 0.0 {
        return Yaml::Value(Scalar::FloatingPoint(f.into()));
    }

    // Rust prints `1.0` as `1`, which YAML 1.2 resolves as an int; ensure the scalar parses as a float.
    let mut rendered = f.to_string();
    if !rendered.contains('.') && !rendered.contains('e') && !rendered.contains('E') {
        rendered.push_str(".0");
    }
    Yaml::Representation(Cow::Owned(rendered), ScalarStyle::Plain, None)
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
        let tag_str = tag_obj.cast::<PyString>()?.to_str()?;
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
        return Ok(float_to_yaml_scalar(f));
    }

    if ty.is(str_type.bind(py)) {
        let s: &Bound<'_, PyString> = obj.cast()?;
        return Ok(Yaml::Value(Scalar::String(Cow::Owned(
            s.to_str()?.to_owned(),
        ))));
    }

    if obj.is_instance_of::<PyDict>() {
        let dict: &Bound<'_, PyDict> = obj.cast()?;
        let mut mapping = Mapping::with_capacity(dict.len());
        for (key_obj, value_obj) in dict.iter() {
            let key_yaml = py_to_yaml(py, &key_obj, true)?;
            let value_yaml = py_to_yaml(py, &value_obj, false)?;
            mapping.insert(key_yaml, value_yaml);
        }
        return Ok(Yaml::Mapping(mapping));
    }

    if obj.is_instance_of::<PyList>() {
        let list: &Bound<'_, PyList> = obj.cast()?;
        let mut values = Vec::with_capacity(list.len());
        for item in list.iter() {
            values.push(py_to_yaml(py, &item, false)?);
        }
        return Ok(Yaml::Sequence(values));
    }

    if obj.is_instance_of::<PyTuple>() {
        let tuple: &Bound<'_, PyTuple> = obj.cast()?;
        let mut values = Vec::with_capacity(tuple.len());
        for item in tuple.iter() {
            values.push(py_to_yaml(py, &item, false)?);
        }
        return Ok(Yaml::Sequence(values));
    }

    if let Ok(seq) = obj.cast::<PySequence>() {
        if obj.cast::<PyString>().is_err() {
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
                .cast_into()
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
        return Ok(float_to_yaml_scalar(f));
    }

    if let Ok(s) = obj.cast::<PyString>() {
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
    if trimmed.chars().any(|c| c.is_whitespace()) {
        return Err(invalid_tag_error());
    }

    let tag = if trimmed == "!" {
        Tag {
            handle: "!".to_string(),
            suffix: String::new(),
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
    } else if let Some(rest) = trimmed.strip_prefix("!<") {
        let uri = rest.strip_suffix('>').ok_or_else(invalid_tag_error)?;
        if uri.is_empty() {
            return Err(invalid_tag_error());
        }
        Tag {
            handle: "!".to_string(),
            suffix: format!("<{uri}>"),
        }
    } else if let Some(rest) = trimmed.strip_prefix('!') {
        if rest.is_empty() {
            return Err(invalid_tag_error());
        }
        Tag {
            handle: "!".to_string(),
            suffix: rest.to_string(),
        }
    } else if let Some(core_suffix) = trimmed.strip_prefix("tag:yaml.org,2002:") {
        if core_suffix.is_empty() {
            return Err(invalid_tag_error());
        }
        Tag {
            handle: "!".to_string(),
            suffix: format!("!{core_suffix}"),
        }
    } else if is_simple_local_tag_name(trimmed) {
        Tag {
            handle: "!".to_string(),
            suffix: trimmed.to_string(),
        }
    } else {
        // Tag URI (from `!<...>` in YAML, or a `%TAG`-expanded URI). Emit in verbatim form because
        // we do not generate `%TAG` directives.
        Tag {
            handle: "!".to_string(),
            suffix: format!("<{trimmed}>"),
        }
    };
    Ok(tag)
}

#[pymodule]
pub fn yaml12(_py: Python<'_>, m: &Bound<'_, PyModule>) -> Result<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add_function(wrap_pyfunction!(parse_yaml, m)?)?;
    m.add_function(wrap_pyfunction!(read_yaml, m)?)?;
    m.add_function(wrap_pyfunction!(format_yaml, m)?)?;
    m.add_function(wrap_pyfunction!(write_yaml, m)?)?;
    m.add_function(wrap_pyfunction!(_normalize_tag, m)?)?;
    m.add_function(wrap_pyfunction!(_set_yaml_class, m)?)?;
    m.add_function(wrap_pyfunction!(_dbg_yaml, m)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use pyo3::types::{PyDict, PyList, PyModule};
    use saphyr::{LoadableYamlNode, Scalar, Tag};
    use std::ffi::CString;
    use std::sync::Mutex;

    #[derive(Debug, Copy, Clone, PartialEq, Eq)]
    enum ParsedValueKind {
        String,
        Boolean,
    }

    static PY_TEST_LOCK: Mutex<()> = Mutex::new(());

    fn load_scalar(input: &str) -> Yaml<'_> {
        let mut docs = Yaml::load_from_str(input).expect("parser should load tagged scalar");
        docs.pop().expect("expected one document")
    }

    fn init_test_module(py: Python<'_>) -> PyResult<Bound<'_, PyModule>> {
        let module = PyModule::new(py, "yaml12_test")?;
        super::yaml12(py, &module)?;
        let builtins = PyModule::import(py, "builtins")?;
        let yaml_cls = match builtins.getattr("_yaml12_test_yaml_cls") {
            Ok(cls) if !cls.is_none() => cls.unbind(),
            _ => {
                let helpers = CString::new(
                    r#"
from dataclasses import dataclass
from collections.abc import Mapping, Sequence

def _freeze(obj):
    if isinstance(obj, Yaml):
        return ("yaml", obj.tag, _freeze(obj.value))
    if isinstance(obj, Mapping):
        return ("map", tuple((_freeze(k), _freeze(v)) for k, v in obj.items()))
    if isinstance(obj, Sequence) and not isinstance(obj, (str, bytes, bytearray)):
        return ("seq", tuple(_freeze(x) for x in obj))
    try:
        hash(obj)
        return obj
    except TypeError:
        return ("unhashable", id(obj))

@dataclass(frozen=True)
class Yaml:
    value: object
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
                    "#,
                )
                .expect("helpers must not contain null bytes");

                let locals = PyDict::new(py);
                py.run(helpers.as_c_str(), Some(&locals), Some(&locals))?;
                let yaml_cls = locals
                    .get_item("Yaml")?
                    .expect("Yaml class should be defined")
                    .unbind();
                builtins.setattr("_yaml12_test_yaml_cls", yaml_cls.clone_ref(py))?;
                yaml_cls
            }
        };

        module.call_method1("_set_yaml_class", (yaml_cls.clone_ref(py),))?;
        module.setattr("Yaml", yaml_cls)?;
        Ok(module)
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

    #[test]
    fn parse_simple_mapping() -> PyResult<()> {
        let _guard = PY_TEST_LOCK.lock().expect("python test lock poisoned");
        Python::initialize();
        Python::attach(|py| {
            let module = init_test_module(py)?;
            let parse = module.getattr("parse_yaml")?;
            let obj = parse.call1(("foo: 1\nbar: true",))?;
            let mapping: Bound<'_, PyDict> = obj.cast_into()?;
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
        let _guard = PY_TEST_LOCK.lock().expect("python test lock poisoned");
        Python::initialize();
        Python::attach(|py| {
            let module = init_test_module(py)?;
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
            let list: Bound<'_, PyList> = parsed.cast_into()?;
            assert_eq!(list.len(), 2);
            assert_eq!(list.get_item(0)?.extract::<String>()?, "first");
            assert_eq!(list.get_item(1)?.extract::<String>()?, "second");
            Ok(())
        })
    }

    #[test]
    fn preserves_non_core_tags() -> PyResult<()> {
        let _guard = PY_TEST_LOCK.lock().expect("python test lock poisoned");
        Python::initialize();
        Python::attach(|py| {
            let module = init_test_module(py)?;
            let parse_yaml = module.getattr("parse_yaml")?;
            let format_yaml = module.getattr("format_yaml")?;
            let yaml_cls = module.getattr("Yaml")?;

            let parsed = parse_yaml.call1(("!foo 1",))?;
            assert!(parsed.is_instance(&yaml_cls)?);
            assert_eq!(parsed.getattr("tag")?.extract::<String>()?, "!foo");
            assert_eq!(parsed.getattr("value")?.extract::<String>()?, "1");

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
}
