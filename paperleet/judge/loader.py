"""Load implementations by path, never by a reference-solution import."""
import hashlib
import importlib.util
from pathlib import Path
import sys
from paperleet.catalog import root


def load_implementation(path: str | Path, source: bytes | None = None):
    path = Path(path).resolve()
    if source is None and not path.is_file():
        raise FileNotFoundError(f"Implementation not found: {path}")
    # Console entry points start from .venv/bin, not the repository directory.
    # Variant helpers and viz generators may import the problem namespace.
    repository = str(root())
    if repository not in sys.path:
        sys.path.insert(0, repository)
    name = "paperleet_impl_" + hashlib.sha256(str(path).encode()).hexdigest()[:16]
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load Python implementation: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    # Learners often submit equal-length edits within the same second. Python's
    # timestamp-based bytecode cache can otherwise silently execute an old edit.
    try:
        exec(compile(path.read_bytes() if source is None else source, str(path), "exec"), module.__dict__)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return module
