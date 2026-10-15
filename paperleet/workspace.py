"""Local files shared by the command-line tools and the web editor."""
import json
import hashlib
import os
from pathlib import Path
import tempfile

from paperleet.catalog import problem, problems, root, workspace_path
from paperleet.config import data_directory


MAX_CODE_BYTES = 128 * 1024


def local_problem(identifier):
    if not isinstance(identifier, str) or not identifier.strip():
        raise ValueError("A problem identifier is required")
    return problem(identifier)


def workspace_digest(entry):
    try:
        return hashlib.sha256(workspace_path(entry).read_bytes()).hexdigest()
    except FileNotFoundError:
        return None


def save_submission(entry, code):
    """Keep submitted code even when a newer IDE edit must stay in workspace."""
    digest = hashlib.sha256(code.encode("utf-8")).hexdigest()
    path = data_directory() / "solutions" / entry["id"] / f"{digest}.py"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(code, encoding="utf-8")
    return path


def validate_code(code, *, allow_empty=False):
    if not isinstance(code, str) or (not allow_empty and not code.strip()):
        raise ValueError("Python code cannot be empty")
    if len(code.encode("utf-8")) > MAX_CODE_BYTES:
        raise ValueError("Python code must be at most 128 KB")
    return code


def save_code(entry, code):
    """An atomic replacement keeps concurrent readers from seeing half a save."""
    validate_code(code, allow_empty=True)
    path = workspace_path(entry)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                         prefix=f".{path.name}.", delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(code)
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return path


def report_path(entry):
    return root() / "reports" / (entry["id"].replace("/", "-") + ".json")


def read_report(entry):
    path = report_path(entry)
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    if not isinstance(report, dict) or report.get("problem") != entry["id"]:
        return None
    return report


def read_workspace(entry):
    path = workspace_path(entry)
    code = path.read_text(encoding="utf-8") if path.is_file() else None
    return {"problem": entry["id"], "code": code, "report": read_report(entry)}


def read_reports():
    return {entry["id"]: report for entry in problems()
            if (report := read_report(entry)) is not None}
