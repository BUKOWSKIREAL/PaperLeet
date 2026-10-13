"""Filesystem problem registry. Adding a problem never requires editing a list."""
from pathlib import Path
import os
import yaml


def root() -> Path:
    return Path(os.environ.get("PAPERLEET_ROOT", Path(__file__).resolve().parents[1])).resolve()


def problems() -> list[dict]:
    result = []
    numbers = set()
    for path in sorted((root() / "problems").glob("*/*/problem.yaml")):
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        expected = path.parent.relative_to(root() / "problems").as_posix()
        if data.get("id") != expected:
            raise ValueError(f"{path}: id must equal {expected!r}")
        number = data.get("number")
        if type(number) is not int or number < 1:
            raise ValueError(f"{path}: number must be a positive integer")
        if number in numbers:
            raise ValueError(f"{path}: duplicate problem number {number}")
        numbers.add(number)
        data["directory"] = path.parent
        result.append(data)
    return sorted(result, key=lambda p: p["number"])


def problem(identifier: str) -> dict:
    for entry in problems():
        if identifier in (entry["id"], entry.get("short_id"), str(entry["number"])):
            return entry
    raise ValueError(f"Unknown problem {identifier!r}. Run paperleet list.")


def workspace_path(entry: dict) -> Path:
    return root() / "workspace" / (entry["id"] + ".py")
