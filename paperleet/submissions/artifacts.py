"""Shared, numerical-library-free limits for untrusted experiment report data."""
import json
import math
import re

MAX_ARTIFACTS_PER_TEST = 4
MAX_ARTIFACT_BYTES_PER_TEST = 32 * 1024
MAX_ARTIFACT_BYTES_PER_REPORT = 256 * 1024
MAX_ARTIFACT_NODES = 4096
MAX_ARTIFACT_DEPTH = 12
TOY_KINDS = {"author_toy", "paper_derived", "teaching_adaptation"}


def artifact_name(name):
    if not isinstance(name, str) or not re.fullmatch(r"[a-z][a-z0-9_]{0,63}", name):
        raise ValueError("Artifact names must be lowercase identifiers of at most 64 characters")
    return name


def serialize_artifact(value, convert=None):
    """Return a finite JSON copy and its byte size; never truncate scientific data."""
    remaining, ancestors = MAX_ARTIFACT_NODES, set()

    def visit(item, depth):
        nonlocal remaining
        remaining -= 1
        if remaining < 0 or depth > MAX_ARTIFACT_DEPTH:
            raise ValueError("Artifact exceeds the node or nesting limit; summarize or downsample explicitly")
        if item is None or type(item) is bool:
            return item
        if type(item) is int:
            if abs(item) > 2**53 - 1:
                raise ValueError("Artifact integers must fit JavaScript's exact integer range")
            return item
        if type(item) is float:
            if not math.isfinite(item):
                raise ValueError("Artifact values must be finite")
            return item
        if type(item) is str:
            if len(item.encode("utf-8")) > 4096:
                raise ValueError("Artifact strings must be at most 4096 bytes")
            return item
        if type(item) not in (dict, list, tuple):
            if convert is None:
                raise ValueError("Artifact values must be JSON data or bounded numerical tensors")
            converted = convert(item)
            if converted is item:
                raise ValueError("Unsupported artifact value type")
            return visit(converted, depth + 1)
        if id(item) in ancestors:
            raise ValueError("Artifact values cannot contain cycles")
        if len(item) > MAX_ARTIFACT_NODES:
            raise ValueError("Artifact collection is too large")
        ancestors.add(id(item))
        try:
            if type(item) is dict:
                if any(type(key) is not str or len(key.encode("utf-8")) > 128 for key in item):
                    raise ValueError("Artifact object keys must be strings of at most 128 bytes")
                return {key: visit(child, depth + 1) for key, child in item.items()}
            return [visit(child, depth + 1) for child in item]
        finally:
            ancestors.remove(id(item))

    result = visit(value, 0)
    size = len(json.dumps(result, ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode("utf-8"))
    if size > MAX_ARTIFACT_BYTES_PER_TEST:
        raise ValueError("Artifact exceeds the 32 KB per-test budget")
    return result, size


def validate_report_artifacts(rows):
    """Recheck container JSON at the host boundary without importing PyTorch."""
    if not isinstance(rows, list) or len(rows) > 16 or any(not isinstance(row, dict) for row in rows):
        raise ValueError("A sandbox report must contain at most 16 test records")
    report_bytes = 0
    for row in rows:
        values = row.get("artifacts", {})
        if not isinstance(values, dict) or len(values) > MAX_ARTIFACTS_PER_TEST:
            raise ValueError("At most four named artifacts are allowed per test")
        test_bytes = 0
        for name, value in values.items():
            artifact_name(name)
            _, size = serialize_artifact(value)
            test_bytes += size
        if test_bytes > MAX_ARTIFACT_BYTES_PER_TEST:
            raise ValueError("Artifacts exceed the 32 KB per-test budget")
        report_bytes += test_bytes
        if report_bytes > MAX_ARTIFACT_BYTES_PER_REPORT:
            raise ValueError("Artifacts exceed the 256 KB report budget")
