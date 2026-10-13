"""Structural checks do not execute generated Python and are not a sandbox."""
import ast
import copy
import hashlib
import json
import re
from urllib.parse import urlparse

from .artifacts import TOY_KINDS

ALLOWED_IMPORTS = {"math", "typing", "numpy", "torch", "pytest", "collections", "itertools", "functools", "random", "statistics", "dataclasses"}
FORBIDDEN_CALLS = {"eval", "exec", "compile", "open", "input", "__import__", "getattr", "setattr", "delattr", "globals", "locals", "vars", "breakpoint", "exit", "quit"}
FORBIDDEN_ATTRIBUTES = {"save", "load", "loadtxt", "savetxt", "fromfile", "tofile", "memmap", "system", "popen", "settrace", "skip", "xfail", "skipif", "fixture", "parametrize", "main"}


def package_hash(candidate):
    return hashlib.sha256(json.dumps(candidate, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def https_url(value, required=False):
    if not value and not required:
        return ""
    if not isinstance(value, str) or len(value) > 2048:
        raise ValueError("Source URL must be a HTTPS URL under 2048 characters")
    parsed = urlparse(value)
    if parsed.scheme != "https" or not parsed.hostname or parsed.username or parsed.password or parsed.port not in (None, 443):
        raise ValueError("Source URL must be a public HTTPS URL without credentials")
    host = parsed.hostname.lower()
    if host in {"localhost", "127.0.0.1", "::1"} or "." not in host or host.endswith((".local", ".internal")):
        raise ValueError("Source URL must be public")
    import ipaddress
    try:
        if not ipaddress.ip_address(host).is_global:
            raise ValueError("Source URL must be public")
    except ValueError as exc:
        if str(exc) == "Source URL must be public":
            raise
    return value.strip()


def python_tree(source, filename):
    if not isinstance(source, str) or not source.strip() or len(source.encode()) > 64000:
        raise ValueError(f"{filename}: empty source or source larger than 64 KB")
    tree = ast.parse(source, filename)
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            modules = [node.module or ""] if isinstance(node, ast.ImportFrom) else [a.name for a in node.names]
            if (isinstance(node, ast.ImportFrom) and node.level) or any(m.split(".")[0] not in ALLOWED_IMPORTS for m in modules):
                raise ValueError(f"{filename}: unsupported import; generated packages use the CPU numerical allowlist")
        if isinstance(node, ast.Name) and node.id in FORBIDDEN_CALLS:
            raise ValueError(f"{filename}: forbidden reflective or filesystem API {node.id}")
        if isinstance(node, ast.Attribute) and (node.attr.startswith("__") or node.attr in FORBIDDEN_ATTRIBUTES):
            raise ValueError(f"{filename}: unsupported API {node.attr}")
        if isinstance(node, ast.Try):
            raise ValueError(f"{filename}: exception swallowing is unsupported in candidate code")
    return tree


def static_checks(candidate):
    checks = []
    def check(name, action):
        try:
            detail = action()
            checks.append({"name": name, "passed": True, "detail": detail or "Passed"})
        except (ValueError, TypeError, KeyError, SyntaxError, AttributeError) as exc:
            checks.append({"name": name, "passed": False, "detail": str(exc)})

    def content():
        if not isinstance(candidate, dict) or len(json.dumps(candidate).encode()) > 512000:
            raise ValueError("Candidate must be an object under 500 KB")
        for key in ("title", "title_en", "summary", "readme", "readme_en", "notes", "notes_en"):
            if not isinstance(candidate.get(key), str) or not candidate[key].strip():
                raise ValueError(f"Missing nonempty {key}")
        if candidate.get("difficulty") not in ("E", "M", "H"):
            raise ValueError("Difficulty must be E, M or H")
        evidence = candidate.get("evidence", [])
        if not isinstance(evidence, list) or not evidence:
            raise ValueError("At least one source and exact locator are required")
        for row in evidence:
            https_url(row["source_url"], required=True)
            if not row.get("claim") or not row.get("locator"):
                raise ValueError("Evidence must have a claim and equation/section/code locator")
        toy = candidate.get("toy")
        if toy is not None:
            if not isinstance(toy, dict) or toy.get("kind") not in TOY_KINDS:
                raise ValueError("Toy provenance must be author_toy, paper_derived or teaching_adaptation")
            for field in ("source_url", "locator", "adaptation"):
                if not isinstance(toy.get(field), str) or not toy[field].strip() or len(toy[field].encode()) > 4096:
                    raise ValueError(f"Toy provenance requires bounded nonempty {field}")
            https_url(toy["source_url"], required=True)
            if toy["source_url"] not in {row["source_url"] for row in evidence}:
                raise ValueError("Toy provenance must point to a cited evidence source")

    def interface():
        functions = candidate.get("functions", [])
        if not isinstance(functions, list) or not 1 <= len(functions) <= 6:
            raise ValueError("Define 1–6 required functions")
        names = [f["name"] for f in functions]
        if len(set(names)) != len(names) or any(not re.fullmatch(r"[a-z][a-z0-9_]*", name) for name in names):
            raise ValueError("Function names must be unique Python identifiers")
        signatures = {}
        for function in functions:
            signature = ast.parse("def " + function["signature"] + ":\n    pass\n").body[0]
            if not isinstance(signature, ast.FunctionDef) or signature.name != function["name"]:
                raise ValueError("Function signature must match its declared name")
            signatures[function["name"]] = signature.args
        def signature_shape(args):
            # Type aliases may differ; parameter names, modes and defaults may not.
            args = copy.deepcopy(args)
            for node in ast.walk(args):
                if isinstance(node, ast.arg):
                    node.annotation = None
            return ast.dump(args, include_attributes=False)
        for label in ("starter", "solution"):
            tree = python_tree(candidate[label], label + ".py")
            functions_by_name = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
            if not set(names) <= functions_by_name.keys():
                raise ValueError(f"{label}: must export all required functions")
            if any(signature_shape(functions_by_name[name].args) != signature_shape(signatures[name]) for name in names):
                raise ValueError(f"{label}: argument names, parameter kinds and defaults must match the declared contract")
            if label == "starter" and any(not any(isinstance(n, ast.Raise) and ((isinstance(n.exc, ast.Name) and n.exc.id == "NotImplementedError") or (isinstance(n.exc, ast.Call) and isinstance(n.exc.func, ast.Name) and n.exc.func.id == "NotImplementedError")) for n in ast.walk(functions_by_name[name])) for name in names):
                raise ValueError("Starter must raise NotImplementedError in required functions")

    def test_contract():
        tests = candidate.get("tests", [])
        if not isinstance(tests, list) or not 4 <= len(tests) <= 16:
            raise ValueError("Provide 4–16 tests covering all four layers")
        if {row.get("layer") for row in tests} != {"T0", "T1", "T2", "T3"}:
            raise ValueError("Tests must cover T0, T1, T2 and T3")
        names, toy_tests = set(), 0
        for row in tests:
            name = row["name"]
            if not re.fullmatch(r"test_[a-z0-9_]+", name) or name in names:
                raise ValueError("Test names must be unique test_* identifiers")
            names.add(name)
            tree = python_tree(row["code"], name + ".py")
            fns = [n for n in tree.body if isinstance(n, ast.FunctionDef)]
            if len(fns) != 1 or fns[0].name != name or fns[0].decorator_list:
                raise ValueError(f"{name}: one test function accepting impl and optional artifact, without decorators, is required")
            args = fns[0].args
            fixtures = [a.arg for a in args.args]
            if (fixtures not in (["impl"], ["impl", "artifact"]) or args.posonlyargs or args.kwonlyargs
                    or args.vararg or args.kwarg or args.defaults or args.kw_defaults):
                raise ValueError(f"{name}: only (impl) or (impl, artifact) fixtures are supported")
            if "artifact" in fixtures:
                if not candidate.get("toy"):
                    raise ValueError(f"{name}: artifact fixture requires declared toy provenance")
                if not any(isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "artifact" for n in ast.walk(fns[0])):
                    raise ValueError(f"{name}: requested artifact fixture must save measured output")
                toy_tests += row["layer"] == "T3"
            if not any(isinstance(n, ast.Assert) or (isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr.startswith("assert")) for n in ast.walk(fns[0])):
                raise ValueError(f"{name}: no assertion")
            if not any(isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and isinstance(n.func.value, ast.Name) and n.func.value.id == "impl" for n in ast.walk(fns[0])):
                raise ValueError(f"{name}: must invoke impl")
        if candidate.get("toy") and not toy_tests:
            raise ValueError("A declared toy requires a T3 test that records an artifact from impl")
        return f"{len(tests)} independently generated tests across T0–T3"

    def mutation_contract():
        mutants = candidate.get("mutants", [])
        if not isinstance(mutants, list) or not 2 <= len(mutants) <= 4:
            raise ValueError("Provide 2–4 plausible bug implementations")
        names = set()
        required = {f["name"] for f in candidate["functions"]}
        for row in mutants:
            if not re.fullmatch(r"[a-z][a-z0-9_]*", row["name"]) or row["name"] in names or not row.get("bug"):
                raise ValueError("Mutants need unique identifiers and bug explanations")
            names.add(row["name"])
            tree = python_tree(row["code"], row["name"] + ".py")
            if not required <= {n.name for n in tree.body if isinstance(n, ast.FunctionDef)}:
                raise ValueError("Every mutant must implement all required functions")
            if row["code"].strip() == candidate["solution"].strip():
                raise ValueError("Mutant cannot equal the reference")

    check("documented_contract_and_evidence", content)
    check("python_interface", interface)
    check("independent_test_contract", test_contract)
    check("mutation_contract", mutation_contract)
    return checks
