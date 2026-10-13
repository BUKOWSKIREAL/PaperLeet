import ast
import difflib
from pathlib import Path
import time

from paperleet.catalog import problems, root
from paperleet.judge.runner import make_report, run_batch, write_report


def changed_lines(reference: Path, mutant: Path):
    changes = list(difflib.ndiff(reference.read_text().splitlines(), mutant.read_text().splitlines()))
    return sum(line.startswith(("+ ", "- ")) for line in changes)


def validate_package(entry):
    directory = entry["directory"]
    issues = []
    for filename in ("README.md", "starter.py", "solution.py", "notes.md", "viz/gen.py"):
        if not (directory / filename).is_file():
            issues.append(f"{entry['id']}: missing {filename}")
    mutants = sorted((directory / "mutants").glob("*.py"))
    declared = {m["id"] for m in entry.get("mutants", [])}
    found = {path.stem for path in mutants}
    if declared != found:
        issues.append(f"{entry['id']}: declared mutants {sorted(declared)}, found {sorted(found)}")
    for path in mutants:
        lines = changed_lines(directory / "solution.py", path)
        if not 1 <= lines <= 10:
            issues.append(f"{entry['id']}/{path.name}: changed lines expected 1..10, got {lines}")
        if not path.read_text().startswith("#"):
            issues.append(f"{path}: first line must explain the bug")
        try:
            ast.parse(path.read_text())
        except SyntaxError as exc:
            issues.append(f"{path}: invalid mutant syntax: {exc}")
    for path in (directory / "tests").glob("test_*.py"):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module and node.module.endswith("solution"):
                issues.append(f"{path}: tests must access solutions through impl/ref fixtures")
            if isinstance(node, ast.Import) and any(alias.name.endswith("solution") for alias in node.names):
                issues.append(f"{path}: direct reference import bypasses the judge")
    return issues


def mutate(entry, report_path=None):
    directory = entry["directory"]
    plan = [{"path": str(directory / "solution.py"), "variant": "reference"}]
    plan.extend({"path": str(path), "variant": path.stem} for path in sorted((directory / "mutants").glob("*.py")))
    result = run_batch({entry["id"]: plan})
    refs = [test for test in result["tests"] if test["variant"] == "reference"]
    report = make_report(entry["id"], directory / "solution.py", refs)
    report["mutation"] = {variant["variant"]: [test["id"] for test in result["tests"] if test["variant"] == variant["variant"] and test["outcome"] == "failed"] for variant in plan[1:]}
    report["mutation_tests"] = [test for test in result["tests"] if test["variant"] != "reference"]
    report["quality_issues"] = validate_package(entry)
    report["passed"] = report["passed"] and all(report["mutation"].values()) and not report["quality_issues"] and result["exit_code"] in (0, 1) and not result.get("incomplete")
    if result.get("incomplete") or result["exit_code"] not in (0, 1):
        report["error"] = result["output"]
    if report_path:
        write_report(report, report_path)
    return report


def ci(repeats=3, identifiers=None, budget=300):
    started = time.perf_counter()
    entries = [entry for entry in problems() if not identifiers or entry["id"] in identifiers]
    if not entries:
        raise ValueError("No problem packages found")
    issues = [issue for entry in entries for issue in validate_package(entry)]
    plan = {}
    for entry in entries:
        directory = entry["directory"]
        variants = [("reference", directory / "solution.py"), *((path.stem, path) for path in sorted((directory / "mutants").glob("*.py")))]
        plan[entry["id"]] = [{"path": str(path), "variant": name, "repeat": repeat} for name, path in variants for repeat in range(repeats)]
    result = run_batch(plan, timeout=max(budget * 4, 300))
    if result.get("incomplete") or result["exit_code"] not in (0, 1):
        issues.append(result["output"])
    matrix = {}
    for entry in entries:
        rows = [row for row in result["tests"] if row["problem"] == entry["id"]]
        reference = [row for row in rows if row["variant"] == "reference" and row["repeat"] == 0]
        report = make_report(entry["id"], entry["directory"] / "solution.py", reference)
        for layer in ("T0", "T1", "T2", "T3"):
            actual_count = sum(row["layer"] == layer for row in reference)
            declared_count = entry.get("tests", {}).get(layer, 0)
            if actual_count != declared_count:
                issues.append(f"{entry['id']}/{layer}: metadata declares {declared_count} tests, collected {actual_count}")
        mutation = {}
        for variant in dict.fromkeys(p["variant"] for p in plan[entry["id"]]):
            runs = [[r for r in rows if r["variant"] == variant and r["repeat"] == repeat] for repeat in range(repeats)]
            signatures = [[(r["id"], r["passed"], r["outcome"]) for r in run] for run in runs]
            if not runs[0] or any(signature != signatures[0] for signature in signatures[1:]):
                issues.append(f"{entry['id']}/{variant}: missing tests or inconsistent results across {repeats} runs")
            if variant == "reference":
                for run in runs:
                    for row in run:
                        if not row["passed"]:
                            issues.append(f"{entry['id']}: reference failed {row['id']}: {row['message'][-1200:]}")
                        if row["layer"] == "T3" and row["duration_s"] > entry.get("t3_budget_s", 10):
                            issues.append(f"{entry['id']}/{row['id']}: T3 {row['duration_s']:.2f}s exceeds budget")
            else:
                mutation[variant] = [row["id"] for row in runs[0] if row["outcome"] == "failed"]
                if not mutation[variant]:
                    issues.append(f"{entry['id']}/{variant}: survived (test setup errors do not count as kills)")
        report["mutation"] = mutation
        write_report(report, root() / "reports" / (entry["id"].replace("/", "-") + "-reference.json"))
        matrix[entry["id"]] = mutation
    elapsed = time.perf_counter() - started
    if elapsed > budget:
        issues.append(f"CI wall time {elapsed:.2f}s exceeds {budget}s budget")
    summary = {"passed": not issues, "duration_s": elapsed, "repeats": repeats, "problems": len(entries), "mutation": matrix, "issues": issues, "tests_executed": len(result["tests"])}
    write_report(summary, root() / "reports" / "ci.json")
    return summary
