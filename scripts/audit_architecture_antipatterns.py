#!/usr/bin/env python3
"""Scan large modules for architecture antipattern refactor candidates."""

from __future__ import annotations

import argparse
import ast
import json
import re
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

DEFAULT_ROOTS = ("algobet", "scripts", "frontend")
DEFAULT_EXTENSIONS = (".py", ".ts", ".tsx", ".js", ".jsx")
EXCLUDES = {
    ".git",
    ".mypy_cache",
    ".next",
    ".pytest_cache",
    ".ruff_cache",
    ".venv",
    "__pycache__",
    "coverage",
    "dist",
    "node_modules",
    "playwright-report",
    "venv",
}


@dataclass
class Finding:
    category: str
    severity: str
    message: str
    line: int | None = None
    metric: str | None = None


@dataclass
class ModuleReport:
    path: str
    lines: int
    findings: list[Finding] = field(default_factory=list)

    @property
    def score(self) -> int:
        weights = {"low": 1, "medium": 2, "high": 4}
        return sum(weights.get(finding.severity, 1) for finding in self.findings)

    @property
    def categories(self) -> list[str]:
        return sorted({finding.category for finding in self.findings})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Scan modules above a line threshold for KISS, DRY, LSP-risk, "
            "god-module, and god-class refactor candidates."
        )
    )
    parser.add_argument("--root", action="append", dest="roots")
    parser.add_argument("--min-lines", type=int, default=500)
    parser.add_argument("--format", choices=("text", "json"), default="text")
    parser.add_argument(
        "--fail-on", choices=("none", "low", "medium", "high"), default="none"
    )
    parser.add_argument(
        "--max-score",
        type=int,
        default=None,
        help="Fail if any scanned module exceeds this total finding score.",
    )
    parser.add_argument(
        "--max-god-class-methods",
        type=int,
        default=None,
        help="Flag classes with more methods than this limit.",
    )
    parser.add_argument(
        "--max-function-lines",
        type=int,
        default=None,
        help="Flag functions or methods longer than this limit.",
    )
    parser.add_argument("--extensions", default=",".join(DEFAULT_EXTENSIONS))
    return parser.parse_args()


def severity_for(value: int | float, medium: int | float, high: int | float) -> str:
    if value >= high:
        return "high"
    if value >= medium:
        return "medium"
    return "low"


def severity_rank(severity: str) -> int:
    return {"none": 99, "low": 1, "medium": 2, "high": 3}[severity]


def is_excluded(path: Path) -> bool:
    return any(part in EXCLUDES for part in path.parts)


def read_lines(path: Path) -> list[str]:
    try:
        return path.read_text(encoding="utf-8").splitlines()
    except UnicodeDecodeError:
        return path.read_text(encoding="utf-8", errors="ignore").splitlines()


def iter_candidate_files(
    base_dir: Path,
    roots: list[str],
    extensions: tuple[str, ...],
) -> list[Path]:
    files: list[Path] = []
    for root_name in roots:
        root = (base_dir / root_name).resolve()
        if root.is_file():
            if root.suffix in extensions and not is_excluded(root):
                files.append(root)
            continue
        if not root.exists():
            continue
        files.extend(
            path
            for path in root.rglob("*")
            if path.is_file() and path.suffix in extensions and not is_excluded(path)
        )
    return sorted(set(files))


def normalized_line(line: str) -> str:
    # Order matters: collapse string literals before stripping comments so
    # that `//` inside quoted URLs (e.g. 'http://...') isn't mistaken for a
    # JS line comment.
    line = re.sub(r"(['\"])(?:\\.|(?!\1).)*\1", "S", line.strip())
    line = re.sub(r"#.*$|//.*$", "", line)
    line = re.sub(r"\b\d+(\.\d+)?\b", "N", line)
    return re.sub(r"\s+", " ", line)


def duplicate_block_findings(lines: list[str], window: int = 8) -> list[Finding]:
    normalized = [normalized_line(line) for line in lines]
    blocks: dict[tuple[str, ...], list[int]] = {}
    for index in range(max(0, len(normalized) - window + 1)):
        block = tuple(line for line in normalized[index : index + window] if line)
        if len(block) == window and sum(len(line) for line in block) >= 160:
            blocks.setdefault(block, []).append(index + 1)

    repeated = [values for values in blocks.values() if len(values) >= 2]
    if not repeated:
        return []

    repeated.sort(key=lambda values: (-len(values), values[0]))
    return [
        Finding(
            category="DRY",
            severity=severity_for(len(repeated), medium=3, high=8),
            line=repeated[0][0],
            metric=f"{len(repeated)} repeated {window}-line block(s)",
            message=(
                "Repeated normalized code blocks detected; extract shared helpers "
                "where intent is the same."
            ),
        )
    ]


def lineno(node: ast.AST) -> int:
    return int(getattr(node, "lineno", 0) or 0)


def end_lineno(node: ast.AST) -> int:
    return int(getattr(node, "end_lineno", getattr(node, "lineno", 0)) or 0)


def node_length(node: ast.AST) -> int:
    return max(0, end_lineno(node) - lineno(node) + 1)


def branch_count(node: ast.AST) -> int:
    branch_nodes = (
        ast.If,
        ast.For,
        ast.AsyncFor,
        ast.While,
        ast.Try,
        ast.ExceptHandler,
        ast.BoolOp,
        ast.IfExp,
        ast.Match,
        ast.comprehension,
    )
    return sum(1 for child in ast.walk(node) if isinstance(child, branch_nodes))


def max_nesting(node: ast.AST) -> int:
    nesting_nodes = (ast.If, ast.For, ast.AsyncFor, ast.While, ast.Try, ast.With)

    def visit(child: ast.AST, depth: int) -> int:
        next_depth = depth + 1 if isinstance(child, nesting_nodes) else depth
        child_depths = [
            visit(grandchild, next_depth) for grandchild in ast.iter_child_nodes(child)
        ]
        return max([next_depth, *child_depths])

    return visit(node, 0)


def required_arg_count(method: ast.FunctionDef | ast.AsyncFunctionDef) -> int:
    args = method.args
    positional = list(args.posonlyargs) + list(args.args)
    if positional and positional[0].arg in {"self", "cls"}:
        positional = positional[1:]
    return max(0, len(positional) - len(args.defaults)) + sum(
        1 for default in args.kw_defaults if default is None
    )


def raises_not_implemented(method: ast.AST) -> bool:
    for node in ast.walk(method):
        if not isinstance(node, ast.Raise):
            continue
        exc = node.exc.func if isinstance(node.exc, ast.Call) else node.exc
        if getattr(exc, "id", getattr(exc, "attr", "")) == "NotImplementedError":
            return True
    return False


def _module_findings(lines: list[str], tree: ast.Module) -> list[Finding]:
    findings: list[Finding] = []
    top_defs = [
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef)
    ]
    functions = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef)
    ]
    if len(lines) >= 1000 or len(top_defs) >= 18:
        findings.append(
            Finding(
                "God Module",
                severity_for(len(lines), medium=750, high=1200),
                "Large module with many top-level responsibilities; split by concern.",
                metric=f"{len(lines)} lines, {len(top_defs)} top-level defs",
            )
        )
    if len(functions) >= 40:
        findings.append(
            Finding(
                "God Module",
                severity_for(len(functions), medium=40, high=70),
                "High function count in one module; look for cohesive submodules.",
                metric=f"{len(functions)} functions/methods",
            )
        )
    return findings


def _class_findings(
    class_node: ast.ClassDef,
    method_limit: int,
    public_method_limit: int,
) -> list[Finding]:
    methods = [
        node
        for node in class_node.body
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef)
    ]
    public_methods = [m for m in methods if not m.name.startswith("_")]
    class_lines = node_length(class_node)
    class_metric = f"{class_node.name}: {class_lines} lines, {len(methods)} methods"
    findings: list[Finding] = []
    if (
        class_lines >= 300
        or len(methods) > method_limit
        or len(public_methods) > public_method_limit
    ):
        findings.append(
            Finding(
                "God Class",
                severity_for(class_lines, medium=300, high=600),
                "Class appears to own too many responsibilities.",
                line=lineno(class_node),
                metric=class_metric,
            )
        )
    for method in methods:
        if raises_not_implemented(method):
            findings.append(
                Finding(
                    "LSP",
                    "medium",
                    "Method raises NotImplementedError; verify substitutability.",
                    line=lineno(method),
                    metric=f"{class_node.name}.{method.name}",
                )
            )
    return findings


def _lsp_override_findings(
    class_methods: dict[str, dict[str, ast.FunctionDef | ast.AsyncFunctionDef]],
    class_bases: dict[str, list[str]],
) -> list[Finding]:
    findings: list[Finding] = []
    for class_name, bases in class_bases.items():
        for base in bases:
            base_lookup = class_methods.get(base)
            if not base_lookup:
                continue
            for method_name, method in class_methods[class_name].items():
                base_method = base_lookup.get(method_name)
                if base_method and required_arg_count(method) > required_arg_count(
                    base_method
                ):
                    findings.append(
                        Finding(
                            "LSP",
                            "high",
                            "Override narrows the accepted call contract.",
                            line=lineno(method),
                            metric=f"{class_name}.{method_name}",
                        )
                    )
    return findings


def _function_findings(
    func: ast.FunctionDef | ast.AsyncFunctionDef,
    function_line_limit: int,
) -> list[Finding]:
    findings: list[Finding] = []
    length = node_length(func)
    if length > function_line_limit:
        findings.append(
            Finding(
                "KISS",
                severity_for(
                    length,
                    medium=function_line_limit,
                    high=max(function_line_limit * 2, function_line_limit + 40),
                ),
                "Long function or method; extract smaller named steps.",
                line=lineno(func),
                metric=f"{func.name}: {length} lines",
            )
        )
    complexity = branch_count(func)
    if complexity >= 20:
        findings.append(
            Finding(
                "KISS",
                severity_for(complexity, medium=20, high=35),
                "High branching complexity; consider smaller policies.",
                line=lineno(func),
                metric=f"{func.name}: branch score {complexity}",
            )
        )
    nesting = max_nesting(func)
    if nesting >= 5:
        findings.append(
            Finding(
                "KISS",
                severity_for(nesting, medium=5, high=7),
                "Deep nesting makes behavior hard to reason about.",
                line=lineno(func),
                metric=f"{func.name}: nesting depth {nesting}",
            )
        )
    return findings


def _collect_class_index(
    classes: list[ast.ClassDef],
) -> tuple[
    dict[str, dict[str, ast.FunctionDef | ast.AsyncFunctionDef]],
    dict[str, list[str]],
]:
    class_methods: dict[str, dict[str, ast.FunctionDef | ast.AsyncFunctionDef]] = {}
    class_bases: dict[str, list[str]] = {}
    for class_node in classes:
        class_methods[class_node.name] = {
            node.name: node
            for node in class_node.body
            if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef)
        }
        class_bases[class_node.name] = [
            getattr(base, "id", getattr(base, "attr", "")) for base in class_node.bases
        ]
    return class_methods, class_bases


def analyze_python(
    lines: list[str],
    max_god_class_methods: int | None = None,
    max_function_lines: int | None = None,
) -> list[Finding]:
    tree = ast.parse("\n".join(lines))
    method_limit = max_god_class_methods or 15
    public_method_limit = min(method_limit, 10)
    function_line_limit = max_function_lines or 80

    classes = [node for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]
    functions = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef)
    ]
    class_methods, class_bases = _collect_class_index(classes)

    findings: list[Finding] = []
    findings.extend(_module_findings(lines, tree))
    for class_node in classes:
        findings.extend(_class_findings(class_node, method_limit, public_method_limit))
    findings.extend(_lsp_override_findings(class_methods, class_bases))
    for func in functions:
        findings.extend(_function_findings(func, function_line_limit))
    findings.extend(duplicate_block_findings(lines))
    return findings


def analyze_text(lines: list[str]) -> list[Finding]:
    findings: list[Finding] = []
    class_count = sum(
        1 for line in lines if re.search(r"^\s*(export\s+)?class\s+", line)
    )
    function_count = sum(
        1
        for line in lines
        if re.search(r"^\s*(export\s+)?(async\s+)?(function|const)\s+", line)
    )
    if len(lines) >= 1000 or class_count + function_count >= 25:
        findings.append(
            Finding(
                "God Module",
                severity_for(len(lines), medium=750, high=1200),
                "Large module with mixed responsibilities; split UI, state, and IO.",
                metric=(
                    f"{len(lines)} lines, {class_count} classes, "
                    f"{function_count} function-like blocks"
                ),
            )
        )

    branch_tokens = ("if ", "for ", "while ", "switch", "case ", "catch ", "&&", "||")
    branch_score = sum(line.count(token) for line in lines for token in branch_tokens)
    if branch_score >= 80:
        findings.append(
            Finding(
                "KISS",
                severity_for(branch_score, medium=80, high=140),
                "High branching token count; inspect for mixed workflows.",
                metric=f"branch token score {branch_score}",
            )
        )
    for line_number, line in enumerate(lines, start=1):
        if "extends " in line:
            window = "\n".join(lines[line_number - 1 : line_number + 120])
            if re.search(
                r"throw new Error\((['\"]).*not implemented.*\1\)", window, re.I
            ):
                findings.append(
                    Finding(
                        "LSP",
                        "medium",
                        "Subclass-like module contains not-implemented behavior.",
                        line=line_number,
                        metric="extends + not implemented",
                    )
                )
    findings.extend(duplicate_block_findings(lines))
    return findings


def analyze_file(
    path: Path,
    base_dir: Path,
    max_god_class_methods: int | None = None,
    max_function_lines: int | None = None,
    lines: list[str] | None = None,
) -> ModuleReport:
    if lines is None:
        lines = read_lines(path)
    report = ModuleReport(str(path.relative_to(base_dir)), len(lines))
    try:
        report.findings = (
            analyze_python(lines, max_god_class_methods, max_function_lines)
            if path.suffix == ".py"
            else analyze_text(lines)
        )
    except SyntaxError as exc:
        report.findings = [
            Finding("Parse", "low", f"Could not parse module: {exc.msg}", exc.lineno)
        ]
    return report


def render_text(reports: list[ModuleReport], scanned_count: int, min_lines: int) -> str:
    lines = [
        "# Architecture Antipattern Audit",
        "",
        f"Scanned modules over {min_lines} lines: {scanned_count}",
        f"Modules with findings: {sum(1 for report in reports if report.findings)}",
        "",
    ]
    ranked = sorted(
        reports,
        key=lambda report: (-report.score, -report.lines, report.path),
    )
    category_counts = Counter(
        finding.category for report in ranked for finding in report.findings
    )
    if category_counts:
        lines.append("## Category Summary")
        lines.extend(
            f"- {category}: {count}"
            for category, count in category_counts.most_common()
        )
        lines.append("")

    lines.append("## Modules")
    for report in ranked:
        if not report.findings:
            lines.append(f"- {report.path} ({report.lines} lines): no findings")
            continue
        lines.append(
            f"- {report.path} ({report.lines} lines, score {report.score}, "
            f"{', '.join(report.categories)})"
        )
        for finding in report.findings:
            location = f":{finding.line}" if finding.line else ""
            metric = f" [{finding.metric}]" if finding.metric else ""
            lines.append(
                f"  - {finding.severity.upper()} {finding.category}{location}{metric}: "
                f"{finding.message}"
            )
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    base_dir = Path.cwd()
    roots = args.roots or list(DEFAULT_ROOTS)
    extensions = tuple(
        ext if ext.startswith(".") else f".{ext}"
        for ext in args.extensions.split(",")
        if ext.strip()
    )
    candidates = iter_candidate_files(base_dir, roots, extensions)
    large_files: list[tuple[Path, list[str]]] = []
    for path in candidates:
        lines = read_lines(path)
        if len(lines) > args.min_lines:
            large_files.append((path, lines))
    reports = [
        analyze_file(
            path,
            base_dir,
            max_god_class_methods=args.max_god_class_methods,
            max_function_lines=args.max_function_lines,
            lines=lines,
        )
        for path, lines in large_files
    ]

    if args.format == "json":
        payload: dict[str, Any] = {
            "min_lines": args.min_lines,
            "scanned_modules": len(large_files),
            "reports": [
                {
                    **asdict(report),
                    "score": report.score,
                    "categories": report.categories,
                }
                for report in reports
            ],
        }
        print(json.dumps(payload, indent=2))
    else:
        print(render_text(reports, len(large_files), args.min_lines))

    failed = False
    if args.max_score is not None:
        failed = any(report.score > args.max_score for report in reports)
    if failed:
        return 1
    if args.fail_on == "none":
        return 0
    threshold = severity_rank(args.fail_on)
    return int(
        any(
            severity_rank(finding.severity) >= threshold
            for report in reports
            for finding in report.findings
        )
    )


if __name__ == "__main__":
    raise SystemExit(main())
