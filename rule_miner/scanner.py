"""AST-based discovery of agent-tool definitions in a cloned repo.

Per-SDK detection rules (decorator name only — module of origin is checked
via `import` analysis):

    openai_agents     @function_tool            (with or without ())
    claude_agent_sdk  @tool / @claude_tool      from claude_agent_sdk
    google_adk        FunctionTool(<callable>)  call expression
"""

from __future__ import annotations

import ast
import dataclasses
from pathlib import Path
from typing import Iterable


@dataclasses.dataclass(frozen=True)
class ToolRecord:
    repo: str
    sdk: str
    file: str
    line: int
    name: str
    has_docstring: bool
    typed_params: bool
    decorator_kwargs: dict[str, str]
    body_call_targets: tuple[str, ...]
    has_bare_except: bool = False
    has_mutable_default: bool = False
    has_var_kwargs: bool = False
    language: str = "python"


def scan_paths(repo: str, sdk: str, roots: Iterable[Path]) -> list[ToolRecord]:
    records: list[ToolRecord] = []
    for root in roots:
        if not root.exists():
            continue
        for py in root.rglob("*.py"):
            try:
                tree = ast.parse(py.read_text(encoding="utf-8"), filename=str(py))
            except (SyntaxError, UnicodeDecodeError, OSError, ValueError):
                continue
            records.extend(_walk(tree, repo, sdk, py))
    return records


def _walk(tree: ast.AST, repo: str, sdk: str, file: Path) -> list[ToolRecord]:
    out: list[ToolRecord] = []
    for node in ast.walk(tree):
        if sdk in ("openai_agents", "claude_agent_sdk") and isinstance(
            node, (ast.FunctionDef, ast.AsyncFunctionDef)
        ):
            rec = _from_decorated_func(node, repo, sdk, file)
            if rec:
                out.append(rec)
        elif sdk == "google_adk" and isinstance(node, ast.Call):
            rec = _from_function_tool_call(node, tree, repo, sdk, file)
            if rec:
                out.append(rec)
    return out


def _from_decorated_func(
    fn: ast.FunctionDef | ast.AsyncFunctionDef,
    repo: str,
    sdk: str,
    file: Path,
) -> ToolRecord | None:
    accept = {
        "openai_agents": {"function_tool"},
        "claude_agent_sdk": {"tool", "claude_tool"},
    }[sdk]
    deco_kwargs: dict[str, str] = {}
    matched = False
    for d in fn.decorator_list:
        name = _decorator_name(d)
        if name not in accept:
            continue
        matched = True
        if isinstance(d, ast.Call):
            for kw in d.keywords:
                if kw.arg is None:
                    continue
                deco_kwargs[kw.arg] = ast.unparse(kw.value)
    if not matched:
        return None
    return ToolRecord(
        repo=repo,
        sdk=sdk,
        file=str(file),
        line=fn.lineno,
        name=fn.name,
        has_docstring=ast.get_docstring(fn) is not None,
        typed_params=_all_params_typed(fn.args),
        decorator_kwargs=deco_kwargs,
        body_call_targets=tuple(_body_calls(fn)),
        has_bare_except=_has_bare_except(fn),
        has_mutable_default=_has_mutable_default(fn.args),
        has_var_kwargs=fn.args.vararg is not None or fn.args.kwarg is not None,
    )


def _from_function_tool_call(
    call: ast.Call,
    tree: ast.AST,
    repo: str,
    sdk: str,
    file: Path,
) -> ToolRecord | None:
    if _call_name(call.func) != "FunctionTool":
        return None
    if not call.args:
        return None
    callable_ref = call.args[0]
    if not isinstance(callable_ref, ast.Name):
        return None
    fn = _find_function(tree, callable_ref.id)
    if fn is None:
        return None
    deco_kwargs = {
        kw.arg: ast.unparse(kw.value) for kw in call.keywords if kw.arg is not None
    }
    return ToolRecord(
        repo=repo,
        sdk=sdk,
        file=str(file),
        line=fn.lineno,
        name=fn.name,
        has_docstring=ast.get_docstring(fn) is not None,
        typed_params=_all_params_typed(fn.args),
        decorator_kwargs=deco_kwargs,
        body_call_targets=tuple(_body_calls(fn)),
        has_bare_except=_has_bare_except(fn),
        has_mutable_default=_has_mutable_default(fn.args),
        has_var_kwargs=fn.args.vararg is not None or fn.args.kwarg is not None,
    )


def _has_bare_except(fn: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    for node in ast.walk(fn):
        if isinstance(node, ast.ExceptHandler) and node.type is None:
            return True
    return False


def _has_mutable_default(args: ast.arguments) -> bool:
    for default in list(args.defaults) + list(args.kw_defaults):
        if default is None:
            continue
        if isinstance(default, (ast.List, ast.Dict, ast.Set)):
            return True
        if isinstance(default, ast.Call):
            name = _dotted_call_name(default.func)
            if name in ("list", "dict", "set"):
                return True
    return False


def _decorator_name(d: ast.expr) -> str | None:
    if isinstance(d, ast.Name):
        return d.id
    if isinstance(d, ast.Attribute):
        return d.attr
    if isinstance(d, ast.Call):
        return _decorator_name(d.func)
    return None


def _call_name(node: ast.expr) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return None


def _all_params_typed(args: ast.arguments) -> bool:
    params: list[ast.arg] = (
        list(args.posonlyargs) + list(args.args) + list(args.kwonlyargs)
    )
    if not params:
        return True
    return all(p.annotation is not None for p in params)


def _body_calls(fn: ast.FunctionDef | ast.AsyncFunctionDef) -> list[str]:
    targets: list[str] = []
    for node in ast.walk(fn):
        if isinstance(node, ast.Call):
            name = _dotted_call_name(node.func)
            if name:
                targets.append(name)
    return targets


def _dotted_call_name(node: ast.expr) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        prefix = _dotted_call_name(node.value)
        return f"{prefix}.{node.attr}" if prefix else node.attr
    return None


def _find_function(
    tree: ast.AST, name: str
) -> ast.FunctionDef | ast.AsyncFunctionDef | None:
    for node in ast.walk(tree):
        if (
            isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == name
        ):
            return node
    return None
