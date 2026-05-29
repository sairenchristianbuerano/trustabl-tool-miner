"""Regex-based TypeScript tool discovery.

The stdlib `scanner.py` parses Python with `ast`. There is no TS parser in
the stdlib and we deliberately avoid a node/tree-sitter dependency, so this
module does a pragmatic regex pass over `.ts`/`.tsx` sources to surface tool
definitions for the SDKs that ship TypeScript:

  - Claude Agent SDK:  tool("name", "description", schema, handler)
  - OpenAI Agents JS:  tool({ name: "...", description: "...", execute })

Fidelity is lower than the Python AST scanner — it cannot see real param
types or scope a call to a single tool body. It is good enough to surface
*candidates* (name, description presence, risky body calls); the agent step
re-reads the source via read_callsite to confirm before drafting.

Emits the same `ToolRecord` shape as `scanner.py`, with language set to
"typescript". `typed_params` is reported True (TS is typed; we do not try to
prove an untyped param and would rather not false-positive missing_typed_params).
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable

from .scanner import ToolRecord

TS_EXTS = (".ts", ".mts", ".cts", ".tsx")

# tool("name", "description", ...) — Claude Agent SDK factory form.
_FACTORY = re.compile(
    r"""\btool\s*\(\s*(['"`])(?P<name>[^'"`]+)\1\s*,\s*"""
    r"""(['"`])(?P<desc>[^'"`]*)\3""",
    re.DOTALL,
)
# tool({ ... }) — OpenAI Agents JS object-config form. Captures the object body.
_OBJECT = re.compile(r"\btool\s*\(\s*\{(?P<body>.*?)\}\s*\)", re.DOTALL)
_OBJ_NAME = re.compile(r"""\bname\s*:\s*(['"`])(?P<v>[^'"`]+)\1""")
_OBJ_DESC = re.compile(r"""\bdescription\s*:\s*(['"`])(?P<v>[^'"`]*)\1""")

# Risky body-call tokens, emitted into body_call_targets so the existing
# patterns.FEATURE_CHECKS fire on TS records too.
_CALL_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"\bfetch\s*\("), "fetch"),
    (re.compile(r"\baxios\.(get|post|put|delete|patch|request)\b"), "axios.{0}"),
    (re.compile(r"\baxios\s*\("), "axios"),
    (re.compile(r"\bchild_process\b"), "child_process.exec"),
    (re.compile(r"\bexecSync\s*\("), "execSync"),
    (re.compile(r"\bspawnSync?\s*\("), "spawn"),
    (re.compile(r"(?<![\w.])exec\s*\("), "exec"),
    (re.compile(r"(?<![\w.])eval\s*\("), "eval"),
]

_BODY_WINDOW = 1200  # chars after a match scanned for risky calls (heuristic)


def scan_paths(repo: str, sdk: str, roots: Iterable[Path]) -> list[ToolRecord]:
    records: list[ToolRecord] = []
    seen: set[tuple[str, str]] = set()
    for root in roots:
        if not root.exists():
            continue
        for ext in TS_EXTS:
            for src in root.rglob(f"*{ext}"):
                if _is_skippable(src):
                    continue
                try:
                    text = src.read_text(encoding="utf-8")
                except (OSError, UnicodeDecodeError, ValueError):
                    continue
                for rec in _scan_text(repo, sdk, src, text):
                    key = (rec.file, rec.name)
                    if key in seen:
                        continue
                    seen.add(key)
                    records.append(rec)
    return records


def _is_skippable(src: Path) -> bool:
    parts = set(src.parts)
    return bool(parts & {"node_modules", ".d.ts", "dist", "build"}) or \
        src.name.endswith(".d.ts")


def _line_of(text: str, idx: int) -> int:
    return text.count("\n", 0, idx) + 1


def _body_calls(window: str) -> tuple[str, ...]:
    out: list[str] = []
    for pat, token in _CALL_PATTERNS:
        for m in pat.finditer(window):
            if "{0}" in token and m.groups():
                out.append(token.format(m.group(1)))
            else:
                out.append(token)
    return tuple(dict.fromkeys(out))  # dedup, preserve order


def _scan_text(repo: str, sdk: str, src: Path, text: str) -> list[ToolRecord]:
    out: list[ToolRecord] = []

    for m in _FACTORY.finditer(text):
        name = m.group("name")
        desc = m.group("desc")
        window = text[m.start():m.start() + _BODY_WINDOW]
        out.append(_record(repo, sdk, src, text, m.start(), name, bool(desc),
                            _body_calls(window)))

    for m in _OBJECT.finditer(text):
        # The non-greedy {...} match stops at the first nested closing brace
        # (e.g. parameters: z.object({...})), so read name/description from the
        # match but scan a wider window for risky calls in execute/handler.
        window = text[m.start():m.start() + _BODY_WINDOW]
        nm = _OBJ_NAME.search(window)
        if not nm:
            continue
        dm = _OBJ_DESC.search(window)
        out.append(_record(repo, sdk, src, text, m.start(), nm.group("v"),
                            bool(dm and dm.group("v")), _body_calls(window)))

    return out


def _record(repo, sdk, src, text, idx, name, has_doc, calls) -> ToolRecord:
    return ToolRecord(
        repo=repo,
        sdk=sdk,
        file=str(src),
        line=_line_of(text, idx),
        name=name,
        has_docstring=has_doc,
        typed_params=True,
        decorator_kwargs={},
        body_call_targets=calls,
        language="typescript",
    )
