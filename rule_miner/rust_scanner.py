"""Regex-based Rust tool discovery (detection only).

Like `ts_scanner`, this is a pragmatic regex pass over `.rs` sources — there
is no stdlib Rust parser and we avoid native deps. It exists so the miner can
SCRAPE rust repos and surface candidates. NOTE: the engine has no `rust` rule
language yet, so `write_rule_yaml` refuses to draft rust rules
(SUPPORTED_RULE_LANGUAGES). Detection now, drafting once the engine adds rust.

Heuristics (low fidelity — the agent confirms via read_callsite):
  - #[tool] / #[tool(...)] attribute on a fn         -> tool fn
  - Tool::new("name", ...)                            -> builder form
  - preceding /// doc comment                         -> has_docstring
Risky body calls (reqwest / std::process::Command) are emitted as tokens so
the shared patterns.FEATURE_CHECKS fire on rust records too.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable

from .scanner import ToolRecord

RUST_EXTS = (".rs",)

_ATTR_FN = re.compile(
    r"#\[tool[^\]]*\]\s*(?:///.*\n\s*)*(?:pub\s+)?(?:async\s+)?fn\s+"
    r"(?P<name>\w+)",
    re.MULTILINE,
)
_TOOL_NEW = re.compile(r'Tool::new\s*\(\s*"(?P<name>[^"]+)"')

_CALL_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"\breqwest::(get|post|put|delete|patch)\b"), "reqwest.{0}"),
    (re.compile(r"\breqwest\b"), "reqwest"),
    (re.compile(r"\bureq::"), "reqwest"),
    (re.compile(r"\b(?:std::process::)?Command::new\s*\("), "Command.new"),
    (re.compile(r"\bprocess::Command\b"), "Command"),
]

_BODY_WINDOW = 1200


def scan_paths(repo: str, sdk: str, roots: Iterable[Path]) -> list[ToolRecord]:
    records: list[ToolRecord] = []
    seen: set[tuple[str, str]] = set()
    for root in roots:
        if not root.exists():
            continue
        for src in root.rglob("*.rs"):
            if "target" in src.parts:  # skip rust build dir
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


def _line_of(text: str, idx: int) -> int:
    return text.count("\n", 0, idx) + 1


def _has_doc_before(text: str, idx: int) -> bool:
    # Look at the line(s) immediately before the match for a /// doc comment.
    prefix = text[:idx].rstrip()
    last_line = prefix.rsplit("\n", 1)[-1].strip()
    return last_line.startswith("///")


def _body_calls(window: str) -> tuple[str, ...]:
    out: list[str] = []
    for pat, token in _CALL_PATTERNS:
        for m in pat.finditer(window):
            if "{0}" in token and m.groups():
                out.append(token.format(m.group(1)))
            else:
                out.append(token)
    return tuple(dict.fromkeys(out))


def _scan_text(repo: str, sdk: str, src: Path, text: str) -> list[ToolRecord]:
    out: list[ToolRecord] = []
    for m in _ATTR_FN.finditer(text):
        window = text[m.start():m.start() + _BODY_WINDOW]
        out.append(_record(repo, sdk, src, text, m.start(), m.group("name"),
                            _has_doc_before(text, m.start()), _body_calls(window)))
    for m in _TOOL_NEW.finditer(text):
        window = text[m.start():m.start() + _BODY_WINDOW]
        out.append(_record(repo, sdk, src, text, m.start(), m.group("name"),
                            False, _body_calls(window)))
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
        language="rust",
    )
