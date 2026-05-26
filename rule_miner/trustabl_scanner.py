"""Optional integration with the `trustabl` static analyzer.

`trustabl scan <path> --format json` emits a structured `ScanResult` with
deeper, multi-language discovery than this miner's stdlib AST scanner:
- Python tool / agent / subagent / MCP server inventory.
- TypeScript Claude SDK tools and agents.
- Subagent markdown frontmatter, Claude settings, hosted tools.

We use it as an *additional* scanner — when the binary is available we
fold its tool inventory into the rule-miner record stream, deduplicated by
(file, line, name) against the stdlib AST scanner so existing tests still
pin shape. Agent and subagent inventories pass through unchanged for
future agent-scope predicate work.

Trustabl must be built separately (Go + CGO, tree-sitter). See
https://github.com/trustabl/trustabl#from-source. If the binary is not on
PATH, callers should fall back to `rule_miner.scanner` alone.
"""

from __future__ import annotations

import dataclasses
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

from .scanner import ToolRecord

# trustabl ToolKind -> rule-miner SDK string. SDK enum strings happen to
# match rule-miner's own SDK identifiers exactly; the mapping below stays
# explicit so a trustabl rename doesn't silently break us.
_KIND_TO_SDK = {
    "claude_sdk_tool": "claude_agent_sdk",
    "openai_tool": "openai_agents",
    "adk_function_tool": "google_adk",
}


@dataclasses.dataclass(frozen=True)
class AgentRecord:
    repo: str
    sdk: str
    file: str
    line: int
    name: str
    kwargs: dict[str, str]
    tool_grants: tuple[str, ...]


@dataclasses.dataclass(frozen=True)
class SubagentRecord:
    repo: str
    file: str
    name: str
    description: str
    tools: tuple[str, ...]
    model: str


@dataclasses.dataclass
class TrustablResult:
    tools: list[ToolRecord]
    agents: list[AgentRecord]
    subagents: list[SubagentRecord]


def available() -> bool:
    return shutil.which("trustabl") is not None


def scan(repo: str, repo_root: Path, timeout: float = 300.0) -> TrustablResult:
    """Invoke `trustabl scan <repo_root> --format json` and parse the output."""
    proc = subprocess.run(
        ["trustabl", "scan", str(repo_root), "--format", "json"],
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )
    if proc.returncode not in (0, 1):
        # 2 = scanner error per trustabl docs. 1 = findings present (OK).
        raise RuntimeError(
            f"trustabl exited {proc.returncode} on {repo_root}: {proc.stderr[:500]}"
        )
    return parse_scan_result(repo, repo_root, json.loads(proc.stdout))


def parse_scan_result(
    repo: str, repo_root: Path, scan_result: dict[str, Any]
) -> TrustablResult:
    return TrustablResult(
        tools=_parse_tools(repo, repo_root, scan_result.get("tools") or []),
        agents=_parse_agents(repo, repo_root, scan_result.get("agents") or []),
        subagents=_parse_subagents(
            repo, repo_root, scan_result.get("subagents") or []
        ),
    )


def merge_tools(
    primary: list[ToolRecord], extra: list[ToolRecord]
) -> list[ToolRecord]:
    """Append `extra` records that don't collide with `primary` by (file, line, name)."""
    seen = {(r.file, r.line, r.name) for r in primary}
    out = list(primary)
    for r in extra:
        key = (r.file, r.line, r.name)
        if key in seen:
            continue
        seen.add(key)
        out.append(r)
    return out


def _parse_tools(
    repo: str, repo_root: Path, raw: list[dict[str, Any]]
) -> list[ToolRecord]:
    records: list[ToolRecord] = []
    for t in raw:
        kind = t.get("kind", "")
        sdk = _KIND_TO_SDK.get(kind)
        if sdk is None:
            continue  # mcp_tool / shell_invocation / unknown -- not yet mined
        file_path = _abs_path(repo_root, t.get("file_path", ""))
        records.append(
            ToolRecord(
                repo=repo,
                sdk=sdk,
                file=file_path,
                line=int(t.get("line") or 0),
                name=t.get("name", ""),
                has_docstring=bool((t.get("description") or "").strip()),
                typed_params=bool(t.get("has_typed_params", False)),
                decorator_kwargs={
                    k: str(v) for k, v in (t.get("config") or {}).items()
                },
                # trustabl Python detector doesn't populate body_call_targets;
                # leave empty and let the stdlib scanner cover that signal.
                body_call_targets=tuple(),
            )
        )
    return records


def _parse_agents(
    repo: str, repo_root: Path, raw: list[dict[str, Any]]
) -> list[AgentRecord]:
    out: list[AgentRecord] = []
    sdk_class_to_sdk = {
        "openai_agent": "openai_agents",
        "openai_sandbox_agent": "openai_agents",
        "claude_agent_definition": "claude_agent_sdk",
        "adk_llm_agent": "google_adk",
        "adk_sequential_agent": "google_adk",
        "adk_parallel_agent": "google_adk",
        "adk_loop_agent": "google_adk",
        "adk_langgraph_agent": "google_adk",
    }
    for a in raw:
        sdk = sdk_class_to_sdk.get(a.get("class") or "", "")
        if not sdk:
            continue
        kwargs = {k: str(v) for k, v in (a.get("kwargs") or {}).items()}
        tool_grants = tuple(a.get("tool_names") or a.get("tools") or [])
        out.append(
            AgentRecord(
                repo=repo,
                sdk=sdk,
                file=_abs_path(repo_root, a.get("file_path", "")),
                line=int(a.get("line") or 0),
                name=a.get("name") or "",
                kwargs=kwargs,
                tool_grants=tool_grants,
            )
        )
    return out


def _parse_subagents(
    repo: str, repo_root: Path, raw: list[dict[str, Any]]
) -> list[SubagentRecord]:
    out: list[SubagentRecord] = []
    for s in raw:
        out.append(
            SubagentRecord(
                repo=repo,
                file=_abs_path(repo_root, s.get("file_path", "")),
                name=s.get("name") or "",
                description=s.get("description") or "",
                tools=tuple(s.get("tools") or []),
                model=s.get("model") or "",
            )
        )
    return out


def _abs_path(repo_root: Path, file_path: str) -> str:
    if not file_path:
        return ""
    p = Path(file_path)
    return str(p if p.is_absolute() else repo_root / p)
