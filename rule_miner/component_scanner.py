"""Pure-Python scanners for non-tool entities in a cloned repo.

The stdlib `scanner.py` finds Python *tool* definitions. This module adds
the entities a pre-deployment scan cares about, without depending on the
external `trustabl` Go binary:

  - sub-agents      .claude/agents/**/*.md frontmatter      -> SubagentRecord
  - repo components CLAUDE.md / settings / commands / ...    -> RepoComponents
  - skills          .claude/skills/**/SKILL.md frontmatter   -> SkillRecord
  - agents          Agent()/AgentDefinition()/LlmAgent() ... -> AgentRecord

`AgentRecord` and `SubagentRecord` are reused from `trustabl_scanner` so the
downstream aggregation treats records identically regardless of whether they
came from this scanner or the Go binary.

NOTE: the agent AST scanner reimplements constructor parsing the engine
already does. It is intentionally isolated here so it can be swapped for the
trustabl binary later without touching the candidate pipeline.
"""

from __future__ import annotations

import ast
import dataclasses
import json
from pathlib import Path

import yaml

from .scanner import _dotted_call_name  # reuse AST helper
from .trustabl_scanner import AgentRecord, SubagentRecord


@dataclasses.dataclass(frozen=True)
class SkillRecord:
    repo: str
    file: str
    name: str
    description: str
    allowed_tools: tuple[str, ...]


@dataclasses.dataclass(frozen=True)
class RepoComponents:
    """Presence of engine ComponentKind entities in one repo. Field names
    mirror models.ComponentKind so repo-scope features line up with the
    repo_component_present predicate's vocabulary."""
    repo: str
    claude_md: bool = False
    claude_settings: bool = False
    subagent: bool = False
    slash_command: bool = False
    hook_script: bool = False
    mcp_config: bool = False
    dependency_manifest: bool = False

    def present(self) -> set[str]:
        return {
            kind
            for kind in (
                "claude_md", "claude_settings", "subagent", "slash_command",
                "hook_script", "mcp_config", "dependency_manifest",
            )
            if getattr(self, kind)
        }


# Agent constructor class name -> miner sdk string. Mirrors
# trustabl_scanner._parse_agents' mapping.
AGENT_CLASS_TO_SDK = {
    "Agent": "openai_agents",
    "SandboxAgent": "openai_agents",
    "AgentDefinition": "claude_agent_sdk",
    "LlmAgent": "google_adk",
    "SequentialAgent": "google_adk",
    "ParallelAgent": "google_adk",
    "LoopAgent": "google_adk",
    "LanggraphAgent": "google_adk",
}
# `Agent` is ambiguous (google ADK aliases LlmAgent as Agent). Default the
# bare `Agent` to openai_agents; ADK code usually imports LlmAgent explicitly.

_TOOL_GRANT_KWARGS = ("tools",)


def _parse_frontmatter(path: Path) -> dict | None:
    """Return the YAML frontmatter dict of a markdown file, or None."""
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return None
    if not text.startswith("---"):
        return None
    parts = text.split("---", 2)
    if len(parts) < 3:
        return None
    try:
        meta = yaml.safe_load(parts[1])
    except yaml.YAMLError:
        return None
    return meta if isinstance(meta, dict) else None


def _as_tuple(value: object) -> tuple[str, ...]:
    """Normalize a frontmatter tools field (list OR comma string) to a tuple."""
    if isinstance(value, list):
        return tuple(str(v).strip() for v in value if str(v).strip())
    if isinstance(value, str):
        return tuple(p.strip() for p in value.split(",") if p.strip())
    return tuple()


def scan_subagents(repo: str, root: Path) -> list[SubagentRecord]:
    """Parse .claude/agents/**/*.md frontmatter into SubagentRecords."""
    out: list[SubagentRecord] = []
    agents_dir = root / ".claude" / "agents"
    if not agents_dir.exists():
        return out
    for md in agents_dir.rglob("*.md"):
        meta = _parse_frontmatter(md)
        if meta is None:
            continue
        out.append(
            SubagentRecord(
                repo=repo,
                file=str(md),
                name=str(meta.get("name") or md.stem),
                description=str(meta.get("description") or ""),
                tools=_as_tuple(meta.get("tools")),
                model=str(meta.get("model") or ""),
            )
        )
    return out


def scan_skills(repo: str, root: Path) -> list[SkillRecord]:
    """Parse .claude/skills/**/SKILL.md frontmatter into SkillRecords.

    Provisional: the engine has no skill scope yet (next trustabl release).
    """
    out: list[SkillRecord] = []
    skills_dir = root / ".claude" / "skills"
    if not skills_dir.exists():
        return out
    for md in skills_dir.rglob("SKILL.md"):
        meta = _parse_frontmatter(md)
        if meta is None:
            continue
        allowed = meta.get("allowed-tools")
        if allowed is None:
            allowed = meta.get("allowed_tools")
        if allowed is None:
            allowed = meta.get("tools")
        out.append(
            SkillRecord(
                repo=repo,
                file=str(md),
                name=str(meta.get("name") or md.parent.name),
                description=str(meta.get("description") or ""),
                allowed_tools=_as_tuple(allowed),
            )
        )
    return out


def scan_components(repo: str, root: Path) -> RepoComponents:
    """Detect presence of engine ComponentKind entities in `root`."""
    claude_dir = root / ".claude"
    settings_files = [
        claude_dir / "settings.json",
        claude_dir / "settings.local.json",
    ]
    settings_present = any(p.exists() for p in settings_files)
    hooks_present = False
    mcp_in_settings = False
    for p in settings_files:
        if not p.exists():
            continue
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(data, dict):
            if data.get("hooks"):
                hooks_present = True
            if data.get("mcpServers"):
                mcp_in_settings = True

    claude_md = (root / "CLAUDE.md").exists() or any(
        root.rglob("CLAUDE.md")
    )
    commands_dir = claude_dir / "commands"
    slash_command = commands_dir.exists() and any(commands_dir.rglob("*.md"))
    mcp_config = mcp_in_settings or (root / ".mcp.json").exists()
    dependency_manifest = any(
        (root / name).exists()
        for name in ("pyproject.toml", "requirements.txt", "package.json")
    )

    return RepoComponents(
        repo=repo,
        claude_md=bool(claude_md),
        claude_settings=settings_present,
        subagent=(claude_dir / "agents").exists()
        and any((claude_dir / "agents").rglob("*.md")),
        slash_command=bool(slash_command),
        hook_script=hooks_present,
        mcp_config=bool(mcp_config),
        dependency_manifest=dependency_manifest,
    )


def scan_agents(repo: str, roots: list[Path]) -> list[AgentRecord]:
    """Detect agent-constructor calls via Python AST under each root.

    Reimplements (a subset of) the engine's agent discovery. Captures the
    constructor class, kwargs (unparsed text), and granted tool names from a
    `tools=[...]` list literal.
    """
    out: list[AgentRecord] = []
    for root in roots:
        if not root.exists():
            continue
        for py in root.rglob("*.py"):
            try:
                tree = ast.parse(py.read_text(encoding="utf-8"), filename=str(py))
            except (SyntaxError, UnicodeDecodeError, OSError):
                continue
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call):
                    continue
                cls = _dotted_call_name(node.func)
                if not cls:
                    continue
                cls = cls.split(".")[-1]
                sdk = AGENT_CLASS_TO_SDK.get(cls)
                if sdk is None:
                    continue
                kwargs: dict[str, str] = {}
                tool_grants: tuple[str, ...] = tuple()
                name = ""
                for kw in node.keywords:
                    if kw.arg is None:
                        continue
                    kwargs[kw.arg] = ast.unparse(kw.value)
                    if kw.arg == "name" and isinstance(kw.value, ast.Constant):
                        name = str(kw.value.value)
                    if kw.arg in _TOOL_GRANT_KWARGS:
                        tool_grants = _grants_from_node(kw.value)
                out.append(
                    AgentRecord(
                        repo=repo,
                        sdk=sdk,
                        file=str(py),
                        line=node.lineno,
                        name=name or cls,
                        kwargs=kwargs,
                        tool_grants=tool_grants,
                    )
                )
    return out


def _grants_from_node(node: ast.expr) -> tuple[str, ...]:
    """Extract granted tool names from a tools=[...] kwarg value."""
    if not isinstance(node, (ast.List, ast.Tuple)):
        return tuple()
    names: list[str] = []
    for el in node.elts:
        if isinstance(el, ast.Constant) and isinstance(el.value, str):
            names.append(el.value)
        else:
            ref = _dotted_call_name(el if not isinstance(el, ast.Call) else el.func)
            if ref:
                names.append(ref.split(".")[-1])
    return tuple(names)
