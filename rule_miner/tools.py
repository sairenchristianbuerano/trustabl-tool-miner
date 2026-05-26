"""Plain-Python callables that the LLM agent invokes as tools.

Kept framework-agnostic so they're unit-testable. `agent.py` wraps them
with `@tool` from `claude_agent_sdk` at runtime.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from . import patterns
from .scanner import ToolRecord


SDK_TO_TOOL_TOKEN = {
    "openai_agents": "openai_tool",
    "claude_agent_sdk": "claude_sdk_tool",
    "google_adk": "adk_function_tool",
}


@dataclass
class CandidatePattern:
    sdk: str
    feature: str
    occurrence_count: int
    example_callsites: list[tuple[str, int, str]]  # (file, line, tool_name)


@dataclass
class MiningState:
    repo_root: Path
    dry_run: bool
    candidates: list[CandidatePattern] = field(default_factory=list)
    written_rules: list[tuple[str, str]] = field(default_factory=list)


def list_candidate_patterns(state: MiningState) -> str:
    """Returns JSON: every candidate pattern + its example callsites."""
    payload = [
        {
            "sdk": c.sdk,
            "feature": c.feature,
            "occurrences": c.occurrence_count,
            "examples": [
                {"file": f, "line": ln, "tool_name": n}
                for f, ln, n in c.example_callsites[:5]
            ],
        }
        for c in state.candidates
    ]
    return json.dumps(payload, indent=2)


def read_callsite(file: str, line: int, span: int = 30) -> str:
    """Returns the code snippet around `line` in `file`."""
    path = Path(file)
    if not path.exists():
        return f"# ERROR: file not found: {file}"
    text = path.read_text(encoding="utf-8", errors="replace").splitlines()
    start = max(0, line - 1)
    end = min(len(text), start + span)
    numbered = (f"{i + 1:>4}  {text[i]}" for i in range(start, end))
    return "\n".join(numbered)


def write_rule_yaml(state: MiningState, draft: dict, topic: str) -> str:
    """Validate `draft` and append it to <rules_repo>/<sdk_dir>/<topic>.yaml.

    Creates the topic file if absent. In dry-run mode prints the path +
    resulting YAML and does not write.
    """
    required = {
        "id", "title", "severity", "confidence", "applies_to",
        "match", "explanation", "fix", "sdk",
    }
    missing = required - set(draft.keys())
    if missing:
        return f"REJECTED: missing required fields: {sorted(missing)}"

    if draft["severity"] not in {"low", "medium", "high"}:
        return "REJECTED: severity must be low|medium|high (info is reserved)"
    if not isinstance(draft["confidence"], (int, float)):
        return "REJECTED: confidence must be a number 0..1"
    if not 0 <= float(draft["confidence"]) <= 1:
        return "REJECTED: confidence must be 0..1"

    sdk = draft["sdk"]
    if sdk not in patterns.SDK_DIRS:
        return (
            f"REJECTED: sdk must be one of {sorted(patterns.SDK_DIRS.keys())}, "
            f"got {sdk!r}"
        )

    expected_token = SDK_TO_TOOL_TOKEN[sdk]
    applies_to = draft.get("applies_to") or []
    if not isinstance(applies_to, list) or expected_token not in applies_to:
        return (
            f"REJECTED: applies_to must include {expected_token!r} for sdk "
            f"{sdk!r}"
        )

    existing = patterns.load_existing_rule_ids(state.repo_root)
    if draft["id"] in existing:
        return f"REJECTED: id {draft['id']} already exists in the rule pack"

    if not topic or "/" in topic or "\\" in topic or topic.endswith(".yaml"):
        return "REJECTED: topic must be a bare filename stem (no slashes, no .yaml)"

    sdk_dir = state.repo_root / patterns.SDK_DIRS[sdk]
    path = sdk_dir / f"{topic}.yaml"

    persisted = {k: v for k, v in draft.items() if k != "sdk"}

    if path.exists():
        try:
            doc = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        except yaml.YAMLError as exc:
            return f"REJECTED: existing {path} is not valid YAML: {exc}"
        rules = doc.get("rules") or []
        if not isinstance(rules, list):
            return f"REJECTED: existing {path} has non-list rules field"
        rules.append(persisted)
        doc["rules"] = rules
    else:
        doc = {"rules": [persisted]}

    rendered = yaml.safe_dump(doc, sort_keys=False, indent=2)

    if state.dry_run:
        print(f"\n=== DRY-RUN WRITE: {path} ===")
        print(rendered)
        state.written_rules.append((draft["id"], str(path)))
        return f"DRY_RUN: would have written {draft['id']} -> {path}"

    sdk_dir.mkdir(parents=True, exist_ok=True)
    path.write_text(rendered, encoding="utf-8")
    state.written_rules.append((draft["id"], str(path)))
    return f"WROTE: {draft['id']} -> {path}"
