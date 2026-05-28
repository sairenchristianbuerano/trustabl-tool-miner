"""Plain-Python callables that the LLM agent invokes as tools.

Kept framework-agnostic so they're unit-testable. `agent.py` wraps them
with `@tool` from `claude_agent_sdk` at runtime.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
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


def write_rule_yaml(
    state: MiningState,
    draft: dict,
    topic: str,
    rationale_md: str | None = None,
    owasp_refs: list[str] | None = None,
    fix_type: str = "code",
    policy_meta: dict | None = None,
) -> str:
    """Validate `draft`, append it to <rules_repo>/<sdk_dir>/<topic>.yaml,
    AND write the paired rationale doc at
    <rules_repo>/docs/Policy/<sdk_dir>/<topic>.md per the trustabl-rules
    template (`docs/policy-rationale-doc-template-guide.md`).

    `rationale_md` is the agent-authored body (everything BELOW the
    metadata block: "What this policy covers", threat model,
    "Rule-by-rule defense" with the rule_id as its own H3, "What this
    policy does not cover", "Recommendations beyond the fix"). The tool
    composes the metadata block (Policy ID / File / Rules / Severities /
    Fix types / References) from draft fields + `owasp_refs` and prepends
    it. If the rationale doc already exists, the new rule's H3 block is
    appended under "Rule-by-rule defense" instead of overwriting.

    `owasp_refs` is a list of OWASP LLM Top 10:2025 IDs (e.g. ["LLM01",
    "LLM06"]) that anchor the rule in an external standard. Required by
    the template's metadata block.

    `fix_type` ∈ {"config", "code"} per the template — config fixes are
    prioritized in scan output because they carry lower breakage risk.

    Creates the YAML topic file if absent. In dry-run mode prints both
    files and does not write.
    """
    required = {
        "id", "title", "severity", "confidence", "applies_to",
        "match", "explanation", "fix", "sdk",
    }
    missing = required - set(draft.keys())
    if missing:
        return f"REJECTED: missing required fields: {sorted(missing)}"

    if draft["severity"] not in {"low", "medium", "high", "critical"}:
        return (
            "REJECTED: severity must be low|medium|high|critical "
            "(info is reserved per schema.yaml)"
        )
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

    scope = draft.get("scope", "tool")
    if scope not in {"tool", "agent", "subagent", "repo"}:
        return f"REJECTED: scope must be tool|agent|subagent|repo, got {scope!r}"

    if not topic or "/" in topic or "\\" in topic or topic.endswith(".yaml"):
        return "REJECTED: topic must be a bare filename stem (no slashes, no .yaml)"

    sdk_dir = state.repo_root / patterns.SDK_DIRS[sdk]
    path = sdk_dir / f"{topic}.yaml"

    persisted = _canonicalize_rule(draft, scope)

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
        # New topic file: schema.yaml requires top-level policy: wrapper
        # (id/name/category/description) plus rules:. Agent supplies via
        # policy_meta — reject if missing on a fresh file.
        meta_err = _validate_policy_meta(policy_meta, sdk)
        if meta_err:
            return meta_err
        doc = {
            "policy": _canonicalize_policy(policy_meta, sdk),
            "rules": [persisted],
        }

    rendered = yaml.safe_dump(doc, sort_keys=False, indent=2)

    rationale_body = (rationale_md or "").strip()
    refs = [r for r in (owasp_refs or []) if isinstance(r, str) and r.strip()]
    if fix_type not in ("config", "code"):
        return f"REJECTED: fix_type must be 'config' or 'code', got {fix_type!r}"

    sdk_dir_name = patterns.SDK_DIRS[sdk]
    rationale_path = (
        state.repo_root / "docs" / "Policy" / sdk_dir_name / f"{topic}.md"
    )

    warnings: list[str] = []
    if not rationale_body:
        warnings.append("no rationale_md")
    if not refs:
        warnings.append("no owasp_refs")
    warn_suffix = f" [WARN: {', '.join(warnings)}]" if warnings else ""

    if state.dry_run:
        print(f"\n=== DRY-RUN WRITE YAML: {path} ===")
        print(rendered)
        print(f"\n=== DRY-RUN WRITE RATIONALE: {rationale_path} ===")
        print(_compose_rationale_doc(
            yaml_relpath=f"{sdk_dir_name}/{topic}.yaml",
            rules=[(draft["id"], draft["severity"], fix_type)],
            owasp_refs=refs,
            body_md=rationale_body or "<rationale missing>",
            existing_text=None,
        ))
        state.written_rules.append((draft["id"], str(path)))
        return f"DRY_RUN: would have written {draft['id']} -> {path}{warn_suffix}"

    sdk_dir.mkdir(parents=True, exist_ok=True)
    path.write_text(rendered, encoding="utf-8")

    rationale_path.parent.mkdir(parents=True, exist_ok=True)
    existing = rationale_path.read_text(encoding="utf-8") if rationale_path.exists() else None
    composed = _compose_rationale_doc(
        yaml_relpath=f"{sdk_dir_name}/{topic}.yaml",
        rules=[(draft["id"], draft["severity"], fix_type)],
        owasp_refs=refs,
        body_md=rationale_body,
        existing_text=existing,
    )
    rationale_path.write_text(composed, encoding="utf-8")

    state.written_rules.append((draft["id"], str(path)))
    return (
        f"WROTE: {draft['id']} -> {path} + rationale {rationale_path}{warn_suffix}"
    )


_VALID_CATEGORIES = {"claude_sdk", "openai_sdk", "google_adk", "openshell", "mcp"}
_SDK_TO_CATEGORY = {
    "openai_agents": "openai_sdk",
    "claude_agent_sdk": "claude_sdk",
    "google_adk": "google_adk",
}


def _validate_policy_meta(meta: dict | None, sdk: str) -> str | None:
    if not isinstance(meta, dict):
        return (
            "REJECTED: new topic file requires policy_meta dict with id, "
            "name, category, description (per schema.yaml top-level policy:)"
        )
    required = {"id", "name", "category", "description"}
    missing = required - set(meta.keys())
    if missing:
        return f"REJECTED: policy_meta missing fields: {sorted(missing)}"
    pid = meta["id"]
    if not isinstance(pid, str) or not pid.islower() or " " in pid:
        return "REJECTED: policy_meta.id must be lowercase_underscored"
    cat = meta["category"]
    if cat not in _VALID_CATEGORIES:
        return (
            f"REJECTED: policy_meta.category must be one of "
            f"{sorted(_VALID_CATEGORIES)}, got {cat!r}"
        )
    expected = _SDK_TO_CATEGORY[sdk]
    if cat != expected:
        return (
            f"REJECTED: policy_meta.category {cat!r} does not match sdk "
            f"{sdk!r} (expected {expected!r})"
        )
    return None


def _canonicalize_policy(meta: dict, sdk: str) -> dict:
    return {
        "id": meta["id"],
        "name": meta["name"],
        "category": meta["category"],
        "description": meta["description"],
    }


def _canonicalize_rule(draft: dict, scope: str) -> dict:
    """Reorder + default-fill fields to match the canonical shipped shape
    (id, title, scope, severity, confidence, language, applies_to, match,
    explanation, fix). Strips `sdk` (miner-internal). Defaults `language`
    to `python` per CLAUDE.md.
    """
    out: dict = {}
    out["id"] = draft["id"]
    out["title"] = draft["title"]
    out["scope"] = scope
    out["severity"] = draft["severity"]
    out["confidence"] = float(draft["confidence"])
    out["language"] = draft.get("language", "python")
    out["applies_to"] = list(draft["applies_to"])
    out["match"] = draft["match"]
    out["explanation"] = draft["explanation"]
    out["fix"] = draft["fix"]
    return out


def _compose_rationale_doc(
    yaml_relpath: str,
    rules: list[tuple[str, str, str]],  # (rule_id, severity, fix_type)
    owasp_refs: list[str],
    body_md: str,
    existing_text: str | None,
) -> str:
    """Build (or extend) a policy rationale doc per the template guide."""
    if existing_text:
        # Topic doc exists — append this rule's H3 block under the existing
        # "Rule-by-rule defense" section instead of rewriting metadata.
        # The body_md from the agent should already start at the rule's H3.
        if not body_md.endswith("\n"):
            body_md += "\n"
        if not existing_text.endswith("\n"):
            existing_text += "\n"
        return existing_text + "\n" + body_md

    rule_ids = ", ".join(r[0] for r in rules)
    severities = ", ".join(r[1] for r in rules)
    fix_types = ", ".join(r[2] for r in rules)
    refs_str = ", ".join(owasp_refs) if owasp_refs else "(none — fill before merging)"
    metadata = (
        f"# Policy Rationale: {yaml_relpath.split('/')[-1].replace('.yaml','').replace('_',' ').title()}\n"
        f"\n"
        f"**Policy ID:** `{rules[0][0].split('-')[0]}-policy`  \n"
        f"**File:** `{yaml_relpath}`  \n"
        f"**Rules:** {rule_ids}  \n"
        f"**Severities:** {severities}  \n"
        f"**Fix types:** {fix_types}  \n"
        f"**References:** {refs_str}\n"
        f"\n"
        f"---\n\n"
    )
    return metadata + body_md.lstrip() + ("\n" if not body_md.endswith("\n") else "")
