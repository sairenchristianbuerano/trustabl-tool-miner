"""Reflects the existing rule pack's match: predicates into Python checks.

`load_existing_rules(repo_root)` returns the set of feature checks already
covered. `uncovered_features(tool)` returns features present on a tool
that no shipped rule catches.

Single source of truth: the YAMLs themselves. Anything not encoded here
falls back to the literal `match:` block for human comparison.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import yaml

from .scanner import ToolRecord

SDK_DIRS = {
    "openai_agents": "openai_sdk",
    "claude_agent_sdk": "claude_sdk",
    "google_adk": "google_adk",
}

MUTATING_PREFIXES = (
    "create_", "send_", "delete_", "post_", "update_",
    "refund_", "charge_", "issue_",
)

NETWORK_CALLEES = {
    "requests.get", "requests.post", "requests.put", "requests.delete",
    "requests.patch", "requests.head", "requests.request",
    "httpx.get", "httpx.post", "httpx.put", "httpx.delete",
    "httpx.patch", "httpx.head", "httpx.request",
    "urllib.request.urlopen",
}

AMBIGUOUS_NAMES = {
    "process", "handle", "run", "do", "execute", "perform",
    "work", "go", "thing", "stuff",
}


@dataclass(frozen=True)
class CoveredFeature:
    rule_id: str
    description: str


# Each predicate maps to: does this tool exhibit the feature?
# The right-hand-side is the existing rule that catches it.
FEATURE_CHECKS: dict[str, Callable[[ToolRecord], bool]] = {
    "missing_docstring": lambda t: not t.has_docstring,
    "missing_typed_params": lambda t: not t.typed_params,
    "ambiguous_name": lambda t: t.name in AMBIGUOUS_NAMES,
    "mutating_prefix_no_idempotency_kwarg": lambda t: (
        any(t.name.startswith(p) for p in MUTATING_PREFIXES)
        and not _has_idempotency_kwarg(t)
    ),
    "network_call": lambda t: any(
        c in NETWORK_CALLEES for c in t.body_call_targets
    ),
    "calls_subprocess": lambda t: any(
        c.startswith("subprocess.") for c in t.body_call_targets
    ),
    "calls_shell_true": lambda t: any(
        c in ("os.system", "os.popen") for c in t.body_call_targets
    ),
    "uses_pickle": lambda t: any(
        c.startswith("pickle.") for c in t.body_call_targets
    ),
    "writes_env_var": lambda t: any(
        c in ("os.environ.__setitem__", "os.putenv") for c in t.body_call_targets
    ),
    "bare_except": lambda t: t.has_bare_except,
    "mutable_default_arg": lambda t: t.has_mutable_default,
    "accepts_var_kwargs": lambda t: t.has_var_kwargs,
    "prints_to_stdout": lambda t: "print" in t.body_call_targets,
    "uses_eval_or_exec": lambda t: any(
        c in ("eval", "exec", "compile") for c in t.body_call_targets
    ),
}

# Map: feature name -> shipped rule id that catches it (None = uncovered).
COVERED_BY_RULE: dict[str, dict[str, str]] = {
    "openai_agents": {
        "missing_docstring": "OAI-001",
        "missing_typed_params": "OAI-002",
        "ambiguous_name": "OAI-007",
        "mutating_prefix_no_idempotency_kwarg": "OAI-009",
        "network_call": "OAI-005",
    },
    "claude_agent_sdk": {
        "missing_docstring": "CSDK-001",
        "missing_typed_params": "CSDK-002",
        "ambiguous_name": "CSDK-007",
        "mutating_prefix_no_idempotency_kwarg": "CSDK-006",
        "network_call": "CSDK-003",
    },
    "google_adk": {
        "missing_docstring": "ADK-001",
        "missing_typed_params": "ADK-002",
        "ambiguous_name": "ADK-007",
        "mutating_prefix_no_idempotency_kwarg": "ADK-006",
        "network_call": "ADK-003",
    },
}


def features_present(tool: ToolRecord) -> set[str]:
    return {name for name, check in FEATURE_CHECKS.items() if check(tool)}


def uncovered_features(tool: ToolRecord) -> set[str]:
    covered = set(COVERED_BY_RULE.get(tool.sdk, {}).keys())
    return features_present(tool) - covered


def load_existing_rule_ids(repo_root: Path) -> set[str]:
    """Read every <sdk>/<topic>.yaml and return the set of shipped rule IDs."""
    ids: set[str] = set()
    for sdk_dir in SDK_DIRS.values():
        sdk_path = repo_root / sdk_dir
        if not sdk_path.exists():
            continue
        for yml in sdk_path.glob("*.yaml"):
            try:
                doc = yaml.safe_load(yml.read_text(encoding="utf-8")) or {}
            except yaml.YAMLError:
                continue
            for rule in doc.get("rules", []) or []:
                if isinstance(rule, dict) and "id" in rule:
                    ids.add(rule["id"])
    return ids


def _has_idempotency_kwarg(tool: ToolRecord) -> bool:
    # We don't currently extract the tool's parameter names from scanner.
    # Conservative: assume absent if decorator doesn't mention one.
    # The agent re-reads the source snippet to confirm.
    for kw_value in tool.decorator_kwargs.values():
        lowered = kw_value.lower()
        if "idempot" in lowered or "request_id" in lowered or "txn_id" in lowered:
            return True
    return False
