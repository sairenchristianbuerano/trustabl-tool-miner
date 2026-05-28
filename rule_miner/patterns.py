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

from .component_scanner import RepoComponents, SkillRecord
from .scanner import ToolRecord
from .trustabl_scanner import AgentRecord, SubagentRecord

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

# ── Non-tool scope feature checks ───────────────────────────────────────
# Feature names here are namespaced "<scope>:<feature>" in coverage/candidate
# bookkeeping so they never collide with tool features (e.g. an agent's
# grants_bash is distinct from a subagent's grants_bash).

def _agent_missing_guardrails(a: AgentRecord) -> bool:
    # OpenAI Agents SDK is the only one with first-class input/output
    # guardrail kwargs. A non-empty value means guarded.
    if a.sdk != "openai_agents":
        return False
    for k in ("input_guardrails", "output_guardrails"):
        v = a.kwargs.get(k)
        if v and v not in ("[]", "None", "()", "{}"):
            return False
    return True


AGENT_FEATURE_CHECKS: dict[str, Callable[[AgentRecord], bool]] = {
    "missing_guardrails": _agent_missing_guardrails,
    "grants_bash": lambda a: "Bash" in a.tool_grants,
    "grants_websearch": lambda a: "WebSearch" in a.tool_grants,
}

SUBAGENT_FEATURE_CHECKS: dict[str, Callable[[SubagentRecord], bool]] = {
    "grants_bash": lambda s: "Bash" in s.tools,
    "grants_write": lambda s: "Write" in s.tools,
    "grants_bash_and_write": lambda s: "Bash" in s.tools and "Write" in s.tools,
    "grants_webfetch": lambda s: bool({"WebFetch", "WebSearch"} & set(s.tools)),
}

SKILL_FEATURE_CHECKS: dict[str, Callable[[SkillRecord], bool]] = {
    "skill_no_description": lambda s: not s.description.strip(),
    "skill_grants_broad_tools": lambda s: not s.allowed_tools,
}


def repo_features(
    components: RepoComponents, sdks_in_repo: set[str], has_shell: bool
) -> set[str]:
    """Per-repo gaps. Returns bare feature names (caller namespaces)."""
    feats: set[str] = set()
    if sdks_in_repo and not components.claude_md:
        feats.add("uses_sdk_no_claude_md")
    if components.subagent and not components.claude_settings:
        feats.add("subagents_no_settings")
    if has_shell and not components.claude_settings:
        feats.add("shell_tools_no_settings")
    return feats


# Body-text fragments that, when present in a rule's match.has_body_text,
# indicate the rule already covers that feature. Compared against shipped
# YAMLs at runtime in derive_covered_features() — replaces the old
# hardcoded COVERED_BY_RULE map which went stale every time a new rule
# shipped.
BODY_TEXT_SIGNATURES: dict[str, tuple[str, ...]] = {
    "calls_subprocess": ("subprocess.",),
    "calls_shell_true": ("os.system", "os.popen", "shell=True"),
    "uses_pickle": ("pickle.",),
    "writes_env_var": ("os.environ[", "os.putenv"),
    "prints_to_stdout": ("print(",),
    "uses_eval_or_exec": ("eval(", "exec("),
    "bare_except": ("except:",),
    "network_call": (
        "requests.", "httpx.", "urllib.request.urlopen", "aiohttp.",
    ),
}

# Match-key signatures: rule fields that signal coverage directly.
MATCH_KEY_SIGNATURES: dict[str, tuple[str, ...]] = {
    "missing_docstring": ("requires_docstring", "missing_docstring",
                          "has_docstring"),
    "missing_typed_params": ("requires_typed_params", "untyped_params",
                             "has_typed_params"),
    "ambiguous_name": ("ambiguous_name", "name_in_list", "tool_name_matches"),
    "mutating_prefix_no_idempotency_kwarg": (
        "missing_idempotency", "mutating_prefix_no_idempotency",
        "requires_idempotency",
    ),
    "has_var_kwargs": ("accepts_var_kwargs", "has_kwargs", "has_var_kwargs"),
    "mutable_default_arg": ("has_mutable_default", "mutable_default"),
}

# param_name_matches values that signal a feature is already covered. A rule
# that inspects a tool's parameter names for these tokens is enforcing that
# feature (e.g. an idempotency rule checks params for `idempot`/`request_id`).
# Compared against the flattened contains/exact values of every
# param_name_matches predicate found anywhere in the match tree.
PARAM_NAME_SIGNATURES: dict[str, tuple[str, ...]] = {
    "mutating_prefix_no_idempotency_kwarg": (
        "idempot", "request_id", "txn_id",
    ),
}

SDK_TO_APPLIES_TO = {
    "openai_agents": "openai_tool",
    "claude_agent_sdk": "claude_sdk_tool",
    "google_adk": "adk_function_tool",
}


def features_present(tool: ToolRecord) -> set[str]:
    return {name for name, check in FEATURE_CHECKS.items() if check(tool)}


def agent_features_present(agent: AgentRecord) -> set[str]:
    return {n for n, chk in AGENT_FEATURE_CHECKS.items() if chk(agent)}


def subagent_features_present(sub: SubagentRecord) -> set[str]:
    return {n for n, chk in SUBAGENT_FEATURE_CHECKS.items() if chk(sub)}


def skill_features_present(skill: SkillRecord) -> set[str]:
    return {n for n, chk in SKILL_FEATURE_CHECKS.items() if chk(skill)}


def uncovered_scoped(
    scope: str, sdk: str, present: set[str], covered: dict[str, set[str]] | None
) -> set[str]:
    """Bare feature names in `present` not covered for (sdk, scope). Coverage
    is stored namespaced "<scope>:<feature>" for non-tool scopes."""
    sdk_covered = (covered or {}).get(sdk, set())
    return {f for f in present if f"{scope}:{f}" not in sdk_covered}


def uncovered_features(
    tool: ToolRecord,
    covered: dict[str, set[str]] | None = None,
) -> set[str]:
    """Features present on `tool` that no shipped rule catches.

    `covered` is the precomputed result of `derive_covered_features()`
    keyed by SDK. If None, treats every feature as uncovered (the caller
    should always pass derived coverage in real use).
    """
    sdk_covered = (covered or {}).get(tool.sdk, set())
    return features_present(tool) - sdk_covered


def derive_covered_features(repo_root: Path) -> dict[str, set[str]]:
    """Walk shipped YAMLs and infer which features are already covered.

    Returns {sdk_name: {feature_name, ...}}. Match logic:
      - rule.match.has_body_text fragments → BODY_TEXT_SIGNATURES lookup
      - rule.match keys → MATCH_KEY_SIGNATURES lookup
      - rule.applies_to entries → SDK assignment

    Replaces the old hardcoded COVERED_BY_RULE map. New rules ship → next
    scan auto-skips the matching feature; no patterns.py edit required.
    """
    covered: dict[str, set[str]] = {sdk: set() for sdk in SDK_DIRS}
    for sdk, sdk_dir in SDK_DIRS.items():
        sdk_path = repo_root / sdk_dir
        if not sdk_path.exists():
            continue
        for yml in sdk_path.glob("*.yaml"):
            try:
                doc = yaml.safe_load(yml.read_text(encoding="utf-8")) or {}
            except yaml.YAMLError:
                continue
            for rule in doc.get("rules", []) or []:
                if not isinstance(rule, dict):
                    continue
                match = rule.get("match") or {}
                if not isinstance(match, dict):
                    continue
                applies = rule.get("applies_to") or []
                tool_token = SDK_TO_APPLIES_TO.get(sdk)
                is_tool_rule = not tool_token or tool_token in applies
                _attribute_coverage(covered[sdk], match, is_tool_rule)
    return covered


def _attribute_coverage(
    bucket: set[str], match: dict, is_tool_rule: bool = True
) -> None:
    keys, body_texts, param_values, pred_values = _collect_match_signals(match)

    # Tool-feature attribution is gated on the rule actually being a
    # tool-scope rule for this SDK's tool kind — otherwise an mcp_tool rule
    # in the same dir would wrongly mark openai_tool features covered.
    if is_tool_rule:
        body_blob = " ".join(body_texts)
        for feature, fragments in BODY_TEXT_SIGNATURES.items():
            if any(frag in body_blob for frag in fragments):
                bucket.add(feature)

        param_blob = " ".join(param_values)
        for feature, fragments in PARAM_NAME_SIGNATURES.items():
            if any(frag in param_blob for frag in fragments):
                bucket.add(feature)

        for feature, key_options in MATCH_KEY_SIGNATURES.items():
            if keys & set(key_options):
                bucket.add(feature)

    # Agent/subagent coverage is namespaced and always attributed.
    _attribute_new_scope_coverage(bucket, keys, pred_values)


def _attribute_new_scope_coverage(
    bucket: set[str], keys: set[str], pred_values: dict[str, list[str]]
) -> None:
    """Attribute agent/subagent coverage from list-valued predicates. Feature
    names are namespaced "<scope>:<feature>" to match the candidate side."""
    sub_tools = set(pred_values.get("subagent_grants_tool", []))
    if "Bash" in sub_tools:
        bucket.add("subagent:grants_bash")
    if "Write" in sub_tools:
        bucket.add("subagent:grants_write")
    if {"Bash", "Write"} <= sub_tools:
        bucket.add("subagent:grants_bash_and_write")
    if sub_tools & {"WebFetch", "WebSearch"}:
        bucket.add("subagent:grants_webfetch")

    agent_tools = set(pred_values.get("agent_grants_builtin_tool", []))
    agent_tools |= set(pred_values.get("agent_uses_hosted_tool_class", []))
    if "Bash" in agent_tools:
        bucket.add("agent:grants_bash")
    if "WebSearch" in agent_tools:
        bucket.add("agent:grants_websearch")

    if keys & {"agent_kwarg_list_empty", "agent_kwarg_missing"}:
        guard_vals = set(pred_values.get("agent_kwarg_list_empty", []))
        guard_vals |= set(pred_values.get("agent_kwarg_missing", []))
        if guard_vals & {"input_guardrails", "output_guardrails"}:
            bucket.add("agent:missing_guardrails")

    # Repo coverage: a rule that inspects a repo component (typically under a
    # `not`, e.g. "fires when claude_md absent") covers that repo feature.
    comp = set(pred_values.get("repo_component_present", []))
    if "claude_md" in comp:
        bucket.add("repo:uses_sdk_no_claude_md")
    if "claude_settings" in comp:
        bucket.add("repo:subagents_no_settings")
        bucket.add("repo:shell_tools_no_settings")


def _collect_match_signals(
    node: object,
) -> tuple[set[str], list[str], list[str], dict[str, list[str]]]:
    """Recursively gather predicate keys, has_body_text fragments,
    param_name_matches values, and per-predicate scalar list values from a
    match block.

    Descends through all/any/not combinators so a predicate nested inside
    a boolean group (the common case) is still attributed. Without this,
    a rule like `match: {all: [{name_has_prefix: ...}, {not: ...}]}` looks
    like it only has the key `all` and its real predicates are invisible.
    """
    keys: set[str] = set()
    body: list[str] = []
    params: list[str] = []
    pred_values: dict[str, list[str]] = {}
    if isinstance(node, dict):
        for key, value in node.items():
            keys.add(key)
            if key == "has_body_text" and isinstance(value, list):
                body.extend(str(t) for t in value)
            elif key == "param_name_matches" and isinstance(value, dict):
                for sub in value.values():
                    if isinstance(sub, list):
                        params.extend(str(x) for x in sub)
                    else:
                        params.append(str(sub))
            elif isinstance(value, list) and all(
                isinstance(x, (str, int, float, bool)) for x in value
            ):
                pred_values.setdefault(key, []).extend(str(x) for x in value)
            sub_keys, sub_body, sub_params, sub_pred = _collect_match_signals(
                value
            )
            keys |= sub_keys
            body.extend(sub_body)
            params.extend(sub_params)
            for k, v in sub_pred.items():
                pred_values.setdefault(k, []).extend(v)
    elif isinstance(node, list):
        for item in node:
            sub_keys, sub_body, sub_params, sub_pred = _collect_match_signals(
                item
            )
            keys |= sub_keys
            body.extend(sub_body)
            params.extend(sub_params)
            for k, v in sub_pred.items():
                pred_values.setdefault(k, []).extend(v)
    return keys, body, params, pred_values


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
