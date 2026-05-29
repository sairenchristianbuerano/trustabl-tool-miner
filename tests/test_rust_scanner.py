"""Tests for the regex Rust scanner (detection) + the write-language guard."""

from pathlib import Path

from rule_miner import patterns, rust_scanner, tools
from rule_miner.tools import MiningState

FIXTURES = Path(__file__).parent / "fixtures"


def _scan(sdk="claude_agent_sdk"):
    return {
        r.name: r
        for r in rust_scanner.scan_paths("local/fixtures", sdk, [FIXTURES])
    }


def test_rust_attr_fns_extracted():
    recs = _scan()
    assert {"fetch_doc", "run_shell"} <= set(recs)
    assert all(r.language == "rust" for r in recs.values())


def test_rust_docstring_and_calls():
    recs = _scan()
    assert recs["fetch_doc"].has_docstring is True   # /// above
    assert recs["run_shell"].has_docstring is False
    assert any(c.startswith("reqwest") for c in recs["fetch_doc"].body_call_targets)
    assert "Command.new" in recs["run_shell"].body_call_targets


def test_rust_features_fire():
    recs = _scan()
    assert "network_call" in patterns.features_present(recs["fetch_doc"])
    assert "calls_subprocess" in patterns.features_present(recs["run_shell"])


def test_write_rejects_rust_language(tmp_path):
    draft = {
        "id": "CSDK-902", "title": "x", "severity": "high", "confidence": 0.8,
        "applies_to": ["claude_sdk_tool"], "match": {"has_docstring": False},
        "explanation": "e", "fix": "f", "sdk": "claude_agent_sdk",
        "scope": "tool", "language": "rust",
    }
    res = tools.write_rule_yaml(
        MiningState(repo_root=tmp_path, dry_run=True), draft, "tool_definition"
    )
    assert res.startswith("REJECTED") and "language" in res, res


def test_write_accepts_typescript_language(tmp_path):
    draft = {
        "id": "CSDK-903", "title": "x", "severity": "medium", "confidence": 0.7,
        "applies_to": ["claude_sdk_tool"], "match": {"has_docstring": False},
        "explanation": "e", "fix": "f", "sdk": "claude_agent_sdk",
        "scope": "tool", "language": "typescript",
    }
    res = tools.write_rule_yaml(
        MiningState(repo_root=tmp_path, dry_run=True), draft, "tool_definition",
        policy_meta={"id": "claude_sdk_tool_definition", "name": "Tool def",
                     "category": "claude_sdk", "description": "x"},
    )
    assert res.startswith("DRY_RUN"), res
