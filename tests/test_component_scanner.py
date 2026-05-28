"""Tests for the pure-Python non-tool scanners, scope-aware features,
coverage attribution, and scope-keyed write validation."""

from pathlib import Path

from rule_miner import component_scanner as cs
from rule_miner import patterns, tools
from rule_miner.component_scanner import RepoComponents, SkillRecord
from rule_miner.tools import MiningState

REPO = Path(__file__).parent / "fixtures" / "sample_repo"


# ── scanners ────────────────────────────────────────────────────────────

def test_scan_subagents_parses_frontmatter():
    subs = {s.name: s for s in cs.scan_subagents("local/repo", REPO)}
    assert set(subs) == {"inbox-searcher", "writer"}
    # comma-string tools field
    assert subs["inbox-searcher"].tools == ("Read", "Bash", "Grep")
    assert subs["inbox-searcher"].model == "sonnet"
    # list tools field
    assert subs["writer"].tools == ("Write", "Bash", "WebFetch")


def test_scan_skills_parses_frontmatter():
    skills = {s.name: s for s in cs.scan_skills("local/repo", REPO)}
    assert "deployer" in skills
    assert skills["deployer"].description == ""
    assert skills["deployer"].allowed_tools == ()


def test_scan_components_presence_flags():
    comp = cs.scan_components("local/repo", REPO)
    assert comp.claude_md is True
    assert comp.claude_settings is True
    assert comp.subagent is True
    assert comp.hook_script is True
    assert comp.mcp_config is True
    assert comp.slash_command is False
    assert comp.dependency_manifest is False
    assert "subagent" in comp.present()


def test_scan_agents_ast_detection():
    agents = {a.name: a for a in cs.scan_agents("local/repo", [REPO])}
    assert set(agents) >= {"support", "researcher", "guarded"}
    assert agents["support"].sdk == "openai_agents"
    assert "Bash" in agents["support"].tool_grants
    assert agents["researcher"].sdk == "claude_agent_sdk"
    assert "WebSearch" in agents["researcher"].tool_grants


# ── feature detectors ───────────────────────────────────────────────────

def test_subagent_features():
    subs = {s.name: s for s in cs.scan_subagents("local/repo", REPO)}
    inbox = patterns.subagent_features_present(subs["inbox-searcher"])
    assert "grants_bash" in inbox
    assert "grants_write" not in inbox
    writer = patterns.subagent_features_present(subs["writer"])
    assert {"grants_bash", "grants_write", "grants_bash_and_write",
            "grants_webfetch"} <= writer


def test_agent_features_guardrails_and_grants():
    agents = {a.name: a for a in cs.scan_agents("local/repo", [REPO])}
    support = patterns.agent_features_present(agents["support"])
    assert "grants_bash" in support
    assert "missing_guardrails" in support  # openai, no guardrails
    guarded = patterns.agent_features_present(agents["guarded"])
    assert "missing_guardrails" not in guarded  # has input_guardrails kwarg
    researcher = patterns.agent_features_present(agents["researcher"])
    assert "grants_websearch" in researcher
    assert "missing_guardrails" not in researcher  # not openai


def test_skill_features():
    sk = SkillRecord("r", "f", "deployer", "", ())
    feats = patterns.skill_features_present(sk)
    assert {"skill_no_description", "skill_grants_broad_tools"} <= feats


def test_repo_features():
    bare = RepoComponents(repo="r", claude_md=False, subagent=True,
                          claude_settings=False)
    feats = patterns.repo_features(bare, {"claude_agent_sdk"}, has_shell=False)
    assert "uses_sdk_no_claude_md" in feats
    assert "subagents_no_settings" in feats

    healthy = RepoComponents(repo="r", claude_md=True, subagent=True,
                             claude_settings=True)
    assert patterns.repo_features(healthy, {"claude_agent_sdk"}, False) == set()


# ── coverage attribution (namespaced) ───────────────────────────────────

def test_uncovered_scoped_respects_namespaced_coverage():
    present = {"grants_bash"}
    covered = {"claude_agent_sdk": {"subagent:grants_bash"}}
    assert patterns.uncovered_scoped(
        "subagent", "claude_agent_sdk", present, covered
    ) == set()
    assert patterns.uncovered_scoped(
        "subagent", "claude_agent_sdk", present, {}
    ) == {"grants_bash"}


# ── scope-keyed write validation ────────────────────────────────────────

def _state(tmp_path) -> MiningState:
    return MiningState(repo_root=tmp_path, dry_run=True)


def _base_draft(**over) -> dict:
    d = {
        "id": "CSDK-901",
        "title": "x",
        "severity": "high",
        "confidence": 0.8,
        "applies_to": ["claude_subagent"],
        "match": {"subagent_grants_tool": ["Bash"]},
        "explanation": "because",
        "fix": "remove it",
        "sdk": "claude_agent_sdk",
        "scope": "subagent",
    }
    d.update(over)
    return d


def test_write_accepts_subagent_scope(tmp_path):
    res = tools.write_rule_yaml(
        _state(tmp_path), _base_draft(), "subagent_safety",
        rationale_md="body", owasp_refs=["LLM06"], fix_type="config",
        policy_meta={
            "id": "claude_sdk_subagent_safety", "name": "Subagent safety",
            "category": "claude_sdk", "description": "x",
        },
    )
    assert res.startswith("DRY_RUN"), res


def test_write_accepts_repo_scope(tmp_path):
    draft = _base_draft(
        id="OAI-901", sdk="openai_agents", scope="repo",
        applies_to=["openai_agents"],
        match={"repo_has_sdk_in_code": ["openai_agents"]},
    )
    res = tools.write_rule_yaml(
        _state(tmp_path), draft, "repo_hygiene",
        rationale_md="body", owasp_refs=["LLM06"], fix_type="config",
        policy_meta={
            "id": "openai_sdk_repo_hygiene", "name": "Repo hygiene",
            "category": "openai_sdk", "description": "x",
        },
    )
    assert res.startswith("DRY_RUN"), res


def test_write_rejects_tool_token_on_subagent_scope(tmp_path):
    bad = _base_draft(applies_to=["claude_sdk_tool"])
    res = tools.write_rule_yaml(_state(tmp_path), bad, "subagent_safety")
    assert res.startswith("REJECTED"), res


def test_write_rejects_unknown_scope(tmp_path):
    bad = _base_draft(scope="galaxy")
    res = tools.write_rule_yaml(_state(tmp_path), bad, "subagent_safety")
    assert res.startswith("REJECTED"), res
