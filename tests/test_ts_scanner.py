"""Tests for the regex-based TypeScript tool scanner + language threading."""

from pathlib import Path

from rule_miner import patterns, ts_scanner

FIXTURES = Path(__file__).parent / "fixtures"


def _scan(sdk="claude_agent_sdk"):
    return {
        r.name: r
        for r in ts_scanner.scan_paths("local/fixtures", sdk, [FIXTURES])
    }


def test_ts_factory_and_object_forms_extracted():
    recs = _scan()
    assert {"fetch_data", "run_cmd", "send_email"} <= set(recs)
    assert all(r.language == "typescript" for r in recs.values())


def test_ts_docstring_flag():
    recs = _scan()
    assert recs["fetch_data"].has_docstring is True
    assert recs["run_cmd"].has_docstring is False  # empty description


def test_ts_body_calls_captured():
    recs = _scan()
    assert "fetch" in recs["fetch_data"].body_call_targets
    assert "execSync" in recs["run_cmd"].body_call_targets
    assert any(c.startswith("axios") for c in recs["send_email"].body_call_targets)


def test_ts_features_fire_via_shared_checks():
    recs = _scan()
    assert "network_call" in patterns.features_present(recs["fetch_data"])
    assert "calls_subprocess" in patterns.features_present(recs["run_cmd"])
    assert "missing_docstring" in patterns.features_present(recs["run_cmd"])
    # TS is typed; we report typed_params True so this never false-fires.
    assert "missing_typed_params" not in patterns.features_present(recs["fetch_data"])


def test_ts_coverage_namespaced_by_language():
    recs = _scan()
    rec = recs["fetch_data"]
    # A python network rule (bare "network_call") must NOT silence a TS candidate.
    py_covered = {"claude_agent_sdk": {"network_call"}}
    assert "network_call" in patterns.uncovered_features(rec, py_covered)
    # The TS-namespaced coverage entry does silence it.
    ts_covered = {"claude_agent_sdk": {"typescript:network_call"}}
    assert "network_call" not in patterns.uncovered_features(rec, ts_covered)
