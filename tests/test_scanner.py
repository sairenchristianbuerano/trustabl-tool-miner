"""Fixture-driven tests for the AST scanner.

These pin the extracted ToolRecord shape so future scanner edits don't
silently lose features.
"""

from pathlib import Path

from rule_miner import scanner

FIXTURES = Path(__file__).parent / "fixtures"


def _scan(sdk: str, fixture: str):
    return scanner.scan_paths(
        repo="local/fixtures",
        sdk=sdk,
        roots=[FIXTURES / fixture],
    )


def test_openai_function_tool_extracted():
    records = scanner.scan_paths(
        repo="local/fixtures",
        sdk="openai_agents",
        roots=[FIXTURES],
    )
    names = {r.name for r in records}
    assert "create_payment" in names
    assert "process" in names
    assert "not_a_tool" not in names


def test_openai_typing_and_docstring_flags():
    records = scanner.scan_paths(
        repo="local/fixtures",
        sdk="openai_agents",
        roots=[FIXTURES],
    )
    by_name = {r.name: r for r in records}
    assert by_name["create_payment"].has_docstring is True
    assert by_name["create_payment"].typed_params is True
    assert by_name["process"].has_docstring is False
    assert by_name["process"].typed_params is False


def test_openai_decorator_kwargs_captured():
    records = scanner.scan_paths(
        repo="local/fixtures",
        sdk="openai_agents",
        roots=[FIXTURES],
    )
    by_name = {r.name: r for r in records}
    assert by_name["process"].decorator_kwargs.get("strict_mode") == "False"


def test_openai_body_calls_captured():
    records = scanner.scan_paths(
        repo="local/fixtures",
        sdk="openai_agents",
        roots=[FIXTURES],
    )
    by_name = {r.name: r for r in records}
    assert "requests.post" in by_name["create_payment"].body_call_targets


def test_claude_tool_extracted():
    records = scanner.scan_paths(
        repo="local/fixtures",
        sdk="claude_agent_sdk",
        roots=[FIXTURES],
    )
    names = {r.name for r in records}
    assert {"send_email", "lookup_user"} <= names


def test_adk_function_tool_wrapper_extracted():
    records = scanner.scan_paths(
        repo="local/fixtures",
        sdk="google_adk",
        roots=[FIXTURES],
    )
    names = {r.name for r in records}
    assert {"get_weather", "run"} <= names


def test_adk_typed_params_inferred_from_wrapped_fn():
    records = scanner.scan_paths(
        repo="local/fixtures",
        sdk="google_adk",
        roots=[FIXTURES],
    )
    by_name = {r.name: r for r in records}
    assert by_name["get_weather"].typed_params is True
    assert by_name["run"].typed_params is True  # no params = vacuously typed
