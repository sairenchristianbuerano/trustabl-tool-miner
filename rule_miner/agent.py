"""Claude Agent SDK orchestration.

The LLM only handles step 6 of the pipeline: turning a list of uncovered
features into a coherent rule draft (title + threat-model + fix prose)
and writing the rule YAML straight into the local trustabl-rules pack.

Pre-computation (clone + scan + aggregate) runs deterministically in
`main.py` and lands in `MiningState.candidates` before `run()` is called.

Authentication: `claude-agent-sdk-python` shells out to the local Claude
Code CLI, so a Claude Pro/Max subscription (`claude /login`) authenticates
the agent step — no `ANTHROPIC_API_KEY` required.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from claude_agent_sdk import (  # type: ignore[import-not-found]
    ClaudeAgentOptions,
    create_sdk_mcp_server,
    query,
    tool,
)

from . import tools as miner_tools
from .tools import MiningState, SDK_TO_TOOL_TOKEN

_STATE: MiningState | None = None


def _state() -> MiningState:
    if _STATE is None:
        raise RuntimeError("agent.run() must initialize MiningState first")
    return _STATE


@tool(
    "list_candidate_patterns",
    "List every uncovered tool-feature pattern detected in the target repos. "
    "Returns JSON with sdk, feature name, occurrence count, and up to five "
    "example callsites per candidate.",
    {},
)
async def _t_list_candidates(args: dict) -> dict:
    payload = miner_tools.list_candidate_patterns(_state())
    return {"content": [{"type": "text", "text": payload}]}


@tool(
    "read_callsite",
    "Read a 30-line snippet from a Python file at the given line. Use to "
    "ground rule explanations in real source.",
    {"file": str, "line": int},
)
async def _t_read_callsite(args: dict) -> dict:
    snippet = miner_tools.read_callsite(args["file"], int(args["line"]))
    return {"content": [{"type": "text", "text": snippet}]}


@tool(
    "write_rule_yaml",
    "Write a rule + its paired rationale doc per the trustabl-rules "
    "template guide. Appends YAML to <rules_repo>/<sdk_dir>/<topic>.yaml "
    "AND writes the matching <rules_repo>/docs/Policy/<sdk_dir>/<topic>.md "
    "(template at docs/policy-rationale-doc-template-guide.md). "
    "`draft` MUST match the shipped YAML shape exactly (see "
    "`openai_sdk/observability.yaml` for OAI-010 as the canonical "
    "example). Required keys: id, title, severity (low|medium|high), "
    "confidence (0..1), applies_to (list — must include the SDK's tool "
    "token: openai_tool / claude_sdk_tool / adk_function_tool), match "
    "(dict — e.g. `{has_body_text: [\"print(\"]}`), explanation (str, "
    "single paragraph naming the consequence), fix (str, prescribes the "
    "concrete change). Plus miner-internal `sdk` (openai_agents | "
    "claude_agent_sdk | google_adk). Optional: scope (default `tool`), "
    "language (default `python`). The tool reorders keys into the "
    "canonical order before writing. `topic`: bare filename stem. "
    "`rationale_md`: full "
    "Markdown body BELOW the metadata block — 'What this policy covers', "
    "'Why <topic> is a distinct concern in agent tools' (threat model), "
    "'Rule-by-rule defense' with rule_id as H3 plus 'What we detect / Why "
    "it is flaggable / Real-world consequence (with specific callsite "
    "URLs like https://github.com/owner/repo/blob/main/file.py#L42) / Why "
    "severity is X and not Y / Fix type / Confidence Z', 'What this "
    "policy does not cover', 'Recommendations beyond the fix' with a full "
    "safe-code example. `owasp_refs`: list of OWASP LLM Top 10:2025 IDs "
    "(LLM01-LLM10) anchoring the rule. `fix_type`: 'config' or 'code'. "
    "Returns WROTE / DRY_RUN / REJECTED.",
    {
        "draft": dict,
        "topic": str,
        "rationale_md": str,
        "owasp_refs": list,
        "fix_type": str,
    },
)
async def _t_write_rule(args: dict) -> dict:
    result = miner_tools.write_rule_yaml(
        _state(),
        args["draft"],
        args["topic"],
        rationale_md=args.get("rationale_md"),
        owasp_refs=args.get("owasp_refs"),
        fix_type=args.get("fix_type", "code"),
    )
    return {"content": [{"type": "text", "text": result}]}


def _system_prompt(repo_root: Path) -> str:
    contract = (repo_root / "CLAUDE.md").read_text(encoding="utf-8")
    template = (
        repo_root / "docs" / "policy-rationale-doc-template-guide.md"
    ).read_text(encoding="utf-8")
    return f"""You are a rule-mining assistant for the Trustabl detection-rule
pack. Your task: for each candidate pattern surfaced by
list_candidate_patterns, write ONE rule directly into the local rules
pack at {repo_root}.

For each candidate:
  1. Call read_callsite on 2-3 example callsites to confirm the pattern.
  2. Pick the `topic` filename to write into. Prefer an existing topic
     file under {repo_root}/<sdk_dir>/ when one fits (e.g. network.yaml,
     idempotency.yaml, path_safety.yaml, error_handling.yaml,
     tool_definition.yaml, agent_safety.yaml). If no existing topic is
     a clean fit, create a new topic file with a short descriptive name
     (e.g. shell_safety, deserialization, env_safety).
  3. Build the draft:
       - `id`: matching SDK ID prefix (CSDK- / OAI- / GADK-), lowest
         unused integer in the appropriate range (NNN for tool scope,
         1NN for agent/subagent, 2NN for repo).
       - `sdk`: one of {sorted(SDK_TO_TOOL_TOKEN.keys())}.
       - `applies_to`: must include the matching tool token from this
         map: {SDK_TO_TOOL_TOKEN}.
       - `severity`, `confidence`, `match`, `explanation`, `fix`: per
         the authoring contract below.
  4. Compose the rationale Markdown body per the template (every section,
     no skips). Inside "Real-world consequence" cite at least two
     callsites with full GitHub URLs of the form
     https://github.com/<repo>/blob/<ref>/<file_path>#L<line> built from
     the list_candidate_patterns examples. Pick OWASP LLM Top 10:2025 IDs
     anchoring the rule (LLM01..LLM10). Decide fix_type ('config' if the
     fix is hooks/guardrails/sandbox/agent kwargs, 'code' if it requires
     tool source edits). Then call write_rule_yaml(draft, topic,
     rationale_md, owasp_refs, fix_type). On REJECTED, fix the reported
     issue and retry once; if still rejected, skip and move on.

Stop when every candidate has been processed. Print a final summary
listing every rule_id written and the file path it landed in, then
remind the user to mirror the new rules into the engine's
testdata/rules-fixture/ per the authoring contract step 5.

=== Rule-authoring contract (verbatim) ===
{contract}

=== Rationale doc template (verbatim) ===
{template}
"""


async def _run_async(state: MiningState) -> None:
    global _STATE
    _STATE = state

    server = create_sdk_mcp_server(
        name="rule_miner",
        version="0.1.0",
        tools=[
            _t_list_candidates,
            _t_read_callsite,
            _t_write_rule,
        ],
    )
    options = ClaudeAgentOptions(
        mcp_servers={"rule_miner": server},
        allowed_tools=[
            "mcp__rule_miner__list_candidate_patterns",
            "mcp__rule_miner__read_callsite",
            "mcp__rule_miner__write_rule_yaml",
        ],
        system_prompt=_system_prompt(state.repo_root),
    )
    prompt = (
        f"Process every candidate pattern. Rules repo: {state.repo_root}. "
        f"Dry run: {state.dry_run}. Begin by calling list_candidate_patterns."
    )
    async for message in query(prompt=prompt, options=options):
        _log(message)


def _log(message: object) -> None:
    """Surface every SDK message: type + content blocks + tool calls."""
    msg_type = type(message).__name__
    content = getattr(message, "content", None)
    if isinstance(content, list):
        for block in content:
            block_type = type(block).__name__
            text = getattr(block, "text", None)
            if text:
                print(f"[{msg_type}/{block_type}] {text}")
                continue
            tool_name = getattr(block, "name", None)
            if tool_name:
                tool_input = getattr(block, "input", None)
                input_summary = (
                    str(tool_input)[:300] if tool_input is not None else ""
                )
                print(f"[{msg_type}/{block_type}] tool={tool_name} input={input_summary}")
                continue
            tool_result = getattr(block, "content", None)
            if tool_result is not None:
                result_summary = str(tool_result)[:300]
                print(f"[{msg_type}/{block_type}] result={result_summary}")
                continue
            print(f"[{msg_type}/{block_type}] {block!r}")
        return
    text = getattr(message, "text", None)
    if text:
        print(f"[{msg_type}] {text}")
        return
    result = getattr(message, "result", None)
    if result is not None:
        print(f"[{msg_type}] result={str(result)[:500]}")
        return
    print(f"[{msg_type}] {message!r}")


def run(state: MiningState) -> None:
    asyncio.run(_run_async(state))
