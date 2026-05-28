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
from .tools import (
    MiningState,
    SDK_TO_AGENT_TOKENS,
    SDK_TO_REPO_TOKEN,
    SDK_TO_SKILL_TOKEN,
    SDK_TO_TOOL_TOKEN,
    SUBAGENT_TOKEN,
)

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
    "`draft` MUST match the trustabl rule schema (see `schema.yaml` at "
    "the rule-miner repo root for the full annotated reference). "
    "Required keys: id, title, severity "
    "(low|medium|high|critical — `info` reserved), confidence (0..1], "
    "applies_to (list — tool-scope: openai_tool / claude_sdk_tool / "
    "adk_function_tool / mcp_tool / shell_invocation), match (dict — "
    "use predicates from schema.yaml: has_body_text, has_docstring, "
    "call_without_kwarg, param_name_matches, all/any/not, etc.), "
    "explanation (multi-paragraph naming the consequence), fix "
    "(concrete remediation). Plus miner-internal `sdk` (openai_agents | "
    "claude_agent_sdk | google_adk). Optional: scope (tool|agent|subagent|"
    "repo|skill, default `tool`), language (default `python`; omitted "
    "automatically for subagent/skill). IMPORTANT: applies_to must fit the "
    "scope, not always the tool token — agent: openai_agent / "
    "openai_sandbox_agent / claude_agent_definition / adk_llm_agent / "
    "adk_sequential_agent / adk_parallel_agent / adk_loop_agent / "
    "adk_langgraph_agent; subagent: claude_subagent; repo: openai_agents / "
    "claude_sdk / google_adk; skill: skill (provisional). Use match "
    "predicates valid for the scope (agent_grants_builtin_tool, "
    "agent_kwarg_list_empty, subagent_grants_tool, repo_component_present, "
    "repo_has_sdk_in_code, etc.). "
    "`policy_meta`: REQUIRED when topic file doesn't exist yet — dict "
    "with id (lowercase_underscored, e.g. `openai_sdk_shell_safety`), "
    "name (human title), category (must match sdk: openai_sdk / "
    "claude_sdk / google_adk), description (multi-line intent). Omit "
    "when appending a rule to an existing topic file. The tool reorders "
    "keys into canonical schema order before writing. `topic`: bare "
    "filename stem. `rationale_md`: full "
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
        "policy_meta": dict,
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
        policy_meta=args.get("policy_meta"),
    )
    return {"content": [{"type": "text", "text": result}]}


def _read_or_placeholder(path: Path, placeholder: str) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return placeholder


def _system_prompt(repo_root: Path) -> str:
    contract = _read_or_placeholder(
        repo_root / "CLAUDE.md",
        "<CLAUDE.md missing from rules repo>",
    )
    template = _read_or_placeholder(
        repo_root / "docs" / "policy-rationale-doc-template-guide.md",
        "<rationale doc template missing — fall back to schema.yaml + "
        "write_rule_yaml tool description>",
    )
    miner_root = Path(__file__).resolve().parents[1]
    schema = _read_or_placeholder(
        miner_root / "schema.yaml",
        "<schema.yaml missing from rule-miner root>",
    )
    agent_tokens = {sdk: sorted(v) for sdk, v in SDK_TO_AGENT_TOKENS.items()}
    return f"""You are a rule-mining assistant for the Trustabl detection-rule
pack. Your task: for each candidate pattern surfaced by
list_candidate_patterns, write ONE rule directly into the local rules
pack at {repo_root}.

Each candidate carries a `scope` ("tool", "agent", "subagent", "repo", or
"skill") and a `feature`. The scope decides which `applies_to` tokens,
`match` predicates, ID range, and topic file are valid. Honor it exactly —
the writer REJECTS a draft whose applies_to doesn't fit its scope.

For each candidate:
  1. Confirm the pattern. For tool/agent candidates the examples have real
     file+line — call read_callsite on 2-3 of them. For subagent/skill the
     example file is a .md (read_callsite still works). For repo candidates
     there is no callsite (the "file" is the repo name); rely on the
     feature meaning + the authoring contract.
  2. Pick the `topic` filename under {repo_root}/<sdk_dir>/. Prefer an
     existing topic that fits the scope+feature:
       - tool   -> network.yaml, idempotency.yaml, path_safety.yaml,
                   error_handling.yaml, tool_definition.yaml, shell_safety.yaml
       - agent  -> agent_safety.yaml, builtin_tools.yaml
       - subagent -> subagent_safety.yaml
       - repo   -> a repo-level topic (e.g. repo_hygiene.yaml,
                   project_safety.yaml) — create if none fits
       - skill  -> skill_safety.yaml (provisional; see note below)
     Only create a new topic file (with policy_meta) when none is a clean fit.
  3. Build the draft:
       - `scope`: the candidate's scope verbatim.
       - `sdk`: one of {sorted(SDK_TO_TOOL_TOKEN.keys())}.
       - `id`: matching SDK ID prefix (CSDK- / OAI- / ADK-), lowest unused
         integer in the range for the scope: 0NN for tool, 1NN for
         agent/subagent, 2NN for repo. (skill: reuse the 1NN block,
         provisional.)
       - `applies_to`: a non-empty subset of the tokens valid for this
         scope+sdk. Token sets by scope:
           tool:     {SDK_TO_TOOL_TOKEN}
           agent:    {agent_tokens}
           subagent: ["{SUBAGENT_TOKEN}"] (claude_agent_sdk only)
           repo:     {SDK_TO_REPO_TOKEN}
           skill:    {SDK_TO_SKILL_TOKEN} (provisional)
       - `match`: use ONLY predicates from schema.yaml that are valid for
         this scope (e.g. agent_*/agent_grants_builtin_tool for agent,
         subagent_grants_tool for subagent, repo_component_present/
         repo_has_sdk_in_code for repo). Never invent a YAML key.
       - `severity`, `confidence`, `explanation`, `fix`: per the contract.
  4. Compose the rationale Markdown body per the template (every section,
     no skips). Inside "Real-world consequence" cite at least two
     callsites with full GitHub URLs of the form
     https://github.com/<repo>/blob/<ref>/<file_path>#L<line> built from
     the list_candidate_patterns examples (skip for repo candidates, which
     have no callsite). Pick OWASP LLM Top 10:2025 IDs anchoring the rule
     (LLM01..LLM10). Decide fix_type ('config' if the fix is
     hooks/guardrails/sandbox/agent kwargs/frontmatter, 'code' if it
     requires source edits). Then call write_rule_yaml(draft, topic,
     rationale_md, owasp_refs, fix_type). On REJECTED, fix the reported
     issue and retry once; if still rejected, skip and move on.

SKILL note: skill-scope rules are PROVISIONAL — the engine cannot evaluate
them yet. If you draft one, say so in its explanation ("Provisional: skill
scanning lands in a future engine release").

Stop when every candidate has been processed. Print a final summary
listing every rule_id written and the file path it landed in, then
remind the user to mirror the new rules into the engine's
testdata/rules-fixture/ per the authoring contract step 5.

=== Rule-authoring contract (verbatim) ===
{contract}

=== Rationale doc template (verbatim) ===
{template}

=== Trustabl rule schema (verbatim from rule-miner schema.yaml) ===
{schema}
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
