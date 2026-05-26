# trustabl-rule-miner

A standalone CLI that mines official agent-SDK sample repos for **new
policy-rule candidates** and writes draft rule YAML directly into a local
[`trustabl-rules`](https://github.com/trustabl/trustabl-rules) checkout
for human review at PR time. Lives outside the rules repo so its `.yaml`
config never collides with the engine's rule loader.

## What it does

1. Reads `targets.json` — list of `{repo, sdk, paths, ref}` to scan
2. Shallow-clones each target into `~/.cache/trustabl-rule-miner/`
3. Walks each tree with Python's `ast` module to find every tool
   definition (`@function_tool`, `@tool`, `FunctionTool(fn)`)
4. Feature-matches each tool against the **existing** rule pack in the
   local `trustabl-rules` checkout. Patterns that already have a rule are
   silently covered; patterns that recur ≥ N times **without** a matching
   rule become candidates
5. Hands the candidate list to a Claude Agent SDK agent that drafts rule
   YAML + a threat-model paragraph and **appends each rule directly into
   `<rules_repo>/<sdk_dir>/<topic>.yaml`**

No GitHub issues, no `gh` CLI. Human review happens at PR time on the
rules repo.

## Install

```
pip install -e .
```

Requires Python ≥ 3.10.

**Agent auth.** The agent step uses the
[Claude Code CLI](https://docs.anthropic.com/claude/docs/claude-code)
under the hood, so a Claude Pro or Max subscription is enough — no
`ANTHROPIC_API_KEY` required. Install Claude Code, then run
`claude /login` once.

## Usage

```
rule-miner [options]
```

By default `rule-miner` writes into a sibling `../trustabl-rules` checkout
(i.e. a `trustabl-rules` folder next to this repo). Override the path
with `--rules-repo` when it lives elsewhere.

Options:

- `--rules-repo PATH` — local clone of `trustabl-rules` to write into.
  Default: sibling `../trustabl-rules`.
- `--targets PATH` — override the bundled `targets.json`.
- `--only-sdk {openai_agents,claude_agent_sdk,google_adk}` — restrict.
- `--min-occurrences N` — minimum recurrence to flag (default 3).
- `--dry-run` — print the YAML each rule would land in, do not write.
- `--discover` — before scanning, query Sourcegraph's public stream API
  for repos that import each SDK and merge them into the target list for
  this run. Cross-forge (GitHub / GitLab / Bitbucket) — currently every
  hit is github.com because the three SDKs have ~0 adoption elsewhere.
- `--discover-limit N` — per-SDK cap on discovered repos (default 100).
- `--discover-write` — with `--discover`, also persist the merged target
  list back into `--targets` PATH.
- `--use-trustabl {auto,on,off}` — use the
  [trustabl](https://github.com/trustabl/trustabl) Go binary as an
  additional scanner. Adds Python + TypeScript inventory of tools, agents,
  subagents, MCP servers, hosted tools. Default `auto`: enabled when the
  binary is on PATH. Build trustabl with `CGO_ENABLED=1 go build -o trustabl ./cmd/trustabl`.

### First run (recommended)

```
rule-miner --only-sdk openai_agents --dry-run
```

Spot-check the proposed rules before letting the agent write real files.

## Scheduling

Not built in. Two opt-in patterns:

**Linux/macOS cron** — weekly Monday 09:00:

```
0 9 * * 1  cd /path/to/rule-miner && /path/to/python -m rule_miner
```

**Windows Task Scheduler** — `schtasks /create /sc weekly /d MON /st 09:00 ...`

## How patterns map to candidates

`rule_miner/patterns.py` encodes each shipped rule's `match:` predicate
as a Python check. When the scanner reports a tool exhibiting feature
`X` and the SDK's `COVERED_BY_RULE[X]` entry is missing, that's a
candidate. Add a new entry once a rule ships to silence false-positive
candidates next run.

Currently watched features (uncovered ones become candidates):

| Feature | Description |
| --- | --- |
| `missing_docstring` | tool has no docstring |
| `missing_typed_params` | tool has untyped params |
| `ambiguous_name` | name in {`process`, `handle`, `run`, …} |
| `mutating_prefix_no_idempotency_kwarg` | `create_*` etc. without idempotency hint |
| `network_call` | tool body calls `requests.*` / `httpx.*` |
| `calls_subprocess` | tool body calls `subprocess.*` (**no shipped rule yet**) |
| `calls_shell_true` | `os.system` / `os.popen` (**no shipped rule yet**) |
| `uses_pickle` | `pickle.*` deserialization (**no shipped rule yet**) |
| `writes_env_var` | mutates `os.environ` (**no shipped rule yet**) |

The bottom four are why this tool exists — they'll fire on real repos
today and the agent will draft rules for them on first run.

After a real (non dry-run) write, mirror each new rule into the engine's
`testdata/rules-fixture/` and add fire/silent cases per the rules-repo
`CLAUDE.md` step 5 before merging.

## Layout

```
rule-miner/
├── pyproject.toml
├── targets.json
├── rule_miner/
│   ├── __init__.py
│   ├── main.py               CLI entrypoint
│   ├── scanner.py            stdlib AST tool discovery
│   ├── trustabl_scanner.py   optional adapter for the trustabl binary
│   ├── discover.py           Sourcegraph stream API -> targets
│   ├── patterns.py           feature → covered-by-rule mapping
│   ├── agent.py              Claude Agent SDK orchestration
│   └── tools.py              framework-agnostic tool callables
└── tests/
    ├── fixtures/
    └── test_scanner.py
```

## Testing

```
pip install -e .[dev]
pytest tests/
```

The fixture-driven tests pin the scanner's `ToolRecord` shape. They do
NOT exercise the LLM agent (no API calls in test suite).
