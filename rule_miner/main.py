"""CLI entrypoint: clone targets, scan, aggregate uncovered features,
then hand off to the LLM agent to write rule YAML into the local
trustabl-rules pack."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

from . import agent, discover, patterns, scanner, trustabl_scanner
from .tools import CandidatePattern, MiningState

CACHE_ROOT = Path.home() / ".cache" / "trustabl-rule-miner"
DEFAULT_RULES_REPO_SIBLING = Path(__file__).resolve().parents[2] / "trustabl-rules"


def cli() -> int:
    parser = argparse.ArgumentParser(
        prog="rule-miner",
        description=(
            "Mines official agent-SDK sample repos for new policy-rule "
            "candidates and writes draft rule YAML into the local "
            "trustabl-rules pack."
        ),
    )
    parser.add_argument(
        "--rules-repo",
        type=Path,
        help="Local path to a trustabl-rules checkout. Defaults to a sibling "
        f"directory at {DEFAULT_RULES_REPO_SIBLING}.",
    )
    parser.add_argument(
        "--targets",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "targets.json",
        help="Path to targets.json (default: bundled).",
    )
    parser.add_argument(
        "--only-sdk",
        choices=["openai_agents", "claude_agent_sdk", "google_adk"],
        help="Restrict mining to one SDK.",
    )
    parser.add_argument(
        "--min-occurrences",
        type=int,
        default=3,
        help="Minimum occurrences of an uncovered feature to surface as "
        "a candidate (default: 3).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the YAML each rule would land in, do not write files.",
    )
    parser.add_argument(
        "--discover",
        action="store_true",
        help="Before scanning, query Sourcegraph for public repos that import "
        "each SDK and merge them into the target list for this run.",
    )
    parser.add_argument(
        "--discover-limit",
        type=int,
        default=100,
        help="Per-SDK cap on repos discovered (default: 100).",
    )
    parser.add_argument(
        "--discover-write",
        action="store_true",
        help="With --discover, also persist the merged target list back to "
        "--targets PATH (default: in-memory for this run only).",
    )
    parser.add_argument(
        "--use-trustabl",
        choices=["auto", "on", "off"],
        default="auto",
        help="Use the trustabl Go binary as an additional scanner for tools, "
        "agents, and subagents. 'auto' (default) uses it when found on PATH. "
        "Requires a separate `go build` of github.com/trustabl/trustabl.",
    )
    args = parser.parse_args()

    trustabl_enabled = (
        args.use_trustabl == "on"
        or (args.use_trustabl == "auto" and trustabl_scanner.available())
    )
    if args.use_trustabl == "on" and not trustabl_scanner.available():
        print("error: --use-trustabl on but `trustabl` binary not on PATH",
              file=sys.stderr)
        return 2

    rules_repo_path = _resolve_rules_repo(args.rules_repo)
    if rules_repo_path is None:
        return 2

    targets = json.loads(args.targets.read_text(encoding="utf-8"))

    if args.discover:
        sdks_to_discover = (
            [args.only_sdk] if args.only_sdk else list(discover.SDK_QUERY)
        )
        discovered: list[discover.DiscoveredTarget] = []
        for sdk in sdks_to_discover:
            try:
                hits = discover.discover(sdk, limit=args.discover_limit)
            except Exception as exc:  # noqa: BLE001 -- network/parse errors
                print(f"discover({sdk}): {exc}", file=sys.stderr)
                continue
            print(f"  discover({sdk}): {len(hits)} repos", file=sys.stderr)
            discovered.extend(hits)
        targets, added = discover.merge_into_targets(targets, discovered)
        print(f"  added {added} new targets", file=sys.stderr)
        if args.discover_write and added:
            args.targets.write_text(
                json.dumps(targets, indent=2) + "\n", encoding="utf-8"
            )
            print(f"  wrote merged targets -> {args.targets}", file=sys.stderr)

    if args.only_sdk:
        targets = [t for t in targets if t["sdk"] == args.only_sdk]
    if not targets:
        print("no targets to scan", file=sys.stderr)
        return 1

    # Step 1-3: clone + scan
    all_records = []
    all_agents: list = []
    all_subagents: list = []
    for tgt in targets:
        try:
            clone_path = _ensure_clone(tgt["repo"], tgt.get("ref", "main"))
        except subprocess.CalledProcessError as exc:
            print(f"  {tgt['repo']}: clone failed ({exc.returncode}) -- skipped",
                  file=sys.stderr)
            continue
        roots = [clone_path / p for p in tgt["paths"]]
        records = scanner.scan_paths(tgt["repo"], tgt["sdk"], roots)
        if trustabl_enabled:
            try:
                tr = trustabl_scanner.scan(tgt["repo"], clone_path)
            except Exception as exc:  # noqa: BLE001
                print(f"  {tgt['repo']}: trustabl scan failed -- {exc}",
                      file=sys.stderr)
            else:
                merged = trustabl_scanner.merge_tools(records, tr.tools)
                added = len(merged) - len(records)
                records = merged
                all_agents.extend(tr.agents)
                all_subagents.extend(tr.subagents)
                print(
                    f"  {tgt['repo']}: trustabl +{added} tools, "
                    f"{len(tr.agents)} agents, {len(tr.subagents)} subagents",
                    file=sys.stderr,
                )
        print(f"  {tgt['repo']}: {len(records)} tools", file=sys.stderr)
        all_records.extend(records)

    # Step 4-5: feature-match + aggregate
    candidates = _aggregate_candidates(all_records, args.min_occurrences)
    if not candidates:
        print("no uncovered patterns crossed the threshold; nothing to draft.",
              file=sys.stderr)
        return 0

    # Step 6: hand off to agent
    state = MiningState(
        repo_root=rules_repo_path.resolve(),
        dry_run=args.dry_run,
        candidates=candidates,
    )
    agent.run(state)

    print(f"\nwrote {len(state.written_rules)} rules:")
    for rule_id, path in state.written_rules:
        print(f"  {rule_id} -> {path}")
    print(
        "\nRemember to mirror these into the engine's testdata/rules-fixture/\n"
        "per trustabl-rules CLAUDE.md step 5."
    )
    return 0


def _resolve_rules_repo(explicit: Path | None) -> Path | None:
    if explicit is not None:
        if not explicit.exists():
            print(f"error: --rules-repo path does not exist: {explicit}",
                  file=sys.stderr)
            return None
        return explicit
    if DEFAULT_RULES_REPO_SIBLING.exists():
        print(f"using rules repo at {DEFAULT_RULES_REPO_SIBLING}",
              file=sys.stderr)
        return DEFAULT_RULES_REPO_SIBLING
    print(
        "error: no trustabl-rules clone found. Pass --rules-repo PATH or "
        f"clone it next to this repo (expected at {DEFAULT_RULES_REPO_SIBLING}).",
        file=sys.stderr,
    )
    return None


def _ensure_clone(repo: str, ref: str) -> Path:
    target = CACHE_ROOT / repo.replace("/", "__")
    if not target.exists():
        CACHE_ROOT.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ["git", "clone", "--depth", "1", "--branch", ref,
             f"https://github.com/{repo}.git", str(target)],
            check=True,
        )
    else:
        subprocess.run(
            ["git", "-C", str(target), "fetch", "--depth", "1", "origin", ref],
            check=True,
        )
        subprocess.run(
            ["git", "-C", str(target), "reset", "--hard", f"origin/{ref}"],
            check=True,
        )
    return target


def _aggregate_candidates(
    records: list[scanner.ToolRecord], min_occurrences: int,
) -> list[CandidatePattern]:
    buckets: dict[tuple[str, str], list[scanner.ToolRecord]] = defaultdict(list)
    for rec in records:
        for feature in patterns.uncovered_features(rec):
            buckets[(rec.sdk, feature)].append(rec)

    out: list[CandidatePattern] = []
    for (sdk, feature), recs in buckets.items():
        if len(recs) < min_occurrences:
            continue
        out.append(
            CandidatePattern(
                sdk=sdk,
                feature=feature,
                occurrence_count=len(recs),
                example_callsites=[(r.file, r.line, r.name) for r in recs],
            )
        )
    out.sort(key=lambda c: c.occurrence_count, reverse=True)
    return out


if __name__ == "__main__":
    sys.exit(cli())
