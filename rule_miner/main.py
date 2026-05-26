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

from . import agent, patterns, scanner
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
    args = parser.parse_args()

    rules_repo_path = _resolve_rules_repo(args.rules_repo)
    if rules_repo_path is None:
        return 2

    targets = json.loads(args.targets.read_text(encoding="utf-8"))
    if args.only_sdk:
        targets = [t for t in targets if t["sdk"] == args.only_sdk]
    if not targets:
        print("no targets to scan", file=sys.stderr)
        return 1

    # Step 1-3: clone + scan
    all_records = []
    for tgt in targets:
        clone_path = _ensure_clone(tgt["repo"], tgt.get("ref", "main"))
        roots = [clone_path / p for p in tgt["paths"]]
        records = scanner.scan_paths(tgt["repo"], tgt["sdk"], roots)
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
