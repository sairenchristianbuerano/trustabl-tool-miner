"""CLI entrypoint: clone targets, scan, aggregate uncovered features,
then hand off to the LLM agent to write rule YAML into the local
trustabl-rules pack."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import stat
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

from . import (
    agent,
    discover,
    heartbeat,
    patterns,
    scanned_log,
    scanner,
    trustabl_scanner,
)
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
    parser.add_argument(
        "--heartbeat-interval",
        type=float,
        default=60.0,
        help="Seconds between progress heartbeat lines on stderr (default 60). "
        "Set to 0 to disable.",
    )
    parser.add_argument(
        "--max-targets",
        type=int,
        default=None,
        help="Cap on number of target repos scanned per run (default 3). "
        "Applied after --discover merge. Set to 0 for no cap. Keeps disk "
        "usage bounded. When set explicitly, --max-runtime-seconds "
        "auto-disables (the repo cap is treated as authoritative).",
    )
    parser.add_argument(
        "--max-runtime-seconds",
        type=int,
        default=None,
        help="Stop the scan loop once total wall clock exceeds this many "
        "seconds (default 300 = 5 min). 0 disables. Auto-disabled when "
        "--max-targets is set explicitly.",
    )
    parser.add_argument(
        "--no-skip-scanned",
        action="store_true",
        help="Don't skip repos already recorded in the scanned-log. By "
        "default the miner skips any (repo@ref) it has scanned before.",
    )
    parser.add_argument(
        "--reset-scanned-log",
        action="store_true",
        help="Wipe the scanned-log before this run so every target is fair "
        "game again.",
    )
    parser.add_argument(
        "--keep-clones",
        action="store_true",
        help="Keep cached clones in ~/.cache/trustabl-rule-miner after each "
        "scan. By default the clone dir is deleted once the target's tools "
        "are recorded to the scanned-log -- frees disk between runs.",
    )
    args = parser.parse_args()

    explicit_max_targets = args.max_targets is not None
    if args.max_targets is None:
        args.max_targets = 3
    if args.max_runtime_seconds is None:
        args.max_runtime_seconds = 0 if explicit_max_targets else 300
    if explicit_max_targets:
        print(
            "  --max-targets set explicitly -- runtime cap disabled",
            file=sys.stderr,
        )

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

    if args.reset_scanned_log:
        scanned_log.reset()
        print("  wiped scanned-log", file=sys.stderr)
    scanned_keys = (
        set() if args.no_skip_scanned else scanned_log.load()
    )

    if not args.keep_clones:
        _sweep_orphan_clones(scanned_keys)
    if scanned_keys:
        before = len(targets)
        targets = [
            t for t in targets
            if not scanned_log.already_scanned(
                t["repo"], t.get("ref", "main"), scanned_keys
            )
        ]
        skipped = before - len(targets)
        if skipped:
            print(
                f"  skipping {skipped} previously-scanned repos "
                f"(--no-skip-scanned to override)",
                file=sys.stderr,
            )

    if args.max_targets > 0 and len(targets) > args.max_targets:
        print(
            f"  capping targets {len(targets)} -> {args.max_targets} "
            f"(--max-targets)",
            file=sys.stderr,
        )
        targets = targets[: args.max_targets]
    if not targets:
        print("no targets to scan", file=sys.stderr)
        return 1

    # Step 1-3: clone + scan
    hb: heartbeat.Heartbeat | None = None
    if args.heartbeat_interval > 0:
        hb = heartbeat.Heartbeat(interval=args.heartbeat_interval)
        hb.start(targets_total=len(targets))

    all_records = []
    all_agents: list = []
    all_subagents: list = []
    clones_to_clean: list[Path] = []
    deadline = (
        time.monotonic() + args.max_runtime_seconds
        if args.max_runtime_seconds > 0
        else None
    )
    try:
        for tgt in targets:
            if deadline is not None and time.monotonic() > deadline:
                print(
                    f"  deadline hit ({args.max_runtime_seconds}s) -- "
                    f"stopping scan loop early",
                    file=sys.stderr,
                )
                break
            if hb:
                hb.set_target(tgt["repo"])
            try:
                clone_path = _ensure_clone(tgt["repo"], tgt.get("ref", "main"))
            except subprocess.CalledProcessError as exc:
                print(f"  {tgt['repo']}: clone failed ({exc.returncode}) -- skipped",
                      file=sys.stderr)
                if hb:
                    hb.target_done()
                continue
            roots = [clone_path / p for p in tgt["paths"]]
            records = scanner.scan_paths(tgt["repo"], tgt["sdk"], roots)
            tr_agents: list = []
            tr_subagents: list = []
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
                    tr_agents = tr.agents
                    tr_subagents = tr.subagents
                    all_agents.extend(tr_agents)
                    all_subagents.extend(tr_subagents)
                    print(
                        f"  {tgt['repo']}: trustabl +{added} tools, "
                        f"{len(tr_agents)} agents, {len(tr_subagents)} subagents",
                        file=sys.stderr,
                    )
            print(f"  {tgt['repo']}: {len(records)} tools", file=sys.stderr)
            all_records.extend(records)
            scanned_log.record(
                repo=tgt["repo"],
                ref=tgt.get("ref", "main"),
                tools=len(records),
                agents=len(tr_agents),
                subagents=len(tr_subagents),
            )
            if not args.keep_clones:
                clones_to_clean.append(clone_path)
            if hb:
                hb.add_counts(
                    tools=len(records),
                    agents=len(tr_agents),
                    subagents=len(tr_subagents),
                )
                hb.target_done()
    finally:
        if hb:
            hb.stop()

    # Step 4-5: feature-match + aggregate
    candidates = _aggregate_candidates(all_records, args.min_occurrences)
    if not candidates:
        print("no uncovered patterns crossed the threshold; nothing to draft.",
              file=sys.stderr)
        _cleanup_clones(clones_to_clean)
        return 0

    # Step 6: hand off to agent. Clones MUST stay on disk through this step
    # so the agent's read_callsite tool can ground rule drafts in real code.
    state = MiningState(
        repo_root=rules_repo_path.resolve(),
        dry_run=args.dry_run,
        candidates=candidates,
    )
    try:
        agent.run(state)
    finally:
        _cleanup_clones(clones_to_clean)

    print(f"\nwrote {len(state.written_rules)} rules:")
    for rule_id, path in state.written_rules:
        print(f"  {rule_id} -> {path}")
    print(
        "\nRemember to mirror these into the engine's testdata/rules-fixture/\n"
        "per trustabl-rules CLAUDE.md step 5."
    )
    return 0


def _cleanup_clones(paths: list[Path]) -> None:
    for p in paths:
        if not p.exists():
            continue
        _rmtree_force(p)
        if p.exists():
            print(f"  WARNING: could not fully remove {p}", file=sys.stderr)
        else:
            print(f"  cleaned clone {p}", file=sys.stderr)


def _rmtree_force(path: Path) -> None:
    """rmtree that handles Windows read-only files inside `.git`.

    shutil.rmtree fails on .git/objects/pack/*.idx etc. without this
    onerror callback. Without it the dir is silently left half-deleted
    and the next git fetch trips on `fatal: not a git repository`.
    """
    def onerror(func, target, exc_info):
        try:
            os.chmod(target, stat.S_IWRITE)
            func(target)
        except Exception:
            pass

    shutil.rmtree(path, onerror=onerror)


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


def _sweep_orphan_clones(scanned_keys: set[str]) -> None:
    """Delete any cache subdir whose `owner__repo` matches a scanned-log entry.

    Catches clones left over from older runs (pre-cleanup-feature) or from
    runs that crashed before per-target rmtree fired. Skips dirs whose
    repo isn't in scanned-log so an in-progress clone isn't blown away.
    """
    if not CACHE_ROOT.exists():
        return
    scanned_repos = {k.split("@", 1)[0] for k in scanned_keys}
    swept = 0
    for entry in CACHE_ROOT.iterdir():
        if not entry.is_dir():
            continue
        repo = entry.name.replace("__", "/", 1)
        if repo in scanned_repos:
            _rmtree_force(entry)
            swept += 1
    if swept:
        print(f"  swept {swept} orphan clones from {CACHE_ROOT}",
              file=sys.stderr)


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
