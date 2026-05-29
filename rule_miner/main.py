"""CLI entrypoint: clone targets, scan, aggregate uncovered features,
then hand off to the LLM agent to write rule YAML into the local
trustabl-rules pack."""

from __future__ import annotations

import argparse
import dataclasses
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
    component_scanner,
    discover,
    heartbeat,
    patterns,
    rust_scanner,
    scanned_log,
    scanner,
    trustabl_scanner,
    ts_scanner,
)
from .component_scanner import RepoComponents
from .tools import CandidatePattern, MiningState

CACHE_ROOT = Path.home() / ".cache" / "trustabl-rule-miner"
DEFAULT_RULES_REPO_SIBLING = Path(__file__).resolve().parents[2] / "trustabl-rules"
DEFAULT_RULEBOOK_SIBLING = Path(__file__).resolve().parents[2] / "trustabl-rulebook"


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
        "--rulebook-repo",
        type=Path,
        help="Local path to a trustabl-rulebook checkout. Rationale docs are "
        "written under <rulebook>/docs/Policy/<sdk>/<topic>.md (canonical YAML "
        "still goes to --rules-repo). Defaults to a sibling directory at "
        f"{DEFAULT_RULEBOOK_SIBLING}; falls back to writing docs into "
        "--rules-repo when no rulebook is found.",
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
        "--repo-min-occurrences",
        type=int,
        default=2,
        help="Minimum number of repos exhibiting a repo-scope gap to surface "
        "it as a candidate (default: 2). Repo features are per-repo booleans, "
        "so this is a separate, lower threshold than --min-occurrences.",
    )
    parser.add_argument(
        "--enable-skills",
        action="store_true",
        help="Mine .claude/skills/**/SKILL.md and draft skill-scope rules. "
        "Off by default: the trustabl engine cannot evaluate skill rules yet, "
        "so drafted skill rules are marked provisional until engine support "
        "ships.",
    )
    parser.add_argument(
        "--target-policies",
        type=int,
        default=None,
        metavar="N",
        help="Goal mode: keep scanning repos in batches until N distinct "
        "policy files (topic .yaml) have been created, then stop. Records each "
        "scanned repo to the scanned-log (history, no revisits) and cleans its "
        "clone between rounds to bound disk. Recomputes coverage each round so "
        "rules written earlier silence their own candidates (no duplicates). "
        "When the local target list is exhausted before N, pair with "
        "--discover to replenish (capped by --max-discover-rounds); otherwise "
        "stops and reports the shortfall.",
    )
    parser.add_argument(
        "--goal-batch",
        type=int,
        default=3,
        metavar="N",
        help="Repos scanned per goal-mode round (default 3). Keep >=2 so "
        "repo-scope rules (which need multiple repos) can still fire.",
    )
    parser.add_argument(
        "--max-discover-rounds",
        type=int,
        default=3,
        metavar="N",
        help="Goal mode: max --discover replenish rounds before giving up "
        "(default 3). Bounds the loop so it always terminates.",
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
    explicit_max_runtime = args.max_runtime_seconds is not None
    if args.max_targets is None:
        args.max_targets = 3
    if args.max_runtime_seconds is None:
        # Goal mode is bounded by target exhaustion + --max-discover-rounds,
        # so it runs without a wall-clock cap unless one is set explicitly.
        if args.target_policies:
            args.max_runtime_seconds = 0
        else:
            args.max_runtime_seconds = 0 if explicit_max_targets else 300
    if explicit_max_targets and not args.target_policies:
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
    rulebook_path = _resolve_rulebook(args.rulebook_repo)

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

    # Goal mode batches through the full list itself, so the per-run cap
    # does not apply there.
    if (
        not args.target_policies
        and args.max_targets > 0
        and len(targets) > args.max_targets
    ):
        print(
            f"  capping targets {len(targets)} -> {args.max_targets} "
            f"(--max-targets)",
            file=sys.stderr,
        )
        targets = targets[: args.max_targets]
    if not targets:
        print("no targets to scan", file=sys.stderr)
        return 1

    if args.target_policies:
        return _run_goal_loop(
            args, targets, rules_repo_path, rulebook_path, trustabl_enabled
        )

    # Step 1-3: clone + scan
    hb: heartbeat.Heartbeat | None = None
    if args.heartbeat_interval > 0:
        hb = heartbeat.Heartbeat(interval=args.heartbeat_interval)
        hb.start(targets_total=len(targets))

    all_records = []
    all_agents: list = []
    all_subagents: list = []
    all_skills: list = []
    all_repo_ctx: list[tuple[RepoComponents, set[str], bool]] = []
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
            res = _scan_target(tgt, args, trustabl_enabled, hb)
            if res is None:
                continue
            all_records.extend(res.records)
            all_agents.extend(res.agents)
            all_subagents.extend(res.subagents)
            all_skills.extend(res.skills)
            all_repo_ctx.append(res.repo_ctx)
            if not args.keep_clones:
                clones_to_clean.append(res.clone_path)
    finally:
        if hb:
            hb.stop()

    # Step 4-5: feature-match + aggregate
    covered = patterns.derive_covered_features(rules_repo_path)
    for sdk, feats in sorted(covered.items()):
        if feats:
            print(f"  covered({sdk}): {sorted(feats)}", file=sys.stderr)
    candidates = _aggregate_candidates(
        all_records,
        all_agents,
        all_subagents,
        all_skills,
        all_repo_ctx,
        args.min_occurrences,
        covered,
        repo_min=args.repo_min_occurrences,
    )
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
        rulebook_root=rulebook_path,
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


def _resolve_rulebook(explicit: Path | None) -> Path | None:
    """Resolve where rationale docs land. Returns None to fall back to writing
    docs into the rules repo (with a warning)."""
    if explicit is not None:
        if not explicit.exists():
            print(f"warning: --rulebook-repo path does not exist: {explicit} "
                  "-- writing rationale docs into the rules repo instead",
                  file=sys.stderr)
            return None
        print(f"using rulebook repo at {explicit}", file=sys.stderr)
        return explicit
    if DEFAULT_RULEBOOK_SIBLING.exists():
        print(f"using rulebook repo at {DEFAULT_RULEBOOK_SIBLING}",
              file=sys.stderr)
        return DEFAULT_RULEBOOK_SIBLING
    print("warning: no trustabl-rulebook clone found -- rationale docs will "
          "go into the rules repo. Pass --rulebook-repo PATH to redirect.",
          file=sys.stderr)
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
    records: list[scanner.ToolRecord],
    agents: list,
    subagents: list,
    skills: list,
    repo_ctx: list[tuple[RepoComponents, set[str], bool]],
    min_occurrences: int,
    covered: dict[str, set[str]] | None = None,
    repo_min: int = 2,
) -> list[CandidatePattern]:
    # bucket key: (sdk, scope, feature, language) -> list[(file, line, name)]
    buckets: dict[tuple[str, str, str, str], list[tuple[str, int, str]]] = (
        defaultdict(list)
    )

    for rec in records:
        lang = getattr(rec, "language", "python")
        for feature in patterns.uncovered_features(rec, covered):
            buckets[(rec.sdk, "tool", feature, lang)].append(
                (rec.file, rec.line, rec.name)
            )

    for a in agents:
        present = patterns.agent_features_present(a)
        for feature in patterns.uncovered_scoped("agent", a.sdk, present, covered):
            buckets[(a.sdk, "agent", feature, "python")].append(
                (a.file, a.line, a.name)
            )

    for s in subagents:
        present = patterns.subagent_features_present(s)
        for feature in patterns.uncovered_scoped(
            "subagent", "claude_agent_sdk", present, covered
        ):
            buckets[("claude_agent_sdk", "subagent", feature, "python")].append(
                (s.file, 0, s.name)
            )

    for sk in skills:
        present = patterns.skill_features_present(sk)
        for feature in patterns.uncovered_scoped(
            "skill", "claude_agent_sdk", present, covered
        ):
            buckets[("claude_agent_sdk", "skill", feature, "python")].append(
                (sk.file, 0, sk.name)
            )

    for components, sdks_in_repo, has_shell in repo_ctx:
        feats = patterns.repo_features(components, sdks_in_repo, has_shell)
        for feature in feats:
            for sdk in sdks_in_repo:
                if f"repo:{feature}" in (covered or {}).get(sdk, set()):
                    continue
                buckets[(sdk, "repo", feature, "python")].append(
                    (components.repo, 0, components.repo)
                )

    out: list[CandidatePattern] = []
    for (sdk, scope, feature, language), locs in buckets.items():
        threshold = repo_min if scope == "repo" else min_occurrences
        if len(locs) < threshold:
            continue
        out.append(
            CandidatePattern(
                sdk=sdk,
                feature=feature,
                occurrence_count=len(locs),
                example_callsites=locs,
                scope=scope,
                language=language,
            )
        )
    out.sort(key=lambda c: c.occurrence_count, reverse=True)
    return out


@dataclasses.dataclass
class _ScanResult:
    records: list
    agents: list
    subagents: list
    skills: list
    repo_ctx: tuple
    clone_path: Path


def _scan_target(
    tgt: dict, args, trustabl_enabled: bool, hb: "heartbeat.Heartbeat | None"
) -> _ScanResult | None:
    """Clone + scan one target into tool/agent/subagent/skill/component
    inventory. Records the scan to the scanned-log. Returns None on clone
    failure. Caller owns clone cleanup."""
    if hb:
        hb.set_target(tgt["repo"])
    try:
        clone_path = _ensure_clone(tgt["repo"], tgt.get("ref", "main"))
    except subprocess.CalledProcessError as exc:
        print(f"  {tgt['repo']}: clone failed ({exc.returncode}) -- skipped",
              file=sys.stderr)
        if hb:
            hb.target_done()
        return None
    roots = [clone_path / p for p in tgt["paths"]]
    records = scanner.scan_paths(tgt["repo"], tgt["sdk"], roots)
    records += ts_scanner.scan_paths(tgt["repo"], tgt["sdk"], roots)
    records += rust_scanner.scan_paths(tgt["repo"], tgt["sdk"], roots)
    tgt_agents: list = []
    tgt_subagents: list = []
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
            tgt_agents = tr.agents
            tgt_subagents = tr.subagents
            print(
                f"  {tgt['repo']}: trustabl +{added} tools, "
                f"{len(tgt_agents)} agents, {len(tgt_subagents)} subagents",
                file=sys.stderr,
            )
    else:
        # Self-owned pure-Python inventory of non-tool entities.
        tgt_agents = component_scanner.scan_agents(tgt["repo"], roots)
        tgt_subagents = component_scanner.scan_subagents(tgt["repo"], clone_path)

    components = component_scanner.scan_components(tgt["repo"], clone_path)
    tgt_skills = (
        component_scanner.scan_skills(tgt["repo"], clone_path)
        if args.enable_skills else []
    )
    has_shell = any(
        c.startswith("subprocess.") or c in ("os.system", "os.popen")
        for rec in records for c in rec.body_call_targets
    )

    print(
        f"  {tgt['repo']}: {len(records)} tools, {len(tgt_agents)} agents, "
        f"{len(tgt_subagents)} subagents, {len(tgt_skills)} skills",
        file=sys.stderr,
    )
    scanned_log.record(
        repo=tgt["repo"],
        ref=tgt.get("ref", "main"),
        tools=len(records),
        agents=len(tgt_agents),
        subagents=len(tgt_subagents),
    )
    if hb:
        hb.add_counts(
            tools=len(records),
            agents=len(tgt_agents),
            subagents=len(tgt_subagents),
        )
        hb.target_done()

    return _ScanResult(
        records=records,
        agents=tgt_agents,
        subagents=tgt_subagents,
        skills=tgt_skills,
        repo_ctx=(components, {tgt["sdk"]}, has_shell),
        clone_path=clone_path,
    )


def _goal_discover(args) -> list[dict]:
    """Replenish the target list via Sourcegraph discovery, filtered to
    repos not already in the scanned-log. Returns [] when --discover is off
    or nothing new is found (which lets the goal loop terminate)."""
    if not args.discover:
        return []
    sdks = [args.only_sdk] if args.only_sdk else list(discover.SDK_QUERY)
    found: list = []
    for sdk in sdks:
        try:
            found.extend(discover.discover(sdk, limit=args.discover_limit))
        except Exception as exc:  # noqa: BLE001 -- network/parse errors
            print(f"  goal discover({sdk}): {exc}", file=sys.stderr)
    merged, _ = discover.merge_into_targets([], found)
    scanned = scanned_log.load()
    return [
        t for t in merged
        if not scanned_log.already_scanned(
            t["repo"], t.get("ref", "main"), scanned
        )
    ]


def _run_goal_loop(
    args, targets: list[dict], rules_repo_path: Path,
    rulebook_path: Path | None, trustabl_enabled: bool,
) -> int:
    """Goal mode: scan repos in batches until N distinct policy files are
    created. Each round cleans its clones (bounded disk) and re-derives
    coverage (so rules written earlier silence their own candidates).
    Terminates on goal, target exhaustion (after optional --discover
    replenish), or an explicit runtime cap."""
    goal = args.target_policies
    batch_size = max(1, args.goal_batch)
    written_rules: list[tuple[str, str]] = []
    written_files: set[str] = set()
    start = time.monotonic()
    deadline = (
        start + args.max_runtime_seconds
        if args.max_runtime_seconds > 0 else None
    )
    pending = list(targets)
    round_no = 0
    discover_rounds = 0

    while len(written_files) < goal:
        if deadline is not None and time.monotonic() > deadline:
            print(f"goal: runtime cap ({args.max_runtime_seconds}s) hit -- "
                  f"stopping at {len(written_files)}/{goal} policy files",
                  file=sys.stderr)
            break
        if not pending:
            replenished = _goal_discover(args)
            if replenished and discover_rounds < args.max_discover_rounds:
                discover_rounds += 1
                pending.extend(replenished)
                print(f"  goal: discover round {discover_rounds} added "
                      f"{len(replenished)} repos", file=sys.stderr)
                continue
            print(f"goal: targets exhausted -- stopping at "
                  f"{len(written_files)}/{goal} policy files", file=sys.stderr)
            break

        round_no += 1
        batch = pending[:batch_size]
        pending = pending[batch_size:]
        elapsed = int(time.monotonic() - start)
        print(f"[goal] round {round_no}: {len(written_files)}/{goal} policy "
              f"files, elapsed {elapsed}s, scanning {len(batch)} repos",
              file=sys.stderr)

        hb = (
            heartbeat.Heartbeat(interval=args.heartbeat_interval)
            if args.heartbeat_interval > 0 else None
        )
        if hb:
            hb.start(targets_total=len(batch))
        recs: list = []
        ags: list = []
        subs: list = []
        sks: list = []
        rcs: list = []
        clones: list[Path] = []
        try:
            for tgt in batch:
                res = _scan_target(tgt, args, trustabl_enabled, hb)
                if res is None:
                    continue
                recs.extend(res.records)
                ags.extend(res.agents)
                subs.extend(res.subagents)
                sks.extend(res.skills)
                rcs.append(res.repo_ctx)
                clones.append(res.clone_path)
        finally:
            if hb:
                hb.stop()

        covered = patterns.derive_covered_features(rules_repo_path)
        candidates = _aggregate_candidates(
            recs, ags, subs, sks, rcs, args.min_occurrences, covered,
            repo_min=args.repo_min_occurrences,
        )
        if not candidates:
            _cleanup_clones(clones)
            continue

        state = MiningState(
            repo_root=rules_repo_path.resolve(),
            dry_run=args.dry_run,
            candidates=candidates,
            rulebook_root=rulebook_path,
        )
        try:
            agent.run(state)
        finally:
            _cleanup_clones(clones)
        for rule_id, path in state.written_rules:
            written_rules.append((rule_id, path))
            written_files.add(path)
        print(f"  goal: round {round_no} wrote {len(state.written_rules)} "
              f"rules; policy files now {len(written_files)}/{goal}",
              file=sys.stderr)

    print(f"\ngoal: {len(written_files)}/{goal} policy files, "
          f"{len(written_rules)} rules:")
    for rule_id, path in written_rules:
        print(f"  {rule_id} -> {path}")
    print(
        "\nRemember to mirror these into the engine's testdata/rules-fixture/\n"
        "per trustabl-rules CLAUDE.md step 5."
    )
    return 0 if len(written_files) >= goal else 1


if __name__ == "__main__":
    sys.exit(cli())
