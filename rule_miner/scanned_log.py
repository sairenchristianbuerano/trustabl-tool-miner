"""Persistent log of repos already scanned, so reruns can skip them.

Stored as JSONL at `~/.cache/trustabl-rule-miner/scanned.json` (one
JSON object per line: `{"repo": "...", "ref": "...", "tools": N,
"agents": N, "subagents": N, "scanned_at": "ISO-8601-UTC"}`). Keys are
`<repo>@<ref>`, so re-pinning a target to a different ref re-scans it.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

CACHE_ROOT = Path.home() / ".cache" / "trustabl-rule-miner"
DEFAULT_LOG = CACHE_ROOT / "scanned.json"


def _key(repo: str, ref: str) -> str:
    return f"{repo}@{ref}"


def load(path: Path = DEFAULT_LOG) -> set[str]:
    if not path.exists():
        return set()
    keys: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue
        repo = entry.get("repo")
        ref = entry.get("ref", "main")
        if repo:
            keys.add(_key(repo, ref))
    return keys


def record(
    repo: str,
    ref: str,
    tools: int,
    agents: int = 0,
    subagents: int = 0,
    url: str | None = None,
    path: Path = DEFAULT_LOG,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    entry = {
        "repo": repo,
        "url": url or f"https://github.com/{repo}.git",
        "ref": ref,
        "tools": tools,
        "agents": agents,
        "subagents": subagents,
        "scanned_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry) + "\n")


def reset(path: Path = DEFAULT_LOG) -> None:
    if path.exists():
        path.unlink()


def already_scanned(repo: str, ref: str, scanned: set[str]) -> bool:
    return _key(repo, ref) in scanned
