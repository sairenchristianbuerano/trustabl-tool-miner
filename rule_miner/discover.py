"""Cross-forge repo discovery via Sourcegraph's public stream API.

Sourcegraph indexes ~2.8M public repos across GitHub, GitLab, Bitbucket,
Gerrit, etc. We hit the unauthenticated `search/stream` endpoint with a
per-SDK query and parse the returned `repository` fields into target
entries the miner can ingest.

Bitbucket/GitLab adoption of these three SDKs is near-zero today, so in
practice every result is a github.com repo, but the query is forge-agnostic.
"""

from __future__ import annotations

import json
import re
import urllib.parse
import urllib.request
from dataclasses import dataclass

SOURCEGRAPH_STREAM = "https://sourcegraph.com/.api/search/stream"

SDK_QUERY = {
    "openai_agents": "from agents import function_tool",
    "claude_agent_sdk": "from claude_agent_sdk import tool",
    "google_adk": "from google.adk.tools import FunctionTool",
}

# Per-SDK default ref/paths. ref="main" is the modern default; paths=["."]
# means "scan the whole repo" since we don't know its layout.
DEFAULT_REF = "main"
DEFAULT_PATHS = ["."]

# Strip the github.com/ prefix; the miner clones via `git clone https://github.com/<owner>/<repo>.git`.
_GITHUB_PREFIX = "github.com/"


@dataclass(frozen=True)
class DiscoveredTarget:
    repo: str
    sdk: str
    forge: str  # "github" | "gitlab" | "bitbucket" | "other"

    def as_target_entry(self) -> dict:
        return {
            "repo": self.repo,
            "sdk": self.sdk,
            "paths": list(DEFAULT_PATHS),
            "ref": DEFAULT_REF,
        }


def discover(sdk: str, limit: int = 200, timeout: float = 60.0) -> list[DiscoveredTarget]:
    """Hit Sourcegraph for repos matching the SDK's import pattern."""
    if sdk not in SDK_QUERY:
        raise ValueError(f"unknown sdk {sdk!r}; expected one of {sorted(SDK_QUERY)}")
    query = f"{SDK_QUERY[sdk]} lang:python count:{limit} select:repo"
    url = f"{SOURCEGRAPH_STREAM}?q={urllib.parse.quote(query)}"
    req = urllib.request.Request(
        url,
        headers={
            "Accept": "text/event-stream",
            "User-Agent": "trustabl-rule-miner/0.1 (+https://github.com/trustabl/trustabl)",
        },
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body = resp.read().decode("utf-8", errors="replace")
    seen: set[str] = set()
    out: list[DiscoveredTarget] = []
    for repo in re.findall(r'"repository":"([^"]+)"', body):
        if repo in seen:
            continue
        seen.add(repo)
        forge, owner_repo = _split_forge(repo)
        if forge != "github" or not owner_repo:
            # Miner currently clones via github.com only; non-github skipped
            # but kept here so a future GitLab/Bitbucket clone path can pick
            # them up without rewriting discover().
            continue
        out.append(DiscoveredTarget(repo=owner_repo, sdk=sdk, forge=forge))
    return out


def discover_all(limit_per_sdk: int = 200) -> list[DiscoveredTarget]:
    out: list[DiscoveredTarget] = []
    for sdk in SDK_QUERY:
        try:
            out.extend(discover(sdk, limit=limit_per_sdk))
        except (urllib.error.URLError, TimeoutError) as exc:
            print(f"discover({sdk}): {exc}")
    return out


def _split_forge(repo: str) -> tuple[str, str]:
    if repo.startswith(_GITHUB_PREFIX):
        return "github", repo[len(_GITHUB_PREFIX):]
    if repo.startswith("gitlab.com/"):
        return "gitlab", repo[len("gitlab.com/"):]
    if repo.startswith("bitbucket.org/"):
        return "bitbucket", repo[len("bitbucket.org/"):]
    return "other", ""


def merge_into_targets(
    targets: list[dict], discovered: list[DiscoveredTarget]
) -> tuple[list[dict], int]:
    """Append discovered targets to `targets`, skipping repos already present.
    Returns (new_list, added_count).
    """
    existing = {t["repo"] for t in targets}
    added = 0
    out = list(targets)
    for d in discovered:
        if d.repo in existing:
            continue
        out.append(d.as_target_entry())
        existing.add(d.repo)
        added += 1
    return out, added
