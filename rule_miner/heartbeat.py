"""Periodic stderr heartbeat so long scans don't look hung.

Emits one line every `interval` seconds with elapsed time, the target
currently being processed, and cumulative tool counts. Runs on a
background thread; the main thread updates state via setters. Stop with
`stop()` to flush the final tick and join the thread.
"""

from __future__ import annotations

import dataclasses
import sys
import threading
import time
from typing import TextIO


@dataclasses.dataclass
class _State:
    current_target: str = "(starting)"
    targets_total: int = 0
    targets_done: int = 0
    tools_total: int = 0
    agents_total: int = 0
    subagents_total: int = 0


class Heartbeat:
    def __init__(self, interval: float = 60.0, stream: TextIO = sys.stderr):
        self.interval = interval
        self.stream = stream
        self._state = _State()
        self._stop = threading.Event()
        self._thread = threading.Thread(
            target=self._loop, name="rule-miner-heartbeat", daemon=True
        )
        self._start_ts: float | None = None

    # ── lifecycle ────────────────────────────────────────────────────────
    def start(self, targets_total: int) -> None:
        self._state.targets_total = targets_total
        self._start_ts = time.monotonic()
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread.is_alive():
            self._thread.join(timeout=self.interval + 1)
        self._emit("done")

    # ── state updates (called from main thread) ──────────────────────────
    def set_target(self, name: str) -> None:
        self._state.current_target = name

    def target_done(self) -> None:
        self._state.targets_done += 1

    def add_counts(
        self, tools: int = 0, agents: int = 0, subagents: int = 0
    ) -> None:
        self._state.tools_total += tools
        self._state.agents_total += agents
        self._state.subagents_total += subagents

    # ── internals ────────────────────────────────────────────────────────
    def _loop(self) -> None:
        # First tick after one full interval, not immediately.
        while not self._stop.wait(self.interval):
            self._emit("tick")

    def _emit(self, kind: str) -> None:
        elapsed = (
            int(time.monotonic() - self._start_ts) if self._start_ts else 0
        )
        s = self._state
        line = (
            f"[heartbeat {kind} t={elapsed}s] "
            f"target {s.targets_done}/{s.targets_total} "
            f"({s.current_target}) -- "
            f"tools={s.tools_total} agents={s.agents_total} "
            f"subagents={s.subagents_total}"
        )
        print(line, file=self.stream, flush=True)
