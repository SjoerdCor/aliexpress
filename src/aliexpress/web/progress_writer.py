"""Persists solve progress as ``progress.json`` in the process directory.

The web-layer half of the progress seam (see ``solver/progress.py``): a
:class:`ProgressWriter` is a :class:`ProgressListener` that turns each solver
event into an updated ``progress.json``, which ``/status`` reads and merges
into its poll response for the processing page.
"""

import json
import os
from dataclasses import asdict
from datetime import datetime, timezone

from ..solver.progress import InputSummary, ProgressListener

_PENDING_STEPS = ("floor", "balance", "satisfaction")


class ProgressWriter(ProgressListener):
    """Writes ``progress.json`` atomically after every solve event.

    Runs in the solver thread. Each write goes to a temp file in the same
    directory followed by ``os.replace``, which is atomic on both POSIX and
    Windows — the ``/status`` route can never read a half-written file, even
    if it polls mid-write.
    """

    def __init__(self, path: str, interim_result_path: str | None = None):
        self.path = path
        # Separate file (mirroring groepsindeling_view.json): the interim view is much
        # larger than the rest of progress.json, and it changes on a different cadence
        # (once per stage boundary vs. every solver event), so it is kept out of the
        # lean progress.json payload — only its timestamp goes there.
        self.interim_result_path = interim_result_path
        # The exact progress.json payload, built once and mutated in place by each
        # event; keeping it as one dict (rather than one attribute per field) is what
        # keeps this class's state small regardless of how many fields progress.json
        # grows to carry.
        self._state = {
            "input_summary": None,
            "steps": {stage: "pending" for stage in _PENDING_STEPS},
            "plateaus": [],
            "tiebreak_busy": False,
            "stage_seconds": [],
            "started_at": datetime.now(timezone.utc).isoformat(),
            "interim_result_updated_at": None,
        }
        self._write()

    def stage_started(self, stage: str) -> None:
        self._state["steps"][stage] = "busy"
        self._write()

    def stage_finished(self, stage: str, seconds: float) -> None:
        self._state["steps"][stage] = "done"
        self._state["stage_seconds"].append({"label": stage, "seconds": seconds})
        self._write()

    def input_summary(self, summary: InputSummary) -> None:
        self._state["input_summary"] = asdict(summary)
        self._write()

    def plateau_finished(self, min_satisfaction: float, n_can_improve: int) -> None:
        # Whole percents per the grilling decision; never clamped, since satisfaction
        # can be negative (ADR-0014). The list only ever grows, matching the "nothing
        # ever disappears" rustregel — the UI can safely re-render it in full each poll.
        self._state["plateaus"].append(
            {"min_pct": round(min_satisfaction * 100), "n_can_improve": n_can_improve}
        )
        self._write()

    def tiebreak_started(self) -> None:
        self._state["tiebreak_busy"] = True
        self._write()

    def interim_result_view(self, view) -> None:
        """Persist the translated interim view to ``interim_result.json``.

        Written to a separate file (not into ``progress.json``) so the lean progress
        payload every poll re-reads stays small; ``progress.json`` only records a fresh
        timestamp, which the processing page uses to detect a new view is available. No
        damping: called on every stage boundary the adapter forwards.
        """
        tmp_path = f"{self.interim_result_path}.tmp"
        with open(tmp_path, "w", encoding="utf-8") as fh:
            json.dump(asdict(view), fh, ensure_ascii=False)
        os.replace(tmp_path, self.interim_result_path)
        self._state["interim_result_updated_at"] = datetime.now(
            timezone.utc
        ).isoformat()
        self._write()

    def _write(self) -> None:
        tmp_path = f"{self.path}.tmp"
        with open(tmp_path, "w", encoding="utf-8") as fh:
            json.dump(self._state, fh, ensure_ascii=False)
        os.replace(tmp_path, self.path)
