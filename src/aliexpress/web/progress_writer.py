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

    def __init__(self, path: str):
        self.path = path
        self.started_at = datetime.now(timezone.utc).isoformat()
        self.steps = {stage: "pending" for stage in _PENDING_STEPS}
        self.stage_seconds: list[dict] = []
        self.input_summary_data: dict | None = None
        self._write()

    def stage_started(self, stage: str) -> None:
        self.steps[stage] = "busy"
        self._write()

    def stage_finished(self, stage: str, seconds: float) -> None:
        self.steps[stage] = "done"
        self.stage_seconds.append({"label": stage, "seconds": seconds})
        self._write()

    def input_summary(self, summary: InputSummary) -> None:
        self.input_summary_data = asdict(summary)
        self._write()

    def _write(self) -> None:
        payload = {
            "input_summary": self.input_summary_data,
            "steps": self.steps,
            "stage_seconds": self.stage_seconds,
            "started_at": self.started_at,
        }
        tmp_path = f"{self.path}.tmp"
        with open(tmp_path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False)
        os.replace(tmp_path, self.path)
