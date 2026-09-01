"""Persists solve progress as ``progress.json`` in the process directory.

The web-layer half of the progress seam (see ``solver/progress.py``): a
:class:`ProgressWriter` is a :class:`ProgressListener` that turns each solver
event into an updated ``progress.json``, which ``/status`` reads and merges
into its poll response for the processing page. ``progress.json`` also carries
an ``estimate`` field (``{"seconds", "phase", "text"}``), a rough remaining-time
estimate recomputed after every stage/plateau event; see
docs/plan-processing-eta-gating.md for the three-phase estimator this feeds.
"""

import json
import math
import os
import time
from dataclasses import asdict
from datetime import datetime, timezone

from ..solver.progress import InputSummary, PlateauOutcome, ProgressListener

_PENDING_STEPS = ("floor", "balance", "satisfaction")

_PHASE_A_TEXT = (
    "Aan het rekenen… dit duurt meestal minder dan een minuut, soms enkele minuten."
)

# The post-performance measurements across all three modes put the ordinary
# satisfaction/balance ratios around 2-5.3. Six covers the observed upper end with a modest
# safety margin for the normal balance regime. Once balance itself exceeds a minute, its
# proof time is no longer a six-for-one predictor of satisfaction: the hard capped runs have
# a ratio below 1. See docs/metingen-processing-eta.md ("Herkalibratie na perf").
SATISFACTION_BALANCE_FACTOR = 6

# Keep the normal six-times estimate for the observed short/medium balance regime. Beyond
# that point, add balance seconds one-for-one; this captures the balance-dominated tail
# without turning a 9-minute satisfaction phase into a 75-minute ETA.
BALANCE_LINEAR_REGION_SECONDS = 60
LONG_BALANCE_FACTOR = 1

# The largest normal post-performance mode (Doorzetten) produced eleven rounds; using that
# upper observed budget keeps the first phase-C estimate on the roomy side for all modes.
# The lower-bound guard still prevents a zero ETA when a run has an unusual tail.
TYPICAL_ROUNDS = 11

# A normal reader only holds the destination open long enough to copy its bytes. One
# second is deliberately much longer than that window, while still surfacing genuine
# permission/ACL problems promptly instead of hanging the solver thread indefinitely.
_WINDOWS_REPLACE_TIMEOUT_SECONDS = 1.0
_WINDOWS_REPLACE_RETRY_SECONDS = 0.01


def _replace_snapshot(tmp_path: str, path: str) -> None:
    """Atomically publish a snapshot despite a transient Windows read handle.

    POSIX permits replacing a pathname while another process still has the previous
    inode open. Windows denies that replace unless every open handle shared delete
    access, which Python's regular ``open`` does not request. Pollers close their handle
    quickly, so retry that transient denial for a bounded period. The call only returns
    after the snapshot is published; a persistent permission error is re-raised rather
    than silently dropping the update.
    """
    if os.name != "nt":
        os.replace(tmp_path, path)
        return

    deadline = time.monotonic() + _WINDOWS_REPLACE_TIMEOUT_SECONDS
    while True:
        try:
            os.replace(tmp_path, path)
            return
        except PermissionError:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise
            time.sleep(min(_WINDOWS_REPLACE_RETRY_SECONDS, remaining))


def _format_remaining(seconds: float) -> str:
    """Render a remaining-time estimate as the user-facing "nog ~X" line.

    Rounds up (never down, so the estimate never looks worse than reality once revealed):
    under a minute to the nearest 10 seconds, from a minute on to whole minutes, with the
    correct Dutch singular/plural ("minuut" vs "minuten").
    """
    if seconds < 60:
        rounded_seconds = math.ceil(seconds / 10) * 10
        amount = f"~{rounded_seconds} seconden"
    else:
        minutes = math.ceil(seconds / 60)
        unit = "minuut" if minutes == 1 else "minuten"
        amount = f"~{minutes} {unit}"
    return f"naar verwachting nog {amount} (ruwe schatting)"


class ProgressWriter(ProgressListener):
    """Writes ``progress.json`` atomically after every solve event.

    Runs in the solver thread. Each write goes to a temp file in the same
    directory followed by an atomic replace. A transient open-reader denial is
    retried on Windows; the ``/status`` route therefore sees either the previous
    complete snapshot or the next complete snapshot, never a half-written file.
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
            "estimate": {"seconds": None, "phase": "a", "text": _PHASE_A_TEXT},
        }
        self._write()

    def stage_started(self, stage: str) -> None:
        self._state["steps"][stage] = "busy"
        self._write()

    def stage_finished(self, stage: str, seconds: float) -> None:
        self._state["steps"][stage] = "done"
        self._state["stage_seconds"].append({"label": stage, "seconds": seconds})
        self._recompute_estimate()
        self._write()

    def input_summary(self, summary: InputSummary) -> None:
        self._state["input_summary"] = asdict(summary)
        self._write()

    def plateau_finished(self, outcome: PlateauOutcome) -> None:
        # Whole percents; never clamped, since satisfaction can be negative (ADR-0014).
        # The list only ever grows, matching the "nothing ever disappears" rustregel —
        # the UI can safely re-render it in full each poll. Each round's wall-clock
        # ``seconds`` is kept per entry to feed the remaining-time estimator.
        self._state["plateaus"].append(
            {
                "min_pct": round(outcome.min_satisfaction * 100),
                "n_can_improve": outcome.n_can_improve,
                "seconds": outcome.seconds,
            }
        )
        self._recompute_estimate()
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
        _replace_snapshot(tmp_path, self.interim_result_path)
        self._state["interim_result_updated_at"] = datetime.now(
            timezone.utc
        ).isoformat()
        self._write()

    def _recompute_estimate(self) -> None:
        """Refresh ``self._state["estimate"]`` from the current stage/plateau history.

        Three phases (see docs/metingen-processing-eta.md and
        docs/plan-processing-eta-gating.md for the calibration): phase A (no number yet,
        balance not finished), phase B (balance finished, no round finished yet — six times
        the balance duration in the normal regime, with a one-for-one long-balance tail),
        and phase C (at
        least one round finished — a budget of typical-minus-done rounds times the
        *longest* round so far, since round durations tend to grow, not shrink, over a
        run). Called at the end of every event that could move the estimate.
        """
        plateaus = self._state["plateaus"]
        if plateaus:
            longest_round = max(plateau["seconds"] for plateau in plateaus)
            rounds_remaining = max(TYPICAL_ROUNDS - len(plateaus), 1)
            seconds = rounds_remaining * longest_round
            phase = "c"
        else:
            balance_entry = next(
                (
                    entry
                    for entry in self._state["stage_seconds"]
                    if entry["label"] == "balance"
                ),
                None,
            )
            if balance_entry is None:
                self._state["estimate"] = {
                    "seconds": None,
                    "phase": "a",
                    "text": _PHASE_A_TEXT,
                }
                return
            seconds = _phase_b_seconds(balance_entry["seconds"])
            phase = "b"

        self._state["estimate"] = {
            "seconds": seconds,
            "phase": phase,
            "text": _format_remaining(seconds),
        }

    def _write(self) -> None:
        tmp_path = f"{self.path}.tmp"
        with open(tmp_path, "w", encoding="utf-8") as fh:
            json.dump(self._state, fh, ensure_ascii=False)
        _replace_snapshot(tmp_path, self.path)


def _phase_b_seconds(balance_seconds: float) -> float:
    """Estimate phase-B satisfaction time from the completed balance duration.

    Balance duration is a useful predictor for ordinary runs, but the post-performance
    stress measurements show a different regime once balance proof itself takes more than
    about a minute. The piecewise continuation keeps the estimate conservative without
    multiplying the balance-dominated tail by six again.
    """
    normal_balance = min(balance_seconds, BALANCE_LINEAR_REGION_SECONDS)
    long_balance = max(balance_seconds - BALANCE_LINEAR_REGION_SECONDS, 0)
    return (
        SATISFACTION_BALANCE_FACTOR * normal_balance
        + LONG_BALANCE_FACTOR * long_balance
    )
