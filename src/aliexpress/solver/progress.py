"""Flask-free progress-reporting seam for the solve pipeline.

The solver package stays web-free: it only knows about a :class:`ProgressListener`
with a no-op default, so callers that don't care about progress (the CLI, most
tests) need pass nothing. The web layer subclasses it to persist progress as
JSON for the processing page to poll (see ``web/progress_writer.py``).

``stage`` is one of the three UI steps shown on the processing page:

- ``"floor"`` — stage 1, finding the minimal relaxation floor (non-positive
  satisfaction count).
- ``"balance"`` — stage 2, minimizing the weighted balance relaxation within
  that floor.
- ``"satisfaction"`` — stage 3, maximizing satisfaction (all lexmaxmin levels
  and the tie-break) within the fixed balance.
"""


class ProgressListener:
    """No-op default: subclass and override to observe solve progress."""

    def stage_started(self, stage: str) -> None:
        """Called when ``stage`` begins."""

    def stage_finished(self, stage: str, seconds: float) -> None:
        """Called when ``stage`` completes, after ``seconds`` seconds."""
