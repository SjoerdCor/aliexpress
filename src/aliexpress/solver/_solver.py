"""Solver selection: pick the best available PuLP backend.

Temporary home while the CBC/HiGHS_CMD fallback is still in place (see ADR-0009).
Once empirically confirmed that ``pulp.HiGHS().available()`` is reliably True in the
target venv, this module disappears and both call sites use ``pulp.HiGHS(...)`` directly.
"""

import logging
import os
import tempfile

import pulp

logger = logging.getLogger(__name__)


def get_solver():
    """Return the best available PuLP solver with proven-optimum settings."""
    # gapRel=0 so we always get the proven optimum, not an early cutoff.
    # logPath goes to the OS temp dir so HiGHS never writes into the project root.
    log_path = os.path.join(tempfile.gettempdir(), "aliexpress-solver.log")
    kwargs = {"logPath": log_path, "msg": False, "gapRel": 0}
    if pulp.HiGHS(msg=False).available():
        return pulp.HiGHS(**kwargs)
    if pulp.HiGHS_CMD(msg=False).available():
        return pulp.HiGHS_CMD(**kwargs)
    logger.warning("Falling back to CBC solver. Might be very slow!")
    return pulp.PULP_CBC_CMD(**kwargs)
