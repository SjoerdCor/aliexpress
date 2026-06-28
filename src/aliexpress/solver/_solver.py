"""Solver configuration: HiGHS backend with proven-optimum settings."""

import os
import tempfile

import pulp


def get_solver():
    """Return the HiGHS PuLP solver with proven-optimum settings."""
    # gapRel=0 so we always get the proven optimum, not an early cutoff.
    # logPath goes to the OS temp dir so HiGHS never writes into the project root.
    log_path = os.path.join(tempfile.gettempdir(), "aliexpress-solver.log")
    return pulp.HiGHS(logPath=log_path, msg=False, gapRel=0)
