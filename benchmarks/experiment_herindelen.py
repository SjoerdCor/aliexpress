"""Herindelen scaling + symmetry experiments on the production solve path.

Runs main.py's exact herindelen path (optimize="lexmaxmin",
solve_within_minimal_relaxation, adaptive balance) on parameterized synthetic
instances, with a per-subsolve HiGHS time limit and an overall watchdog so
every run terminates on its own.

Usage: python benchmarks/experiment_herindelen.py <name> <n_groups> <nietin:0|1>
           <symbreak:0|1> <time_limit_s> [threads] [max_wall_s]

Instances (~22-24 students per destination group, 3 year groups, 2 stamgroepen
per year, equal boys/girls, same wish pattern as the realistic test):
  n_groups=2 -> 48 students, n_groups=3 -> 64, n_groups=4 -> 88.

symbreak=1 adds prefix symmetry breaking (only valid when nietin=0: with
"Niet in" wishes the groups are distinguishable and relabeling is not a
symmetry, so pruning it could cut off the optimum).

Appends one summary line per run to benchmarks/results.log.  Set BASELINE_SRC
to a frozen copy of ``src`` to measure old code instead of the working tree.
"""

import logging
import math
import os
import sys
import threading
import time
from pathlib import Path

import pandas as pd
import pulp

HERE = Path(__file__).parent
NAME = sys.argv[1]
N_GROUPS = int(sys.argv[2])
NIETIN = bool(int(sys.argv[3]))
SYMBREAK = bool(int(sys.argv[4]))
TIME_LIMIT = float(sys.argv[5])
THREADS = int(sys.argv[6]) if len(sys.argv) > 6 else None
MAX_WALL = float(sys.argv[7]) if len(sys.argv) > 7 else 2700.0

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    handlers=[
        logging.FileHandler(HERE / f"progress-{NAME}.log", mode="w", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
for noisy in ("matplotlib", "PIL", "asyncio"):
    logging.getLogger(noisy).setLevel(logging.WARNING)

sys.path.insert(0, os.environ.get("BASELINE_SRC") or str(HERE.parent / "src"))

# The aliexpress imports must come after the sys.path insert above: which code
# is imported (working tree vs frozen baseline) is exactly what it selects.
# pylint: disable=wrong-import-position
import aliexpress.solver._balance as balance_mod  # noqa: E402
import aliexpress.solver.feasibility as feas_mod  # noqa: E402
import aliexpress.solver.optimizationstrategies as strategies_mod  # noqa: E402
import aliexpress.solver.problemsolver as ps_mod  # noqa: E402
from aliexpress.solver.problemsolver import ProblemSolver  # noqa: E402

# pylint: enable=wrong-import-position

# Use the loaded code's solver class so repo runs keep the warm start;
# the frozen baseline has no WarmStartHiGHS and falls back to plain HiGHS.
WarmStartSolver = getattr(balance_mod, "WarmStartHiGHS", pulp.HiGHS)

log = logging.getLogger("experiment")

GROUP_NAMES = ["blauw", "rood", "geel", "groen"][:N_GROUPS]
GROUPS_TO = {g: {"Jongens": 0, "Meisjes": 0} for g in GROUP_NAMES}

# Students per gender per stamgroep, per year, scaled so each destination
# group receives ~22 students (the realistic class size).
PER_GENDER = {
    2: {6: 4, 7: 4, 8: 4},  # 16+16+16 = 48 -> 2 x 24
    3: {6: 5, 7: 6, 8: 5},  # 20+24+20 = 64 -> 3 x ~21
    4: {6: 7, 7: 8, 8: 7},  # 28+32+28 = 88 -> 4 x 22
}[N_GROUPS]

_solve_counter = {"n": 0}


def logged_solver() -> pulp.HiGHS:
    """get_solver's proven-optimum settings + forced HiGHS log + time limit."""
    _solve_counter["n"] += 1
    kwargs = {"threads": THREADS} if THREADS else {}
    return WarmStartSolver(
        msg=False,
        gapRel=0,
        timeLimit=TIME_LIMIT,
        output_flag=True,
        log_to_console=False,
        log_file=str(HERE / f"highs-{NAME}-{_solve_counter['n']:02d}.log"),
        **kwargs,
    )


ps_mod.get_solver = logged_solver
strategies_mod.get_solver = logged_solver
feas_mod.get_solver = logged_solver


def build_students() -> dict:
    """Synthetic students: 3 year groups x 2 stamgroepen x equal boys/girls."""
    students = {}
    for year, per_gender in PER_GENDER.items():
        for grp_letter in ("A", "B"):
            stamgroep = f"{year}{grp_letter}"
            for i in range(per_gender):
                min_sat = 0.01 if i == 0 else math.nan
                for prefix, gender in (("j", "Jongen"), ("m", "Meisje")):
                    students[f"{prefix}{stamgroep}_{i}"] = {
                        "Stamgroep": stamgroep,
                        "Jongen/meisje": gender,
                        "MinimaleTevredenheid": min_sat if prefix == "j" else math.nan,
                        "Jaarlaag": year,
                    }
    return students


def build_prefs(students: dict) -> pd.DataFrame:
    """Same deterministic wish pattern as the realistic test, parameterized."""
    keys = sorted(students.keys())
    n = len(keys)
    records = []
    for idx, leerling in enumerate(keys):
        n_pos = (idx % 5) + 1
        for k in range(n_pos):
            records.append(
                {
                    "Leerling": leerling,
                    "TypeWens": "Graag met",
                    "Nr": k + 1,
                    "Waarde": keys[(idx + k + 1) % n],
                    "Gewicht": 1.0,
                }
            )
        if idx % 5 == 2:
            # Post-negation convention: the solver sees negatives as "Graag met"
            # with a negative weight (datareader.toggle_negative_weights).
            records.append(
                {
                    "Leerling": leerling,
                    "TypeWens": "Graag met",
                    "Nr": n_pos + 1,
                    "Waarde": keys[(idx + n // 3) % n],
                    "Gewicht": -1.0,
                }
            )
        if NIETIN and idx % 8 == 0:
            records.append(
                {
                    "Leerling": leerling,
                    "TypeWens": "Niet in",
                    "Nr": 1,
                    "Waarde": GROUP_NAMES[idx % N_GROUPS],
                    "Gewicht": 1.0,
                }
            )
    df = pd.DataFrame(records).set_index(["Leerling", "TypeWens", "Nr"])
    df.columns.name = "TypeWaarde"
    return df


def build_not_together(students: dict) -> list[dict]:
    """Three niet-samen rules over existing students (mirrors the realistic test)."""
    keys = sorted(students.keys())
    return [
        {"group": set(keys[0:3]), "Max_aantal_samen": 2},
        {"group": set(keys[10:13]), "Max_aantal_samen": 2},
        # "max 1 of these 3 per group" needs at least 3 groups to be feasible.
        {"group": set(keys[20:23]), "Max_aantal_samen": 1 if N_GROUPS >= 3 else 2},
    ]


class SymBreakingSolver(ProblemSolver):
    """Adds prefix symmetry breaking over the (interchangeable) destination groups.

    Valid only when groups are truly interchangeable: identical occupancy and no
    group-targeting preferences.  Scheme: student i may only be placed in group j>0
    if some earlier student (in fixed name order) is in group j-1.  This admits
    exactly one representative per group relabeling.
    """

    def add_fundamental_constraints(self, prob):
        super().add_fundamental_constraints(prob)
        students = sorted(self.students)
        groups = list(self.groups_to)
        for i, student in enumerate(students):
            for j in range(1, len(groups)):
                earlier_in_prev = pulp.lpSum(
                    self.in_group[(students[k], groups[j - 1])] for k in range(i)
                )
                prob += (
                    self.in_group[(student, groups[j])] <= earlier_in_prev
                ), f"SymBreak_{student}_{groups[j]}"


def _watchdog():
    """Hard stop after MAX_WALL seconds: log a DNF line and kill the process."""
    time.sleep(MAX_WALL)
    summary = (
        f"{NAME}: groups={N_GROUPS} students=? nietin={int(NIETIN)} "
        f"symbreak={int(SYMBREAK)} threads={THREADS or 'default'} "
        f"subsolves={_solve_counter['n']} elapsed={MAX_WALL:.0f}s "
        f"outcome=DNF(watchdog: max wall time reached)"
    )
    log.error("RESULT %s", summary)
    with open(HERE / "results.log", "a", encoding="utf-8") as fh:
        fh.write(summary + "\n")
    os._exit(2)  # pylint: disable=protected-access  # threads may be stuck in C code


def main():
    """Run one benchmark instance and append its summary line to results.log."""
    threading.Thread(target=_watchdog, daemon=True).start()
    students = build_students()
    prefs = build_prefs(students)
    not_together = build_not_together(students)
    cls = SymBreakingSolver if SYMBREAK else ProblemSolver
    log.info(
        "RUN %s: %d groups, %d students, %d prefs, nietin=%s, symbreak=%s, limit=%.0fs/solve",
        NAME,
        N_GROUPS,
        len(students),
        len(prefs),
        NIETIN,
        SYMBREAK,
        TIME_LIMIT,
    )
    solver = cls(
        prefs,
        students,
        GROUPS_TO,
        not_together,
        groupbalance=None,
        optimize="lexmaxmin",
    )
    t0 = time.perf_counter()
    outcome = "OK"
    try:
        solver.solve_within_minimal_relaxation()
    # Any failure is a DNF data point (time limit -> non-Optimal status, or a
    # bug); the benchmark must record it and terminate rather than crash.
    except Exception as exc:  # pylint: disable=broad-exception-caught
        outcome = f"DNF({type(exc).__name__}: {exc})"
    elapsed = time.perf_counter() - t0
    summary = (
        f"{NAME}: groups={N_GROUPS} students={len(students)} nietin={int(NIETIN)} "
        f"symbreak={int(SYMBREAK)} threads={THREADS or 'default'} "
        f"subsolves={_solve_counter['n']} elapsed={elapsed:.1f}s outcome={outcome}"
    )
    log.info("RESULT %s", summary)
    with open(HERE / "results.log", "a", encoding="utf-8") as fh:
        fh.write(summary + "\n")


if __name__ == "__main__":
    main()
