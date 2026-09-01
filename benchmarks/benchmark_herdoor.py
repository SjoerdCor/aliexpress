"""Time the stored 72-student herindelen-with-forwarding stress scenario.

The benchmark uses the solver's production data contracts and prints aggregate
timings only. It does not write results or interim distributions back to the
stored process.
"""

import argparse
import json
import time
from collections import Counter
from pathlib import Path

from aliexpress.data import datareader
from aliexpress.data.preferences_data import PreferenceData
from aliexpress.solver import engine, strategies
from aliexpress.solver._balance import BalanceMaxima
from aliexpress.solver.progress import ProgressListener


class Timer(ProgressListener):
    """Print timing events emitted by the production solve pipeline."""

    def stage_started(self, stage):
        print(f"STAGE start {stage}", flush=True)

    def stage_finished(self, stage, seconds):
        print(f"STAGE done  {stage} {seconds:.2f}s", flush=True)

    def plateau_finished(self, outcome):
        print(
            f"PLATEAU min={outcome.min_satisfaction:.6f} "
            f"above={outcome.n_can_improve} {outcome.seconds:.2f}s",
            flush=True,
        )

    def tiebreak_started(self):
        print("TIEBREAK start", flush=True)


def _load_scenario(base: Path):
    payload = json.loads((base / "voorkeuren.json").read_text(encoding="utf-8"))
    payload.pop("source", None)
    preference_data = PreferenceData.from_json(json.dumps(payload))
    target_groups = datareader.read_groups_excel(base / "groups.xlsx")

    raw_rules = json.loads((base / "not_together.json").read_text(encoding="utf-8"))
    not_together = [
        {
            "group": {datareader.matching_key(name) for name in rule["group"]},
            "Max_aantal_samen": rule["Max_aantal_samen"],
        }
        for rule in raw_rules
    ]
    maxima = BalanceMaxima(
        **json.loads((base / "balance_limits.json").read_text(encoding="utf-8"))
    )
    return preference_data, target_groups, not_together, maxima


def _install_substage_timer():
    original_solve_stage = strategies.solve_stage

    def timed_solve_stage(model, label, **objective):
        started = time.perf_counter()
        solver = original_solve_stage(model, label, **objective)
        print(
            f"SUBSTAGE {label}: {time.perf_counter() - started:.2f}s "
            f"objective={solver.ObjectiveValue():.0f} "
            f"bound={solver.BestObjectiveBound():.0f}",
            flush=True,
        )
        return solver

    strategies.solve_stage = timed_solve_stage


def run(base: Path, repetitions: int) -> None:
    """Run the exact production solve ``repetitions`` times."""
    preference_data, target_groups, not_together, maxima = _load_scenario(base)
    print(
        f"BENCHMARK students={len(preference_data.students_info)} "
        f"groups={len(target_groups.counts)} repetitions={repetitions} "
        f"workers={strategies.NUM_WORKERS}",
        flush=True,
    )
    _install_substage_timer()

    for run_number in range(1, repetitions + 1):
        print(f"RUN {run_number} start", flush=True)
        started = time.perf_counter()
        solution = engine.solve_within_minimal_relaxation(
            preferences=preference_data.preferences,
            students=preference_data.students_info,
            groups_to=target_groups.counts,
            not_together=not_together,
            optimize="lexmaxmin",
            listener=Timer(),
            maxima=maxima,
        )
        elapsed = time.perf_counter() - started
        levels = Counter(
            round(value, 6) for value in solution.student_satisfaction.values()
        )
        print(f"RUN {run_number} total {elapsed:.2f}s", flush=True)
        print(f"RUN {run_number} satisfaction {sorted(levels.items())}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scenario",
        type=Path,
        default=Path("instance/storage/testschool/herdoor"),
    )
    parser.add_argument("--repetitions", type=int, default=1)
    arguments = parser.parse_args()
    run(arguments.scenario, arguments.repetitions)
