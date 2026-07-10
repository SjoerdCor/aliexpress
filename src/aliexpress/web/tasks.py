"""Background thread orchestration: solving and sociogram generation.

Each function here runs in its own thread, spawned by the wizard route
``start_distribution`` (which stays responsible for *that* threads start; this module
determines *what* they do).
"""

import json
import logging
from dataclasses import asdict, dataclass
from typing import Any

import pandera as pa

from .. import sociogram
from ..errors import (
    CouldNotReadFileError,
    FeasibilityError,
    SolverError,
    ValidationError,
)
from ..logging_config import bind_log_context
from ..main import distribute_students_from_data
from .extensions import db
from .models import LogLine, Process
from .process_files import load_groups, load_voorkeuren
from .storage import get_file_path
from .validation_messages import to_validation_message

logger = logging.getLogger(__name__)


@dataclass
class ThreadContext:
    """Shared context passed to background solver/sociogram threads.

    Bundles the Flask app object (needed to open a thread-local app context) with the
    process identifiers required to locate files and append log lines.
    """

    app_obj: Any
    school_id: str
    process_name: str
    run_id: int


def _write_result_files(school_id, process_name, result):
    """Persist the solver output as files in the process dir.

    Writes the download workbook (``results.xlsx``), the three analysis tables as HTML
    (``result_tables.json``) and the structured group-card view-model
    (``groepsindeling_view.json``, from :class:`GroepsindelingView`). Written before the status
    flips to "done" so the result page never polls ahead of the files it needs.
    """
    with open(get_file_path(school_id, process_name, "results.xlsx"), "wb") as fh:
        fh.write(result["download"].getbuffer())
    tables = {name: df.to_html(na_rep="") for name, df in result["dataframes"].items()}
    with open(
        get_file_path(school_id, process_name, "result_tables.json"),
        "w",
        encoding="utf-8",
    ) as fh:
        json.dump(tables, fh, ensure_ascii=False)
    with open(
        get_file_path(school_id, process_name, "groepsindeling_view.json"),
        "w",
        encoding="utf-8",
    ) as fh:
        json.dump(asdict(result["groepsindeling_view"]), fh, ensure_ascii=False)


def _handle_failure(exc, school_id, process_name):
    file_reading_errs = (
        pa.errors.SchemaError,
        ValidationError,
        CouldNotReadFileError,
    )
    if isinstance(exc, file_reading_errs):
        log_msg = "Files are incorrect"
    elif isinstance(exc, FeasibilityError):
        log_msg = "Problem is infeasible"
    elif isinstance(exc, SolverError):
        log_msg = "Solver could not solve the problem"
    else:
        log_msg = "Uncaught exception"
    logger.exception(log_msg)
    Process.by_name(school_id, process_name).run.set_status(
        "error", to_validation_message(exc)
    )


def run_solve_thread(ctx: ThreadContext, not_together):
    """Background thread: run the solver and write result artifacts.

    Each call creates its own app context and DB session. ``ctx.run_id`` is the integer
    PK of the Run row so log lines can be appended without a school+name query per line.
    Reads preferences from ``voorkeuren.json`` (written by both input paths) so that the
    solver is independent of the original file format. Likewise loads the destination
    groups itself via ``process_files.load_groups`` rather than taking a file path.
    """

    def on_update(message):
        db.session.add(LogLine(run_id=ctx.run_id, text=message))
        db.session.commit()

    with ctx.app_obj.app_context():
        with bind_log_context(
            school=ctx.school_id,
            process=ctx.process_name,
            run=str(ctx.run_id),
            phase="solve",
        ):
            try:  # pylint: disable=broad-exception-caught
                Process.by_name(ctx.school_id, ctx.process_name).run.set_status(
                    "running"
                )
                preference_data, _ = load_voorkeuren(ctx.school_id, ctx.process_name)
                target_groups = load_groups(ctx.school_id, ctx.process_name)
                result = distribute_students_from_data(
                    preference_data,
                    target_groups,
                    not_together,
                    on_update=on_update,
                )
                logger.info("Distributing students finished successfully")
                # Write artifacts before flipping to "done" so the result page never
                # races ahead of the files it needs.
                _write_result_files(ctx.school_id, ctx.process_name, result)
                Process.by_name(ctx.school_id, ctx.process_name).run.set_status("done")
            except Exception as exc:  # pylint: disable=broad-exception-caught
                _handle_failure(exc, ctx.school_id, ctx.process_name)


def create_sociogram_thread(ctx: ThreadContext):
    """Background thread: build and write the Plotly sociogram HTML.

    Runs concurrently with the solver; log lines are appended via ``ctx.run_id`` just
    like the solver thread does. Reads preferences from ``voorkeuren.json`` (written by
    both input paths) via ``SociogramMaker.from_preference_data``, so the sociogram is
    available for both the Excel and web-form input paths.
    """

    def on_update(message):
        db.session.add(LogLine(run_id=ctx.run_id, text=message))
        db.session.commit()

    with ctx.app_obj.app_context():
        with bind_log_context(
            school=ctx.school_id,
            process=ctx.process_name,
            run=str(ctx.run_id),
            phase="sociogram",
        ):
            try:  # pylint: disable=broad-exception-caught
                on_update("Sociogram tekenen...")
                preference_data, _ = load_voorkeuren(ctx.school_id, ctx.process_name)
                sg = sociogram.SociogramMaker.from_preference_data(preference_data)
                fig, g, pos = sg.plot_sociogram()
                logger.info("Sociogram created")
                fig = sociogram.networkx_to_plotly(g, pos)
                html = fig.to_html(full_html=False, include_plotlyjs="cdn")
                logger.info("HTML created")
                with open(
                    get_file_path(ctx.school_id, ctx.process_name, "sociogram.html"),
                    "w",
                    encoding="utf-8",
                ) as fh:
                    fh.write(html)
                on_update(
                    '<a href=/sociogram target="_blank" class="button">'
                    "Bekijk het sociogram nu!</a>"
                )
            except Exception:  # pylint: disable=broad-exception-caught
                logger.exception("Could not create sociogram")
