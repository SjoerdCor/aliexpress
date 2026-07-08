"""Flask blueprints for the wizard, roster, results, processes, admin and auth flows.

Architecture rule for modules in this package: route modules do only HTTP — read the
request, call helpers, render/redirect/flash. Form parsing lives in
``data/form_parsers.py``, per-process file persistence in ``web/process_files.py``,
background work (solving, sociogram generation) in ``web/tasks.py``, and display/
ordering helpers in ``web/display.py``. Route modules never import from each other.

Small route glue stays in the route module on purpose — it is routing logic, not a
layer violation. Examples: a dispatch helper that picks which parsing/handling path to
take based on the request, or a wrapper that chains a parser's output into a
persistence call. What must not live here: the parsing logic itself, knowledge of
on-disk file names, or thread/background-task orchestration.
"""
