"""Student-distribution solver sub-package.

The central module is ``problemsolver``: it owns the LP model and the public lifecycle
(``run`` / ``solve_within_minimal_relaxation`` / ``extract_solution``). It delegates to:

- ``satisfaction``: maps honored preferences to a satisfaction score per student (the
  metric — currently the concave integral of 0.5^x).
- ``optimizationstrategies``: aggregates per-student satisfaction into a single objective
  (total, minimum, or plateaud lexmaxmin).
- ``feasibility``: reasons about solvability by building disposable LPs over the
  ``ProblemSolver`` substrate.

Helper modules shared by the above:

- ``pulp_thresholds``: generic big-M indicator constraints.
- ``_balance``: ``GroupBalance``, ``STRICTEST_BALANCE``, and ``get_solver()``.
"""
