"""Student-distribution solver sub-package.

The central sub-package is ``cpsat``: it exposes the public lifecycle
(``solve_with_fixed_balance`` / ``solve_within_minimal_relaxation``). Two modules
answer two distinct questions and sit either side of that boundary:

- ``satisfaction`` (outside ``cpsat``, shared with ``cpsat.model``): given a
  student's honored preferences, what is *their* satisfaction score? The metric —
  currently the concave integral of 0.5^x, so the first honored wish counts for
  more than the second.
- ``cpsat.strategies``: given every student's satisfaction, what single objective
  does the solver optimize? Total sum, or plateaud lexmaxmin (raise the lowest
  satisfaction level by level). A different question from the metric per student,
  answered independently of it.

The remaining ``cpsat`` modules:

- ``cpsat.model``: builds the CP-SAT model — assignment variables, hard constraints,
  and the per-student satisfaction lookup table (via ``satisfaction``).
- ``cpsat.engine``: orchestrates a solve — builds the model, fixes any lexicographic
  pre-stages, hands off to ``cpsat.strategies`` for the objective, and extracts the
  solution.
- ``cpsat.feasibility``: diagnoses which hard preference family must give when an
  instance is infeasible.
- ``cpsat._balance_families``: the six class-balance constraint families, hard or soft.
- ``cpsat.scaling``: the integer weight scale CP-SAT needs (it reasons over integers
  only; preference weights are user-entered floats).

Outside ``cpsat``, this package also holds:

- ``solutions``: reports on a solved distribution (``SolutionAnalyzer``).
- ``_balance``: ``GroupBalance``, ``STRICTEST_BALANCE``, and ``solver_log_path``,
  shared by ``cpsat`` and the reporting/web layer.
"""
