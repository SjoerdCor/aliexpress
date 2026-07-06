"""Student-distribution solver: builds a CP-SAT (OR-Tools) model and solves it.

The public lifecycle (``solve_with_fixed_balance`` / ``solve_within_minimal_relaxation``)
lives in ``engine``. Two modules answer two distinct questions and sit either side of
that boundary:

- ``satisfaction``: given a student's honored preferences, what is *their* satisfaction
  score? The metric — currently the concave integral of 0.5^x, so the first honored wish
  counts for more than the second.
- ``strategies``: given every student's satisfaction, what single objective does the
  solver optimize? Total sum, or plateaud lexmaxmin (raise the lowest satisfaction level
  by level). A different question from the metric per student, answered independently
  of it.

The remaining modules:

- ``modelbuilder``: builds the CP-SAT model — assignment variables, hard constraints, and
  the per-student satisfaction lookup table (via ``satisfaction``). CP-SAT reasons over
  integers only; preference weights (user-entered floats) and the satisfaction metric
  are scaled to exact integers before entering the model (see ``scaling``).
- ``engine``: orchestrates a solve — builds the model, fixes any lexicographic
  pre-stages, hands off to ``strategies`` for the objective, and extracts the solution.
- ``feasibility``: diagnoses which hard preference family must give when an instance is
  infeasible.
- ``_balance_families``: the six class-balance constraint families, hard or soft.
- ``scaling``: the integer weight scale CP-SAT needs.
- ``results``: maps a solved instance into the shared, solver-agnostic
  ``SolutionResult``/``GroupComposition``.
- ``solutions``: reports on a solved distribution (``SolutionAnalyzer``).
- ``_balance``: ``GroupBalance`` and ``STRICTEST_BALANCE``, shared with the reporting/web
  layer.

Runtime: most solves finish in seconds. On harder instances — many students, tight
balance, Niet-samen rules that couple students across different Stamgroepen/Jaarlagen
rather than within one — it can take several minutes; around 90 students has been
measured up to 10-15 minutes. CP-SAT's ``num_workers`` setting races several search
strategies in parallel threads, so the proof itself is deterministic but the wall-clock
time to reach it is not. It always terminates with a proven answer, though, however long
it takes.
"""
