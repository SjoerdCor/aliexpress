"""CP-SAT model builder for the student distribution problem.

OR-Tools CP-SAT reasons over integers only. This subpackage translates the
float-valued parts of the problem (preference weights, the satisfaction metric)
into exact integer form; the float world stays at the input and reporting side.

Runtime: most solves finish in seconds. On harder instances — many students,
tight balance, Niet-samen rules that couple students across different
Stamgroepen/Jaarlagen rather than within one — it can take several minutes;
around 90 students has been measured up to 10-15 minutes. CP-SAT's
``num_workers`` setting races several search strategies in parallel threads,
so the proof itself is deterministic but the wall-clock time to reach it is
not. It always terminates with a proven answer, though, however long it takes.
"""
