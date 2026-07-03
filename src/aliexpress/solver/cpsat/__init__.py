"""CP-SAT model builder for the student distribution problem.

OR-Tools CP-SAT reasons over integers only. This subpackage translates the
float-valued parts of the problem (preference weights, the satisfaction metric)
into exact integer form; the float world stays at the input and reporting side.
"""
